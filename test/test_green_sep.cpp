#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "fbr/fbr_gs.h"
#include "fbr/green_overlap.h"
#include "fbr/initial_state.h"

#include <vector>

using namespace arma;
using namespace fbr;

// Green function of the spinless IRLM with the two states evolved in SEPARATE
// frames (one Fbr_dyn each) and aligned only at the measurement
// (green_overlap.h). Two independent checks at U=0, where the answer is exact:
//
//   1) "vs star": G computed by aligning B into A's frame must equal G computed
//      by aligning BOTH states to the fixed star frame and contracting there.
//      The common basis is arbitrary, so the two must agree; they exercise
//      different relative rotations, so agreement is a real test of the alignment.
//   2) "vs exact": at U=0 the Green function is the non-interacting
//      G(i,j,t) = -i sum_{a unocc} e^{-i e_a t} V_ia V_ja.

namespace {

std::pair<Fb_mps<cmpx>, double> addParticle(Fb_mps<cmpx> const &psi0, int j)
{
    auto state = psi0;
    state.apply_local_op("Cdag", j);
    double nrm = std::sqrt(std::real(itensor::innerC(state.psi, state.psi)));
    state.psi.normalize();
    state.update_cc();
    return {state, nrm};
}

// <c_i^dag A | B> with BOTH states aligned to the fixed frame `common`.
cmpx cElementIn(Fb_mps<cmpx> const &A, Fb_mps<cmpx> const &B, int i, cx_mat const &common)
{
    auto Ac = A; align_to_frame(Ac, common);
    auto Bc = B; align_to_frame(Bc, common);
    auto Ai = Ac; Ai.apply_local_op("Cdag", i);
    return itensor::innerC(Ai.psi, Bc.psi);
}

} // namespace

TEST_CASE("separate-frame green: B->A agrees with the star frame and matches exact at U=0",
          "[green_sep]")
{
    const int L = 24, n_part = L / 2, nStep = 12;
    const double V = 0.5, dt = 0.05;

    mat K(L, L, fill::zeros);
    for (int i = 1; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    K(0, 1) = K(1, 0) = V;                       // U = 0

    vec ek_exact; mat evec_exact;
    eig_sym(ek_exact, evec_exact, K);
    auto G_exact = [&](int i, int j, double t) {
        cmpx g = 0;
        for (int a = n_part; a < L; a++)
            g += std::exp(-imag_1 * ek_exact[a] * t) * evec_exact(i, a) * evec_exact(j, a);
        return -imag_1 * g;
    };

    ImpurityParam model{.Kmat = K, .imp_pos = {0, 1}};
    model.to_star();
    cx_mat star = model.rot * cmpx(1, 0);        // fixed common frame (identity on impurity)

    // ground state
    auto gs = Fb_mps<double>::from_slater(model.rot, vec{model.Kmat.diag()},
                                          n_part, model.n_imp(), leading);
    gs.tol = 1e-12;
    auto gsSolver = Fbr_gs(model, gs);
    for (int i = 0; i < 40; i++) gsSolver.iterate({.max_bond_dim = 128});
    REQUIRE(std::abs(gsSolver.energy - sum(ek_exact.head(n_part))) < 1e-6);

    auto psi0 = gsSolver.fb.to_complex();
    psi0.tol = 1e-12;

    struct Run { Fbr_dyn A, B; double nrm; };
    auto runFor = [&](int j) {
        auto [Bj, nrm] = addParticle(psi0, j);
        Bj.tol = psi0.tol;
        return Run{Fbr_dyn(model, psi0, dt), Fbr_dyn(model, Bj, dt), nrm};
    };
    auto run0 = runFor(0);
    auto run1 = runFor(1);

    double devStar = 0, devExact = 0, devLoose = 0;
    int maxActive = 0;
    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;
        for (auto const &run : {std::cref(run0), std::cref(run1)}) {
            int i = 0, j = (&run.get() == &run0) ? 0 : 1;

            cmpx cBA   = c_element(run.get().A.fb, run.get().B.fb, i);            // B -> A, exact
            cmpx cStar = cElementIn(run.get().A.fb, run.get().B.fb, i, star);     // both -> star
            cmpx cLoose= c_element(run.get().A.fb, run.get().B.fb, i, 1e-4);      // loose contraction
            devStar = std::max(devStar, std::abs(cBA - cStar));
            devLoose = std::max(devLoose, std::abs(cBA - cLoose));

            cmpx G     = -imag_1 * run.get().nrm * cBA;
            devExact = std::max(devExact, std::abs(G - G_exact(i, j, t)));

            maxActive = std::max({maxActive, run.get().A.fb.n_active(),
                                  run.get().B.fb.n_active()});
        }
        if (step < nStep) {
            run0.A.iterate({.epsilon_M = 0}); run0.B.iterate({.epsilon_M = 0});
            run1.A.iterate({.epsilon_M = 0}); run1.B.iterate({.epsilon_M = 0});
        }
    }

    INFO("max |G(B->A) - G(star)|      = " << devStar);
    INFO("max |G - G_exact|            = " << devExact);
    INFO("max |exact - loose(1e-4)|    = " << devLoose);
    INFO("largest active window        = " << maxActive << " of L=" << L);
    REQUIRE(devStar < 1e-9);          // the common basis is arbitrary
    REQUIRE(devExact < 1e-4);         // exact at U=0
    REQUIRE(devLoose < 1e-4);         // a loose circuit contraction is enough
    REQUIRE(maxActive <= 8);          // each state keeps its own small window
}
