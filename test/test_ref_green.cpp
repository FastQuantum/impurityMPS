#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "fbr/fbr_gs.h"
#include "test_ref_common.h"

#include <map>
#include <string>
#include <vector>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

// The impurity Green functions of the spinless IRLM, computed with the FBR
// (active-window, star geometry, interaction picture) and compared against the
// trusted chain baseline in example/ref/output/chain_green_irlm_U<U>_ref.txt,
// produced by example/ref/chain_green_irlm.cpp.
//
//     G(i,j,t) = -i <psi0| c_i(t) c_j^dag(0) |psi0> = -i <c_i^dag A(t) | B_j(t)>
//
// with A=|psi0> and B_j=c_j^dag|psi0> evolved together in one common orbital
// basis by Fbr_ns_dyn. The impurity orbitals are never rotated, so the
// real-space site i is the MPS orbital i and c_i is local there.

namespace {

// Same model as example/ref/chain_green_irlm.cpp and example/fbr_dyn_irlm.cpp.
Impurity makeModel(int L, double U, double V)
{
    mat K(L, L, fill::zeros);
    for (int i = 1; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    K(0, 1) = K(1, 0) = V;
    K(0, 0) = K(1, 1) = -U / 2;
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    return Impurity{{.Kmat = K, .Umat = Umat, .impPos = {0, 1}}};
}

// c_j^dag|psi0>, normalized, with the norm it had before normalizing: the states
// of one Fbr_ns_dyn share their Slater determinant, so they have to be
// normalized, and the norm goes back into G afterwards.
std::pair<Fb_mps<cmpx>, double> addParticle(Fb_mps<cmpx> const &psi0, int j)
{
    auto state = psi0;
    state.applyLocalOp("Cdag", j);
    double nrm = std::sqrt(std::real(itensor::innerC(state.psi, state.psi)));
    state.psi.normalize();
    state.update_cc();
    return {state, nrm};
}

cmpx cElement(Fb_mps<cmpx> const &A, Fb_mps<cmpx> const &B, int i)
{
    auto Ai = A;
    Ai.applyLocalOp("Cdag", i);
    return itensor::innerC(Ai.psi, B.psi);
}

// A bound per time window, as chainTol() does for the correlator references.
// Each entry covers the steps from the previous entry's time up to its own, so
// the windows are disjoint: {20.0,...} bounds 10 < t <= 20 only.
struct GreenTol { double t; double tol; };

// Measured against the committed reference, at ~2x above what is observed.
// Worse of U=0.1 and U=0.2, per window: 7.0e-5 for t<=5, 7.1e-5 for 5<t<=10,
// 2.7e-5 for 10<t<=20. The two methods do not drift apart with time -- the last
// window is the smallest because both Green functions have decayed by then
// (|G01| falls from 0.29 to ~0.12), so the absolute difference shrinks with the
// signal. At t=0 they agree to 5e-8, which the separate bound below covers.
std::vector<GreenTol> greenTol()
{
    return {{5.0, 1.5e-4}, {10.0, 1.5e-4}, {20.0, 6e-5}};
}

struct GreenError {
    std::map<double, double> byBucket;   ///< bucket time -> max deviation in it
    double atZero = 0;                   ///< t=0: G only involves the ground state
    double tMax = 0;
};

GreenError const &resultFor(std::string const &us)
{
    static std::map<std::string, GreenError> cache;
    auto it = cache.find(us);
    if (it != cache.end()) return it->second;

    constexpr int L = 100;
    constexpr double dt = 0.1;
    constexpr double V = 0.1;
    double U = std::stod(us);

    auto ref = loadGreenReference("chain_green_irlm_U" + us + "_ref.txt");
    // the reference rows are compared step by step, so they have to be on the
    // same time grid as the evolution below
    REQUIRE(ref.size() > 1);
    REQUIRE(std::abs((ref[1].t - ref[0].t) - dt) < 1e-12);

    auto model = makeModel(L, U, V);

    auto gs = Fb_mps<double>::from_slater(model.param.rot, vec{model.param.Kmat.diag()},
                                          model.param.nPart(), model.param.nImp(), leading);
    gs.tol = 1e-12;
    auto gsSolver = Fbr_gs(model, gs);
    for (int i = 0; i < 80; i++) gsSolver.iterate({.max_bond_dim = 512});

    auto psi0 = gsSolver.fb.to_complex();
    auto [B0, nrm0] = addParticle(psi0, 0);
    auto [B1, nrm1] = addParticle(psi0, 1);
    // the three states share one active window, which has to hold every orbital
    // where they differ; a tight tolerance keeps it wide enough
    psi0.tol = B0.tol = B1.tol = 1e-12;
    auto solver = Fbr_ns_dyn(model, std::vector{psi0, B0, B1}, dt);

    std::size_t nStep = ref.size();
#ifndef FBR_ENABLE_LONG_TEST
    // Default: stop at t=5. Build with -DFBR_ENABLE_LONG_TEST=ON for the full file.
    nStep = std::min<std::size_t>(nStep, 51);
#endif

    GreenError err;
    for (std::size_t step = 0; step < nStep; step++) {
        cmpx G00 = -imag_1 * nrm0 * cElement(solver.states[0], solver.states[1], 0);
        cmpx G01 = -imag_1 * nrm1 * cElement(solver.states[0], solver.states[2], 0);
        double d = std::max(std::abs(G00 - ref[step].G00), std::abs(G01 - ref[step].G01));
        for (auto const &bucket : greenTol())
            if (ref[step].t <= bucket.t + 1e-9) {
                double &slot = err.byBucket[bucket.t];
                slot = std::max(slot, d);
                break;
            }
        if (step == 0) err.atZero = d;
        err.tMax = ref[step].t;
        if (step + 1 < nStep) solver.iterate({.epsilonM = 0});
    }
    return cache.emplace(us, err).first->second;
}

// Both sides are approximations -- the chain baseline is itself a TDVP run -- so
// these bound how far apart two methods drift, not an exact error.
void checkGreen(GreenError const &err)
{
    // At t=0 the Green functions only involve the ground state, so this compares
    // the FBR ground state with the chain DMRG one, and is much tighter than the
    // bound on the evolution that follows.
    INFO("|dG(t=0)| = " << err.atZero);
    REQUIRE(err.atZero < 1e-6);

    for (auto const &bucket : greenTol()) {
        auto it = err.byBucket.find(bucket.t);
        if (it == err.byBucket.end()) continue;      // the run did not get there
        INFO("up to t=" << bucket.t << " (run reached t=" << err.tMax
             << "): max|dG| = " << it->second);
        REQUIRE(it->second < bucket.tol);
    }
}

} // namespace

TEST_CASE("fbr green vs chain reference U=0.2", "[fb_ref_green]") {
    checkGreen(resultFor("0.2"));
}
TEST_CASE("fbr green vs chain reference U=0.1", "[fb_ref_green]") {
    checkGreen(resultFor("0.1"));
}
