// FBR Green functions of the spinless IRLM at L=1000, written out as a
// regression baseline.
//
// Unlike chain_green_irlm.cpp this is NOT a trusted baseline: it is the FBR
// solver recording its own trajectory. At L=1000 there is nothing to check it
// against -- a real-space chain TDVP of three states on 1000 sites is out of
// reach, which is the whole point of the active-window method -- so what this
// pins down is that the solver keeps behaving exactly as it does today at a
// size the L=100 tests never exercise.
//
// The reference state is a Slater determinant with the impurity filled, NOT the
// interacting ground state, so these are the Green functions of that quench and
// not the equilibrium ones. That is deliberate: an Fbr_gs ground state at
// L=1000 is not reproducible from run to run. Two identical runs of this
// program gave energies differing by 4e-6, Im G(0,0) by 5e-4 and bond
// dimensions of 126 against 144 -- the DMRG is choosing between near-degenerate
// orbital sets, and tiny differences pick different ones. Nothing can be
// replayed against that. from_slater is deterministic, so the whole trajectory
// is, which is what lets the test below demand the bond dimension and the
// window width back exactly.
//
// The model is the one of test_ref_green.cpp and example/fbr_dyn_irlm.cpp,
// only longer: chain of hopping 0.5, impurity sites 0 and 1, hybridization
// V=0.1, e_imp=-U/2, U between the two impurity sites.
//
// Columns are t, the two Green functions, the largest bond dimension over the
// three states, and the width of the active window. The last two are what make
// this a sharp test: they are integers, so the replay has to land on them
// exactly.
//
// Usage: fbr_green_irlm_L1000 [U]        (default 0.2)
// Env:   GREEN_L, GREEN_NSTEP override the defaults (for timing trials).

#include "fbr/fbr_dyn.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace arma;
using namespace fbr;

static int envI(char const *k, int fallback)
{
    char const *v = std::getenv(k);
    return v ? std::atoi(v) : fallback;
}

static double envD(char const *k, double fallback)
{
    char const *v = std::getenv(k);
    return v ? std::atof(v) : fallback;
}

/// <A| c_i |B>, for i a non-rotating impurity site: there c_i is one MPS
/// orbital, so the matrix element is a plain overlap.
static cmpx cElement(Fb_mps<cmpx> const &A, Fb_mps<cmpx> const &B, int i)
{
    auto Ai = A;
    Ai.apply_local_op("Cdag", i);
    return itensor::innerC(Ai.psi, B.psi);
}

/// c_j^dag|psi0>, normalized, with the norm it had before normalizing: the
/// states of one Fbr_dyn_shared share their Slater part, so it has to be
/// normalized, and the norm goes back into G.
static std::pair<Fb_mps<cmpx>, double> addParticle(Fb_mps<cmpx> const &psi0, int j)
{
    auto state = psi0;
    state.apply_local_op("Cdag", j);
    double nrm = std::sqrt(std::real(itensor::innerC(state.psi, state.psi)));
    state.psi.normalize();
    state.update_cc();
    return {state, nrm};
}

int main(int argc, char **argv)
{
    double U = argc > 1 ? std::stod(argv[1]) : 0.2;
    int L = envI("GREEN_L", 1000);
    int nStep = envI("GREEN_NSTEP", 200);
    double V = 0.1;
    double dt = 0.1;
    // 1e-10, not the 1e-12 the L=100 tests use: the window keeps every orbital
    // whose occupation sits between tol and 1-tol, and at L=1000 there are ten
    // times as many orbitals carrying a tiny fraction of a particle, so a
    // tighter tol makes the active window (and the cost) explode.
    double tol = envD("GREEN_TOL", 1e-10);
    int n_part = L / 2;

    mat K(L, L, fill::zeros);
    for (int i = 1; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    K(0, 1) = K(1, 0) = V;
    K(0, 0) = K(1, 1) = -U / 2;
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    auto model = Impurity{{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1}}};

    itensor::cpu_time clk;
    // The reference state: a Slater determinant with BOTH impurity orbitals
    // empty. The greater Green function needs c_j^dag|psi> to be non-zero, and
    // on a determinant that means orbital j has to be empty -- filling it, as
    // example/fbr_dyn_irlm.cpp does, makes c_0^dag|psi> vanish outright.
    auto ek = vec{model.param.Kmat.diag()};
    ek[0] = ek[1] = 10;
    auto psi0 = Fb_mps<cmpx>::from_slater(model.param.rot * cmpx(1, 0), ek,
                                          n_part, model.param.n_imp(), leading);
    psi0.tol = tol;
    auto [B0, nrm0] = addParticle(psi0, 0);
    auto [B1, nrm1] = addParticle(psi0, 1);
    B0.tol = B1.tol = tol;
    auto solver = Fbr_dyn_shared(model, std::vector{psi0, B0, B1}, dt);

    string name = "fbr_green_irlm_L" + to_string(L) + "_U" + string(argv[1] ? argv[1] : "0.2") + ".txt";
    ostringstream rows;
    rows << setprecision(12);

    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;
        cmpx G00 = -imag_1 * nrm0 * cElement(solver.states[0], solver.states[1], 0);
        cmpx G01 = -imag_1 * nrm1 * cElement(solver.states[0], solver.states[2], 0);
        int m = 0;
        for (auto const &s : solver.states) m = std::max(m, itensor::maxLinkDim(s.psi));

        rows << t << " " << G00.real() << " " << G00.imag() << " "
             << G01.real() << " " << G01.imag() << " "
             << m << " " << solver.states[0].n_active() << "\n";

        // rewrite every step, so an interrupted run still leaves usable data
        ofstream out(name);
        out << "fbr_green_irlm_ref_v1 L " << L << " U " << U << " V " << V
            << " dt " << dt << " steps " << (step + 1) << "\n"
            << "# t ReG00 ImG00 ReG01 ImG01 maxBondDim n_active\n"
            << rows.str();
        out.close();

        if (step % 10 == 0)
            cerr << "# t=" << t << " m=" << m << " n_active=" << solver.states[0].n_active()
                 << " " << clk.sincemark().wall << " s\n";
        if (step < nStep) solver.iterate({.epsilon_M = 0});
    }
    cerr << "# wrote " << name << " with " << (nStep + 1) << " rows\n";
    return 0;
}
