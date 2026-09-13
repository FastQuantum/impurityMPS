// Cost comparison of two FBR real-time protocols for the SIAM (spin-symmetric):
//
//   1. QUENCH (the protocol of the paper): start from a Slater determinant with
//      the impurity forced doubly occupied and let it relax under the full H.
//      Number-conserving at half filling.
//
//   2. EXCITATION: the Green-function state c_0^dag|gs>, i.e. add one electron to
//      the impurity of the interacting ground state (from Fbr_gs) and evolve it.
//      This is the "hard" evolution the master-slave Fbr_dyn_shared is built for.
//
//   3. GROUND STATE: evolve |gs> itself. It is an eigenstate, so the state barely
//      moves and its window stays minimal -- the master-driven orbital rotations
//      (b) are then negligible and the measured cost isolates (a), the cost of the
//      interaction-picture TDVP evolution alone.
//
// All are evolved by the SAME single-state solver (Fbr_dyn) with the same model,
// dt and tolerance, so the columns below isolate what each protocol costs:
//
//   t  n_active  bond_dim  wall_s  n0  energy
//
// where n0 = <c_0^dag c_0> is the impurity (spin-up) occupation. Written to
//   app/output/quench_siam_L<L>_U<U>.dat
//   app/output/excitation_siam_L<L>_U<U>.dat
//
// Usage: quench_vs_excitation_siam [L] [tmax] [U] [dt]   (100, L/2, 0.025, 0.1)

#include "fbr/fbr.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {

ImpurityParam siam_model(int L, double U, double V)
{
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L - 2; i++)
        K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = -U / 2;
    K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    auto model = ImpurityParam{.Kmat = K, .Umat = Umat,
                               .imp_pos = {0, 1}, .layout = spin_symmetric};
    model.to_star();
    return model;
}

/// Evolve one state with Fbr_dyn, logging t, n_active, bond dim, wall/step, n0.
void run(ImpurityParam const& model, Fb_mps<cmpx> fb, double dt, int nStep,
         string const& name, string const& what, double e_ref)
{
    auto solver = Fbr_dyn(model, fb, dt);
    ofstream out(name);
    out << setprecision(12);
    out << "# SIAM " << what << " evolution, spin-symmetric FBR\n"
        << "# n0(t=0)=" << std::real(solver.correlator(0, 0))
        << "  E_gs(ref)=" << e_ref << "\n"
        << "# t  n_active  bond_dim  wall_s  n0  energy\n";

    itensor::cpu_time t0;
    for (int step = 0; step <= nStep; step++) {
        double n0 = std::real(solver.correlator(0, 0));
        double wall = t0.sincemark().wall;
        out << step * dt
            << " " << solver.fb.n_active()
            << " " << itensor::maxLinkDim(solver.fb.psi)
            << " " << wall
            << " " << n0
            << " " << solver.energy << endl;
        t0.mark();
        if (step < nStep) solver.iterate({.epsilon_M = 0});
    }
    cout << "# wrote " << name << endl;
}

} // namespace

int main(int argc, char** argv)
{
    int L       = argc > 1 ? std::stoi(argv[1]) : 100;
    double tmax = argc > 2 ? std::stod(argv[2]) : L / 2.0;
    string us   = argc > 3 ? argv[3] : "0.025";   // U as typed, names the output
    double U    = std::stod(us);
    double dt   = argc > 4 ? std::stod(argv[4]) : 0.1;
    double V    = 0.1;
    int nStep   = (int)std::llround(tmax / dt);

    auto model = siam_model(L, U, V);

    // ---- ground state (needed for the excitation, and as an energy reference) ----
    auto gs = slater<double>(model);
    gs.tol = 1e-12;
    auto gs_solver = Fbr_gs(model, gs);
    itensor::cpu_time clk;
    for (int i = 0; i < 80; i++) gs_solver.iterate({.max_bond_dim = 512});
    double e_gs = gs_solver.energy;
    cout << setprecision(12) << "# L=" << L << " U=" << U << " V=" << V
         << " dt=" << dt << " tmax=" << tmax
         << "  ground state E=" << e_gs << " in " << clk.sincemark().wall << " s" << endl;

    // ---- protocol 1: quench (impurity forced doubly occupied) ----
    {
        auto ek = vec{model.Kmat.diag()};
        int m = L / 2;                 // impurity orbitals sit at m-1 (up) and m (dw)
        ek[m - 1] = ek[m] = -1e3;      // fill both impurity spins
        auto fb = slater<cmpx>(model, ek);
        fb.tol = 1e-10;
        run(model, fb, dt, nStep,
            "app/output/quench_siam_L" + to_string(L) + "_U" + us + ".dat", "quench", e_gs);
    }

    // ---- protocol 2: excitation c_0^dag|gs> ----
    {
        auto fb = gs_solver.fb.to_complex();
        fb.tol = 1e-10;
        fb.apply_local_op("Cdag", 0);
        fb.psi.normalize();
        fb.update_cc();
        run(model, fb, dt, nStep,
            "app/output/excitation_siam_L" + to_string(L) + "_U" + us + ".dat", "excitation", e_gs);
    }

    // ---- protocol 3: ground state alone (isolates (a), the IP evolution) ----
    {
        auto fb = gs_solver.fb.to_complex();
        fb.tol = 1e-10;
        run(model, fb, dt, nStep,
            "app/output/gs_siam_L" + to_string(L) + "_U" + us + ".dat", "ground state", e_gs);
    }
    return 0;
}
