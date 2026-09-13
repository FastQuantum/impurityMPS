// DATA3b: evolve the interacting ground state with the two schemes and compare.
//
//   IP    : Fbr_dyn        -- interaction picture, bath phase folded into K.
//   FRAME : Fbr_dyn_frame  -- co-moving frame, exp(-i H_bath dt) rotates the
//                             active window each step (Schrodinger picture).
//
// |gs> is an eigenstate, so its Schrodinger observables are stationary. The test
// is whether FRAME keeps the window smaller than IP (whose interaction-picture cc
// precesses), while both keep <n0> and the impurity correlator correct.
//
// Columns per scheme: t  n_active  bond_dim  wall_s  n0  drift
//   n0    = <c_0^dag c_0>            impurity occupation (should stay ~0.5)
//   drift = max_i,j |<c_i^dag c_j>(t) - (t=0)| over the impurity+first-bath block
//           (should stay ~0 for an eigenstate)
//
// Usage: gs_frame_vs_ip_siam [L] [tmax] [U] [dt]   (100, L/2, 0.025, 0.1)

#include "fbr/fbr.h"
#include "fbr/fbr_dyn_frame.h"

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
    for (int i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = -U / 2; K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
    mat Umat(L, L, fill::zeros); Umat(0, 1) = U;
    auto model = ImpurityParam{.Kmat = K, .Umat = Umat,
                               .imp_pos = {0, 1}, .layout = spin_symmetric};
    model.to_star();
    return model;
}

// small block of real-space sites to watch the correlator drift on
const uvec watch = {0, 1, 2, 3};

cx_mat watchBlock(cx_mat const& full) { return full.submat(watch, watch); }

template<class Solver>
void run(Solver& solver, int nStep, double dt, string const& name)
{
    ofstream out(name);
    out << setprecision(12) << "# t  n_active  bond_dim  wall_s  n0  drift\n";
    cx_mat ref;
    itensor::cpu_time t0;
    for (int step = 0; step <= nStep; step++) {
        cx_mat cc = solver.correlator();
        cx_mat blk = watchBlock(cc);
        if (step == 0) ref = blk;
        double drift = arma::abs(blk - ref).max();
        double n0 = std::real(cc(0, 0));
        out << step * dt
            << " " << solver.fb.n_active()
            << " " << itensor::maxLinkDim(solver.fb.psi)
            << " " << t0.sincemark().wall
            << " " << n0
            << " " << drift << endl;
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
    auto gs = slater<double>(model);
    gs.tol = 1e-12;
    auto gs_solver = Fbr_gs(model, gs);
    itensor::cpu_time clk;
    for (int i = 0; i < 80; i++) gs_solver.iterate({.max_bond_dim = 512});
    cout << setprecision(12) << "# L=" << L << " U=" << U << " dt=" << dt
         << "  E_gs=" << gs_solver.energy << " n_active(gs)=" << gs_solver.fb.n_active()
         << " in " << clk.sincemark().wall << " s" << endl;


    {
        auto solver = Fbr_dyn(model, gs_solver.fb.to_complex(), dt);
        solver.fb.tol = 1e-10;
        run(solver, nStep, dt, "app/output/gs_ip_siam_L" + to_string(L) + "_U" + us + ".dat");
    }
    {
        auto solver = Fbr_dyn_frame(model, gs_solver.fb.to_complex(), dt);
        solver.fb.tol = 1e-10;
        run(solver, nStep, dt, "app/output/gs_frame_siam_L" + to_string(L) + "_U" + us + ".dat");
    }
    return 0;
}
