// Convergence and cost of the SIAM ground state (Fbr_gs) as a function of the
// gs tolerance and the number of iterations. The excitation study builds
// c_0up^dag|gs> on this state, so we only need |gs> accurate to the dynamics
// cutoff (1e-7..1e-8), not to 1e-12. This probe prints the convergence curve so
// we can pick the cheapest gs that is good enough.
//
// Columns: iter  energy  dE_from_final  n_active  bond_dim  cum_wall_s
//
// Usage: gs_tune_siam [L] [U] [gs_tol] [niter] [maxdim]
//        defaults: 200 0.1 1e-9 80 512

#include "fbr/fbr.h"
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;
using namespace arma;
using namespace fbr;

static ImpurityParam siam_model(int L, double U, double V)
{
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    auto model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1}, .layout = spin_block};
    model.to_star();
    return model;
}

int main(int argc, char** argv)
{
    int L        = argc > 1 ? std::stoi(argv[1]) : 200;
    string us    = argc > 2 ? argv[2] : "0.1";
    double U     = std::stod(us);
    string ts    = argc > 3 ? argv[3] : "1e-9";
    double gstol = std::stod(ts);
    int niter    = argc > 4 ? std::stoi(argv[4]) : 80;
    int maxdim   = argc > 5 ? std::stoi(argv[5]) : 512;
    double V     = 0.1;

    auto model = siam_model(L, U, V);
    auto gs = slater<double>(model);
    gs.tol = gstol;
    auto solver = Fbr_gs(model, gs);

    cout << setprecision(12)
         << "# SIAM gs convergence  L=" << L << " U=" << U << " gs_tol=" << gstol
         << " niter=" << niter << " maxdim=" << maxdim << "\n"
         << "# iter  energy  n_active  bond_dim  cum_wall_s\n";
    itensor::cpu_time clk;
    vector<double> energies;
    for (int i = 0; i < niter; i++) {
        solver.iterate({.max_bond_dim = maxdim});
        energies.push_back(solver.energy);
        cout << i + 1 << " " << solver.energy << " " << solver.fb.n_active() << " "
             << itensor::maxLinkDim(solver.fb.psi) << " " << clk.sincemark().wall << endl;
    }
    double ef = energies.back();
    cerr << "# dE_from_final per iter:\n";
    for (int i = 0; i < niter; i++)
        cerr << i + 1 << " " << std::abs(energies[i] - ef) << endl;
    return 0;
}
