// Fig. 2 of the paper (docs/TD_NO_paper-3.pdf): the maximum MPS bond dimension
// and the number of active orbitals during the IRLM quench, few-body (FBR)
// against full-MPS evolutions.
//
// Model (example/fbr_dyn_irlm.cpp): spinless chain of hopping 0.5, impurity
// sites 0 and 1 joined by V, e_imp=-U/2 on both and U n0 n1. Quench: at t<0
// site 0 is full, site 1 empty and the bath (sites 2..L-1) is in its own Fermi
// sea; at t=0 the impurity is connected.
//
//   fbr    Fbr_dyn from the Slater state, the paper's algorithm
//   chain  two-site TDVP on the real-space chain ("real space orbitals" of Fig. 2),
//          no global subspace expansion (nearest neighbour, not needed)
//   star   two-site TDVP in the star geometry of the FBR model, whole L kept
//
// All truncate the MPS at the same cutoff 1e-10 (the paper's epsilon). The
// full-MPS runs stop once the bond dimension reaches max_bond_dim.
//
// Output: app/output/quench_bond_irlm_<method>_L<L>_U<U>.dat, columns
//   t  n_active  bond_dim  wall_s  n0  n1
// (n_active = L for the full-MPS runs; n0, n1 the real-space impurity occupations).
//
// Usage: quench_bond_irlm method [L] [tmax] [U] [dt] [max_bond_dim]
//        (defaults fbr 200 100 0.2 0.1 1024)

#include "fbr/fbr.h"
#include "full_mps.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {

constexpr double cutoff = 1e-10;

mat irlm_K(int L, double U, double V)
{
    mat K(L, L, fill::zeros);
    for (int i = 1; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    K(0, 1) = K(1, 0) = V;
    K(0, 0) = K(1, 1) = -U / 2;
    return K;
}

mat irlm_Umat(int L, double U)
{
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    return Umat;
}

struct Row { double t; int n_active, bond; double wall, n0, n1; };

struct Writer {
    ofstream out;
    itensor::cpu_time clk;
    Writer(string const& name, string const& head) : out(name)
    {
        out << setprecision(12) << head << "# t  n_active  bond_dim  wall_s  n0  n1\n";
        cout << "# writing " << name << endl;
    }
    void operator()(Row r)
    {
        r.wall = clk.sincemark().wall;
        out << r.t << " " << r.n_active << " " << r.bond << " " << r.wall
            << " " << r.n0 << " " << r.n1 << endl;
        clk.mark();
    }
};

} // namespace

int main(int argc, char** argv)
{
    string method = argc > 1 ? argv[1] : "fbr";
    int L         = argc > 2 ? std::stoi(argv[2]) : 200;
    double tmax   = argc > 3 ? std::stod(argv[3]) : 100;
    string us     = argc > 4 ? argv[4] : "0.2";   // U as typed, names the output
    double U      = std::stod(us);
    double dt     = argc > 5 ? std::stod(argv[5]) : 0.1;
    int maxdim    = argc > 6 ? std::stoi(argv[6]) : 1024;
    double V      = 0.1;
    int nStep     = (int)std::llround(tmax / dt);

    mat K = irlm_K(L, U, V);
    auto model = ImpurityParam{.Kmat = K, .Umat = irlm_Umat(L, U), .imp_pos = {0, 1}};
    model.to_star();

    string name = "app/output/quench_bond_irlm_" + method + "_L" + to_string(L)
                + "_U" + us + ".dat";
    ostringstream head;
    head << "# IRLM quench (site 0 full, site 1 empty, bath Fermi sea), method=" << method << "\n"
         << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt << " tmax=" << tmax
         << " cutoff=" << cutoff << " max_bond_dim=" << maxdim << "\n";

    // impurity occupation |10>, bath modes filled below the Fermi level
    vec ek = vec{model.Kmat.diag()};
    ek[0] = -10;
    ek[1] = 10;

    if (method == "fbr") {
        auto fb = slater<cmpx>(model, ek);
        fb.tol = cutoff;
        auto solver = Fbr_dyn(model, fb, dt);
        Writer write(name, head.str());
        for (int step = 0; step <= nStep; step++) {
            write({step * dt, solver.fb.n_active(), itensor::maxLinkDim(solver.fb.psi), 0,
                   std::real(solver.correlator(0, 0)), std::real(solver.correlator(1, 1))});
            if (step < nStep) solver.iterate({.max_bond_dim = maxdim, .epsilon_M = 0});
        }
        return 0;
    }

    if (method != "chain" && method != "star")
        throw invalid_argument("method must be fbr, chain or star");

    auto sites = itensor::Fermion(L, {"ConserveNf", true});
    itensor::MPS psi;
    itensor::MPO mpo;
    if (method == "star") {
        // in the star the initial state is a product state; sites 0, 1 are not rotated
        psi = full_mps::product_state(sites, ek, model.n_part());
        mpo = full_mps::hamiltonian(sites, model.Kmat, model.Umat);
    }
    else {
        // real-space chain: the bath Fermi sea is entangled, get it by DMRG of the
        // decoupled Hamiltonian with the impurity pinned to |10>
        mat K0 = K;
        K0.rows(0, 1).zeros();             // sites 0 and 1 fully decoupled
        K0.cols(0, 1).zeros();
        K0(0, 0) = -10;
        K0(1, 1) = 10;
        vec seed(L, fill::ones);           // seed: |10> and every other bath site
        seed[0] = -10;
        seed[1] = 10;
        for (int i = 2; i < L; i += 2) seed[i] = -1;
        psi = full_mps::product_state(sites, seed, model.n_part());
        auto mpo0 = full_mps::hamiltonian(sites, K0, mat(L, L, fill::zeros));
        double e = full_mps::ground_state(psi, mpo0, 16, cutoff);
        cout << "# initial chain state E0=" << e << " m=" << itensor::maxLinkDim(psi) << endl;
        mpo = full_mps::hamiltonian(sites, K, irlm_Umat(L, U));
    }
    psi *= cmpx(1, 0);

    Writer write(name, head.str());
    for (int step = 0; step <= nStep; step++) {
        int m = itensor::maxLinkDim(psi);
        write({step * dt, L, m, 0, full_mps::density(sites, psi, 0),
               full_mps::density(sites, psi, 1)});
        if (m >= maxdim) {
            write.out << "# bond dim " << m << " reached " << maxdim << "; stopping\n";
            break;
        }
        if (step < nStep) full_mps::tdvp_step(psi, mpo, dt, cutoff, maxdim, method == "star");
    }
    return 0;
}
