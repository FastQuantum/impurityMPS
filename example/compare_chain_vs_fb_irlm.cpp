// Spinless IRLM chain vs fb_mps comparator.
// Mirrors compare_chain_vs_fb_siam.cpp but for the spinless model — no Sz-reflection
// assumption, no spin doubling. Tests whether the chain-vs-fb residual seen in the spin
// version is specific to spin reflection, or is a general property of the natural-orbital
// truncation algorithm.

#include "impurityMPS/impurity_dyn.h"
#include <armadillo>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using cmpx = std::complex<double>;

void findGs(itensor::MPS &psi, itensor::MPO const& mpo)
{
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 512;
    sweeps.cutoff() = 1e-10;
    sweeps.niter()  = 4;
    sweeps.noise()  = 1e-8;
    for (int i = 0; i < 20; i++)
        itensor::dmrg(psi, mpo, sweeps, {"Quiet", true, "Silent", true});
}

void doTdvpChain(itensor::MPS &psi, itensor::MPO const& mpo, double dt, int maxdim, double tol)
{
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = maxdim;
    sweeps.cutoff() = tol;
    sweeps.niter()  = 16;
    sweeps.noise()  = 0.0;

    std::vector<double> epsilonK(15, 1e-4);
    itensor::addBasis(psi, mpo, epsilonK,
                      {"Cutoff", 1e-4,
                       "Method", "DensityMatrix",
                       "KrylovOrd", 15,
                       "DoNormalize", true,
                       "Quiet", true,
                       "Silent", true});

    itensor::tdvp(psi, mpo, -cmpx(0,1)*dt, sweeps,
                  {"Truncate", true,
                   "DoNormalize", true,
                   "Quiet", true,
                   "Silent", true,
                   "NumCenter", 2,
                   "ErrGoal", 1e-8});
}

itensor::MPO irlmChainHamiltonian(itensor::Fermion const& sites, mat const& K, double U)
{
    itensor::AutoMPO h(sites);
    int L = K.n_rows;
    h += U, "N", 1, "N", 2;   // sites 0,1 → ITensor 1-indexed → 1,2
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (std::abs(K(i,j)) > 1e-12)
                h += K(i,j), "Cdag", i+1, "C", j+1;
    return itensor::toMPO(h);
}

cx_mat chainCorr(itensor::MPS const& psi, itensor::Fermion const& sites)
{
    int L = sites.length();
    auto C = itensor::correlationMatrixC(psi, sites, "Cdag", "C");
    cx_mat out(L, L);
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            out(i, j) = C[i][j];
    return out;
}

int main()
{
    const int L = 32;
    const double dt = 0.1;
    const double tMax = 10.0;
    const double U = 0.2;
    const double V = 0.1;
    const int maxdim = 256;
    const double tol = 1e-10;

    // Real-space K: site 0 = impurity, site 1 = buffer; sites 2..L-1 = bath chain.
    mat Kreal(L, L, fill::zeros);
    for (int i = 1; i < L-1; i++) Kreal(i, i+1) = Kreal(i+1, i) = 0.5;
    Kreal(0, 1) = Kreal(1, 0) = V;
    Kreal(0, 0) = -U/2;
    Kreal(1, 1) = -U/2;

    // --- 1) chain MPS ---
    itensor::Fermion sites_chain(L, {"ConserveNf", true});
    itensor::MPS psi_chain;
    {
        // Half-filled initial state with impurity occupied and buffer empty.
        int nPart = L/2;
        auto state = itensor::InitState(sites_chain, "0");
        // Fill every other site to half-fill the bath part roughly.
        for (int j = 0; j < nPart; j++) state.set(2*j+1, "1");
        // Force impurity=|1>, buffer=|0>
        state.set(1, "1");
        state.set(2, "0");
        psi_chain = itensor::MPS(state);
    }
    {
        // Prepare GS of decoupled impurity + half-filled bath chain
        mat Kfake = Kreal;
        Kfake(0, 1) = Kfake(1, 0) = 0;
        Kfake(1, 2) = Kfake(2, 1) = 0;
        Kfake(0, 0) = -10;
        Kfake(1, 1) =  10;
        auto mpoFake = irlmChainHamiltonian(sites_chain, Kfake, U);
        findGs(psi_chain, mpoFake);
    }
    auto mpo_chain = irlmChainHamiltonian(sites_chain, Kreal, U);

    // --- 2) fb_mps: use Impurity{} constructor which calls toStar() internally ---
    Impurity model = Impurity{{.Kmat = Kreal, .Umat = mat({{0, U}, {0, 0}})}};
    Fb_mps<cmpx> fb;
    {
        vec ek = model.param.Kmat.diag();
        ek[0] = -10;  // impurity forced filled
        ek[1] =  10;  // buffer forced empty
        bool spin = false;
        fb = Fb_mps<cmpx>::from_slater(model.param.rot * cmpx(1,0), ek,
                                       model.param.nPart(), model.param.nImp(), spin);
    }
    fb.tol = tol;
    Impurity_dyn solver(model, fb, dt);
    solver.fb.tol = tol;

    {
        cx_mat C0_chain = chainCorr(psi_chain, sites_chain);
        cx_mat C0_fb    = solver.correlator_all();
        double init_diff = arma::abs(C0_chain - C0_fb).max();
        cout << "# initial |chain - fb|_inf = " << init_diff << "\n";
    }

    cout << "# t  diff_max  worst(i,j)  bondDim_chain  bondDim_fb  nActive\n" << setprecision(10);
    for (int step = 1; step*dt <= tMax + 1e-9; step++) {
        doTdvpChain(psi_chain, mpo_chain, dt, maxdim, tol);
        solver.iterate({.max_bond_dim = maxdim, .nIter_diag = 16, .epsilonM = 1e-4, .nKrylov = 15});

        cx_mat Cc = chainCorr(psi_chain, sites_chain);
        cx_mat Cf = solver.correlator_all();
        cx_mat D = Cc - Cf;
        arma::mat Dabs = arma::abs(D);
        arma::uword worst_idx = Dabs.index_max();
        arma::uword wi = worst_idx % Dabs.n_rows;
        arma::uword wj = worst_idx / Dabs.n_rows;
        double diff_max = Dabs.max();

        cout << step*dt << "  "
             << diff_max << "  "
             << "worst=(" << wi << "," << wj << ")"
             << "  bd_c=" << itensor::maxLinkDim(psi_chain)
             << "  bd_f=" << itensor::maxLinkDim(solver.fb.psi)
             << "  nActive=" << solver.fb.nActive
             << "  Cc=" << Cc(wi,wj) << "  Cf=" << Cf(wi,wj)
             << "\n" << flush;
    }
    return 0;
}
