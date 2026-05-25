// Side-by-side TDVP run of SIAM in two pictures:
//   - chain:   MPS lives in real space, Hamiltonian = K_real (chain).
//   - fb_mps:  MPS lives in star basis (Kbath diagonal), state in Fb_mps_spin,
//              real-space correlations recovered via fb.correlator_all().
//
// At each time step, the L×L matrix <c_i^dagger c_j> is computed both ways
// and the max absolute difference is printed. They should agree within TDVP
// precision (a few times the cutoff/err_goal).

// impurity_dyn_spin.h already pulls in itensor/all.h, tdvp.h, and basisextension.h
#include "impurityMPS/impurity_dyn_spin.h"
#include <armadillo>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using cmpx = std::complex<double>;

// ---- chain helpers (copied from chain_dyn_siam_center.cpp) ----

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

void doTdvp(itensor::MPS &psi, itensor::MPO const& mpo, double dt, int maxdim, double tol)
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

itensor::MPO siamChainHamiltonian(itensor::Fermion const& sites, mat const& K, mat const& UmatLocal,
                                  int nImp, int nBath)
{
    itensor::AutoMPO h(sites);
    int L = K.n_rows;
    for (int i = 0; i < nImp; i++)
        for (int j = 0; j < nImp; j++) {
            int ii = nBath + i;
            int jj = nBath + j;
            if (std::abs(UmatLocal(i,j)) > 1e-15)
                h += UmatLocal(i,j), "N", ii+1, "N", jj+1;
        }
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (std::abs(K(i,j)) > 1e-12)
                h += K(i,j), "Cdag", i+1, "C", j+1;
    return itensor::toMPO(h);
}

// ---- star helper (copied from impurity_dyn_siam_center.cpp) ----

auto computeKstar(mat K, int nImp)
{
    int L = K.n_rows;
    int nBath = L/2 - nImp/2;

    mat Kstar(L, L, fill::zeros);
    mat rot(L, L, fill::eye);

    auto pos_up = regspace<uvec>(0, L/2-1);
    auto pos_dw = regspace<uvec>(L/2, L-1);

    for (int s : {0, 1}) {
        uvec pos      = (s==0) ? pos_up : pos_dw;
        uvec pos_bath = (s==0) ? pos.head(nBath)  : pos.tail(nBath);
        uvec pos_impu = (s==0) ? pos.tail(nImp/2) : pos.head(nImp/2);

        mat Kbath = K.submat(pos_bath, pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1, evec1, Kbath);
        // Use stable_sort_index + reverse so up and dw orderings are exact
        // mirrors of each other on |ek|-degenerate pairs (required for the
        // Sz-reflection symmetry that Fb_mps_spin::extract_representative assumes).
        uvec iek_asc = stable_sort_index(abs(ek1));
        uvec iek = (s==0) ? uvec(reverse(iek_asc)) : iek_asc;
        mat evec = evec1.cols(iek);
        vec ek   = ek1.rows(iek);

        mat vk = K.submat(pos_impu, pos_bath).eval() * evec;
        Kstar.submat(pos_impu, pos_impu) = K.submat(pos_impu, pos_impu);
        for (auto j = 0u; j < ek.size(); j++) {
            int jj = pos_bath[j];
            Kstar(jj, jj) = ek[j];
            for (auto i = 0u; i < pos_impu.size(); i++) {
                int ii = pos_impu[i];
                Kstar(ii, jj) = Kstar(jj, ii) = vk(i, j);
            }
        }
        rot.cols(pos_bath) = rot.cols(pos_bath).eval() * evec;
    }
    return make_pair(Kstar, rot);
}

// ---- chain correlator: convert itensor result to cx_mat ----
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
    // ---- common parameters ----
    const int L      = 32;
    const int nImp   = 4;                  // 1 buffer + 1 phys imp per spin (impurity block has 4 sites)
    const int nBath  = L/2 - nImp/2;       // = 14 per spin
    const double dt  = 0.1;
    const double tMax = 10.0;
    const double U   = 0.0;   // TEST: non-interacting
    const double V   = 0.1;
    const int maxdim = 256;
    const double tol = 1e-10;

    // ---- build real-space K ----
    mat Kreal(L, L, fill::zeros);
    for (int i = 0; i < L/2-1; i++) Kreal(i,i+1) = Kreal(i+1,i) = 0.5;
    for (int i = L/2; i < L-1; i++) Kreal(i,i+1) = Kreal(i+1,i) = 0.5;
    Kreal(nBath+nImp/2-1, nBath+nImp/2-1) = -U/2;   // imp_up on-site
    Kreal(L/2, L/2)                       = -U/2;   // imp_dw on-site
    Kreal(nBath, nBath+nImp/2-1) = Kreal(nBath+nImp/2-1, nBath) = V;  // buffer_up – imp_up
    Kreal(L/2, L/2+nImp/2-1)     = Kreal(L/2+nImp/2-1, L/2)     = V;  // imp_dw – buffer_dw

    // Hubbard U between phys impurities
    mat UmatChain(nImp, nImp, fill::zeros);
    UmatChain(nImp/2-1, nImp/2) = U;                                  // (1,2) for nImp=4

    mat UmatSite(L, L, fill::zeros);
    UmatSite(nBath+nImp/2-1, L/2) = U;                                // site-indexed

    // ---- 1) chain side: real-space MPS ----
    itensor::Fermion sites_chain(L, {"ConserveNf", true});
    itensor::MPS psi_chain;
    {
        // Half-filled initial state with impurity block in |0,1,1,0> (buffer_up, imp_up, imp_dw, buffer_dw)
        int nPart = L/2;
        auto state = itensor::InitState(sites_chain, "0");
        for (int j = 0; j < nPart; j++) state.set(2*j+1, "1");
        state.set(L/2-1, "0");
        state.set(L/2,   "1");
        state.set(L/2+1, "1");
        state.set(L/2+2, "0");
        psi_chain = itensor::MPS(state);
    }
    {
        // Prepare GS of decoupled impurity-block + half-filled bath
        mat Kfake = Kreal;
        Kfake(nBath, nBath-1)        = Kfake(nBath-1, nBath)        = 0;
        int b2 = L/2 + nImp/2;
        Kfake(b2-1, b2)              = Kfake(b2, b2-1)              = 0;
        Kfake(nBath, nBath+nImp/2-1) = Kfake(nBath+nImp/2-1, nBath) = 0;
        Kfake(L/2, L/2+nImp/2-1)     = Kfake(L/2+nImp/2-1, L/2)     = 0;
        Kfake(L/2-2, L/2-2) =  10;
        Kfake(L/2-1, L/2-1) = -10;
        Kfake(L/2,   L/2)   = -10;
        Kfake(L/2+1, L/2+1) =  10;
        auto mpoFake = siamChainHamiltonian(sites_chain, Kfake, UmatChain, nImp, nBath);
        findGs(psi_chain, mpoFake);
    }
    auto mpo_chain = siamChainHamiltonian(sites_chain, Kreal, UmatChain, nImp, nBath);

    // ---- 2) fb_mps side: star basis ----
    auto [Kstar, rot] = computeKstar(Kreal, nImp);
    Fb_mps_spin<cmpx> fb;
    {
        vec ek = Kstar.diag();
        ek[nBath + nImp/2 - 1] = -10;   // imp_up
        ek[L/2]                = -10;   // imp_dw
        ek[nBath]              =  10;   // buffer_up
        ek[L/2 + nImp/2 - 1]   =  10;   // buffer_dw
        fb = Fb_mps_spin<cmpx>::from_slater(rot*cmpx(1,0), ek, L/2, nImp);
    }
    ImpuritySpin model;
    model.param.Kmat   = Kstar;
    model.param.Umat   = UmatSite;
    model.param.rot    = mat(L, L, fill::eye);
    model.param.impPos = {L/2-2, L/2-1, L/2, L/2+1};

    Impurity_dyn_spin solver(model, fb, dt);
    solver.fb.tol = tol;

    // ---- sanity: initial real-space correlations agree (chain GS vs fb Slater) ----
    {
        cx_mat C0_chain = chainCorr(psi_chain, sites_chain);
        cx_mat C0_fb    = solver.correlator_all();
        double init_diff = arma::abs(C0_chain - C0_fb).max();
        cout << "# initial |chain - fb|_inf = " << init_diff << "\n";
    }

    cout << "# t   max|Corr_chain - Corr_fb|   max|Corr_chain|   "
            "n_imp_up_chain  n_imp_up_fb  bondDim_chain  bondDim_fb\n"
         << setprecision(10);

    int impU_site = nBath + nImp/2 - 1;   // = L/2-1
    for (int step = 1; step*dt <= tMax + 1e-9; step++) {
        // evolve chain
        doTdvp(psi_chain, mpo_chain, dt, maxdim, tol);

        // evolve fb_mps
        solver.iterate({.max_bond_dim = maxdim, .nIter_diag = 16, .epsilonM = 1e-4});

        // compare real-space correlators
        cx_mat Cc = chainCorr(psi_chain, sites_chain);
        cx_mat Cf = solver.correlator_all();
        cx_mat D = Cc - Cf;
        arma::mat Dabs = arma::abs(D);
        arma::uword worst_idx = Dabs.index_max();
        arma::uword wi = worst_idx % Dabs.n_rows;
        arma::uword wj = worst_idx / Dabs.n_rows;
        double diff_max = Dabs.max();
        double Cmax     = arma::abs(Cc).max();

        double n_chain = std::real(Cc(impU_site, impU_site));
        double n_fb    = std::real(Cf(impU_site, impU_site));

        cout << step*dt << "  "
             << diff_max << "  "
             << Cmax     << "  "
             << n_chain  << "  "
             << n_fb     << "  "
             << itensor::maxLinkDim(psi_chain) << "  "
             << itensor::maxLinkDim(solver.fb.psi)
             << "  p1=" << solver.fb.p1 << " p2=" << solver.fb.p2
             << " worst=(" << wi << "," << wj << ")"
             << " Cc=" << Cc(wi,wj) << " Cf=" << Cf(wi,wj)
             << "\n" << flush;
    }
    return 0;
}
