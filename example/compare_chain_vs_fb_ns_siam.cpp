// Side-by-side TDVP comparison for SIAM in two pictures:
//   chain:  MPS in real space (split spin ordering: up=0..L/2-1, dw=L/2..L-1)
//   star:   Impurity_dyn with Fb_mps in star basis built from the interleaved K
//           (even sites = spin-up, odd sites = spin-down).
//
// fb.correlator_all() returns <c_i† c_j> in the interleaved original basis.
// Before comparing we reorder it to the split basis via the permutation
//   P[k]     = 2k       (k < L/2, spin-up)
//   P[L/2+k] = 2k+1     (k < L/2, spin-down)
// so that C_split[i,j] = C_interleaved[P[i], P[j]].

#include "impurityMPS/impurity_dyn.h"
#include <armadillo>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using cmpx = std::complex<double>;

// ---- chain helpers ----

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
                      {"Cutoff", 1e-4, "Method", "DensityMatrix",
                       "KrylovOrd", 15, "DoNormalize", true,
                       "Quiet", true, "Silent", true});
    itensor::tdvp(psi, mpo, -cmpx(0,1)*dt, sweeps,
                  {"Truncate", true, "DoNormalize", true,
                   "Quiet", true, "Silent", true,
                   "NumCenter", 2, "ErrGoal", 1e-8});
}

itensor::MPO chainHamiltonian(itensor::Fermion const& sites,
                               mat const& Kchain, mat const& UmatLocal,
                               int nImp, int nBath)
{
    int L = Kchain.n_rows;
    itensor::AutoMPO h(sites);
    for (int i = 0; i < nImp; i++)
        for (int j = 0; j < nImp; j++) {
            int ii = nBath + i, jj = nBath + j;
            if (std::abs(UmatLocal(i,j)) > 1e-15)
                h += UmatLocal(i,j), "N", ii+1, "N", jj+1;
        }
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (std::abs(Kchain(i,j)) > 1e-12)
                h += Kchain(i,j), "Cdag", i+1, "C", j+1;
    return itensor::toMPO(h);
}

cx_mat chainCorr(itensor::MPS const& psi, itensor::Fermion const& sites)
{
    int L = sites.length();
    auto C = itensor::correlationMatrixC(psi, sites, "Cdag", "C");
    cx_mat out(L, L);
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            out(i,j) = C[i][j];
    return out;
}

// ---- star helpers (interleaved ordering) ----

auto computeKstar(mat K, int nImp)
{
    int L = K.n_rows;
    auto pos_up = regspace<uvec>(0,2,L-1);
    auto pos_dw = regspace<uvec>(1,2,L-1);

    mat Kstar(L,L,fill::zeros);
    mat rot(L,L,fill::eye);

    for (auto pos : {pos_up, pos_dw}) {
        uvec pos_bath = pos.subvec(nImp/2, L/2-1);
        uvec pos_impu = pos.subvec(0, nImp/2-1);

        mat Kbath = K.submat(pos_bath, pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1, evec1, Kbath);
        uvec iek = stable_sort_index(abs(ek1));
        mat evec = evec1.cols(iek);
        vec ek   = ek1.rows(iek);

        mat vk = K.submat(pos_impu, pos_bath) * evec;
        Kstar.submat(pos_impu, pos_impu) = K.submat(pos_impu, pos_impu);
        for (auto j = 0u; j < ek.size(); j++) {
            int jj = pos_bath[j];
            Kstar(jj,jj) = ek[j];
            for (auto i = 0u; i < pos_impu.size(); i++) {
                int ii = pos_impu[i];
                Kstar(ii,jj) = Kstar(jj,ii) = vk(i,j);
            }
        }
        rot.cols(pos_bath) = rot.cols(pos_bath).eval() * evec;
    }
    return make_pair(Kstar, rot);
}

// Reorder a correlator from interleaved basis to split basis.
// P[split_site] = interleaved_site (same physical orbital, different label).
//
// Interleaved spin-up: imp(0), buf(2), bath(4,6,...,2*nBath+2)
// Split    spin-up:    bath(0..nBath-1), buf(nBath), imp(L/2-1)
//
// Interleaved spin-down: imp(1), buf(3), bath(5,7,...,2*nBath+3)
// Split    spin-down:    imp(L/2), buf(L/2+1), bath(L/2+2..L-1)
cx_mat toSplitBasis(cx_mat const& Cinter, int L, int nBath)
{
    const int nImpHalf = L/2 - nBath;   // = nImp/2

    uvec P(L);
    // spin-up bath ordering is reversed: split site k (farthest from imp at k=0)
    // maps to interleaved site 2*(nBath+nImpHalf-1-k) (also farthest from imp at k=0)
    for (int k = 0; k < nBath; k++)
        P[k] = 2*(nBath + nImpHalf - 1 - k);
    // spin-up cluster (reversed): split nBath+j -> interleaved 2*(nImpHalf-1-j)
    for (int j = 0; j < nImpHalf; j++)
        P[nBath + j] = 2*(nImpHalf - 1 - j);
    // spin-down: split L/2+k -> interleaved 2k+1  (same ordering in both)
    for (int k = 0; k < L/2; k++)
        P[L/2 + k] = 2*k + 1;

    cx_mat Csplit(L, L);
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            Csplit(i,j) = Cinter(P[i], P[j]);
    return Csplit;
}

int main()
{
    const int    L      = 32;
    const int    nImp   = 4;
    const int    nBath  = L/2 - nImp/2;   // 14 per spin
    const double dt     = 0.1;
    const double tMax   = 10.0;
    const double U      = 0.2;
    const double V      = 0.1;
    const int    maxdim = 256;
    const double tol    = 1e-10;

    // ---- split K (chain: up=0..L/2-1, dw=L/2..L-1) ----
    mat Kchain(L, L, fill::zeros);
    for (int i = 0; i < L/2-1; i++) Kchain(i,i+1) = Kchain(i+1,i) = 0.5;
    for (int i = L/2; i < L-1;  i++) Kchain(i,i+1) = Kchain(i+1,i) = 0.5;
    Kchain(nBath+nImp/2-1, nBath+nImp/2-1) = -U/2;
    Kchain(L/2,            L/2)             = -U/2;
    Kchain(nBath, nBath+nImp/2-1) = Kchain(nBath+nImp/2-1, nBath) = V;
    Kchain(L/2,   L/2+nImp/2-1)  = Kchain(L/2+nImp/2-1,   L/2)   = V;

    mat UmatChain(nImp, nImp, fill::zeros);
    UmatChain(nImp/2-1, nImp/2) = U;

    // ---- interleaved K (star input: even=up, odd=dw) ----
    mat Kinter(L, L, fill::zeros);
    for (int i = 0; i < L-2; i++) Kinter(i,i+2) = Kinter(i+2,i) = 0.5;
    Kinter(0,0) = Kinter(1,1) = -U/2;
    Kinter(0,2) = Kinter(2,0) = Kinter(1,3) = Kinter(3,1) = V;

    mat Umat(nImp, nImp, fill::zeros);
    Umat(0,1) = U;

    // ---- 1) chain side ----
    itensor::Fermion sites_chain(L, {"ConserveNf", true});
    itensor::MPS psi_chain;
    {
        auto state = itensor::InitState(sites_chain, "0");
        for (int j = 0; j < L/2; j++) state.set(2*j+1, "1");
        state.set(L/2-1, "0");
        state.set(L/2,   "1");
        state.set(L/2+1, "1");
        state.set(L/2+2, "0");
        psi_chain = itensor::MPS(state);
    }
    {
        mat Kfake = Kchain;
        Kfake(nBath,   nBath-1)        = Kfake(nBath-1,   nBath)        = 0;
        int b2 = L/2 + nImp/2;
        Kfake(b2-1,    b2)             = Kfake(b2,         b2-1)         = 0;
        Kfake(nBath,   nBath+nImp/2-1) = Kfake(nBath+nImp/2-1, nBath)   = 0;
        Kfake(L/2,     L/2+nImp/2-1)  = Kfake(L/2+nImp/2-1,   L/2)     = 0;
        Kfake(L/2-2, L/2-2) =  10;
        Kfake(L/2-1, L/2-1) = -10;
        Kfake(L/2,   L/2)   = -10;
        Kfake(L/2+1, L/2+1) =  10;
        auto mpoFake = chainHamiltonian(sites_chain, Kfake, UmatChain, nImp, nBath);
        findGs(psi_chain, mpoFake);
    }
    auto mpo_chain = chainHamiltonian(sites_chain, Kchain, UmatChain, nImp, nBath);

    // ---- 2) star/fb side ----
    auto [Kstar, rot] = computeKstar(Kinter, nImp);
    Fb_mps<cmpx> fb;
    {
        vec ek = Kstar.diag();
        ek[0] = ek[1] = -10;   // physical imps (interleaved sites 0,1)
        ek[2] = ek[3] =  10;   // buffers       (interleaved sites 2,3)
        fb = Fb_mps<cmpx>::from_slater(rot * cmpx(1,0), ek, L/2, nImp, false);
    }

    Impurity model;
    model.param.Kmat   = Kstar;
    model.param.Umat   = Umat;
    model.param.rot    = rot;
    model.param.impPos = iota(nImp);

    auto solver = Impurity_dyn(model, fb, dt);
    solver.fb.tol = tol;

    // ---- sanity: initial correlators agree ----
    {
        cx_mat C0_chain = chainCorr(psi_chain, sites_chain);
        cx_mat C0_star  = toSplitBasis(solver.correlator_all(), L, nBath);
        arma::mat D = arma::abs(C0_chain - C0_star);
        arma::uword wi = D.index_max() % D.n_rows;
        arma::uword wj = D.index_max() / D.n_rows;
        cout << "# initial |chain - star|_inf = " << D.max()
             << "  worst=(" << wi << "," << wj << ")"
             << "  chain=" << C0_chain(wi,wj)
             << "  star=" << C0_star(wi,wj) << "\n";
        // print subblocks near worst pair for diagnosis
        cout << "# C_chain dw-bath 18..21 x 18..21:\n";
        for (int i=18; i<22; i++) {
            for (int j=18; j<22; j++) cout << "  " << C0_chain(i,j).real();
            cout << "\n";
        }
        cx_mat Cinter = solver.correlator_all();
        cout << "# C_inter dw-bath sites 5,7,9,11 (P[18..21]):\n";
        for (int i : {5,7,9,11}) {
            for (int j : {5,7,9,11}) cout << "  " << Cinter(i,j).real();
            cout << "\n";
        }
    }

    const int impUp_chain = nBath + nImp/2 - 1;   // spin-up physical imp in split basis

    cout << "# t   max|C_chain-C_star|   max|C_chain|   "
            "n_imp_up_chain   n_imp_up_star   bondDim_chain   bondDim_star\n"
         << setprecision(10);

    for (int step = 1; step*dt <= tMax + 1e-9; step++) {
        doTdvp(psi_chain, mpo_chain, dt, maxdim, tol);
        solver.iterate({.max_bond_dim = maxdim, .nIter_diag = 16, .epsilonM = 1e-4});

        cx_mat Cc = chainCorr(psi_chain, sites_chain);
        cx_mat Cs = toSplitBasis(solver.correlator_all(), L, nBath);
        double diff  = arma::abs(Cc - Cs).max();
        double Cmax  = arma::abs(Cc).max();
        double n_chain = std::real(Cc(impUp_chain, impUp_chain));
        double n_star  = std::real(Cs(impUp_chain, impUp_chain));

        cout << step*dt   << "  "
             << diff      << "  "
             << Cmax      << "  "
             << n_chain   << "  "
             << n_star    << "  "
             << itensor::maxLinkDim(psi_chain)   << "  "
             << itensor::maxLinkDim(solver.fb.psi)
             << "\n" << flush;
    }
    return 0;
}
