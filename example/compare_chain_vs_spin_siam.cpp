// Side-by-side TDVP comparison for SIAM in two pictures:
//   chain:  MPS in real space (split spin: up=0..L/2-1, dw=L/2..L-1)
//   spin:   Impurity_dyn_spin with Fb_mps_spin; interleaved K (even=up, odd=dw) passed
//           to ImpuritySpin constructor, which calls toStar() internally.
//
// solver.correlator_all() returns <c_i† c_j> in the interleaved original basis.
// toSplitBasis() permutes it to split ordering before comparison.
//
// Permutation P[split_site] = interleaved_site:
//   spin-up bath  (split 0..nBath-1):      reversed relative to interleaved (desc |ek| vs asc |ek|)
//   spin-up cluster (split nBath..L/2-1):  also reversed
//   spin-down (split L/2..L-1):            same order as interleaved

#include "impurityMPS/impurity_dyn_spin.h"
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

// ---- permutation: interleaved basis -> split basis ----
// After ImpuritySpin::toStar(), correlator_all() is indexed by the interleaved
// original basis.  The split chain has:
//   up bath 0..nBath-1 ordered descending |ek| (outermost first),
//   up cluster nBath..L/2-1,
//   dw cluster L/2..L/2+nImp/2-1,
//   dw bath L/2+nImp/2..L-1 ordered ascending |ek|.
//
// In the interleaved basis:
//   imp_up=0, buf_up=2, bath_up={4,6,...} ascending site (= ascending |ek| after diag)
//   imp_dw=1, buf_dw=3, bath_dw={5,7,...} ascending site (= ascending |ek| after diag)
cx_mat toSplitBasis(cx_mat const& Cinter, int L, int nBath)
{
    const int nImpHalf = L/2 - nBath;

    uvec P(L);
    // spin-up bath (reversed: split 0=highest|ek| <-> interleaved 2*(nBath+nImpHalf-1-k))
    for (int k = 0; k < nBath; k++)
        P[k] = 2*(nBath + nImpHalf - 1 - k);
    // spin-up cluster (reversed: split nBath+j <-> interleaved 2*(nImpHalf-1-j))
    for (int j = 0; j < nImpHalf; j++)
        P[nBath + j] = 2*(nImpHalf - 1 - j);
    // spin-down: same order in both bases
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

    // ---- split K (chain) ----
    mat Kchain(L, L, fill::zeros);
    for (int i = 0; i < L/2-1; i++) Kchain(i,i+1) = Kchain(i+1,i) = 0.5;
    for (int i = L/2; i < L-1;  i++) Kchain(i,i+1) = Kchain(i+1,i) = 0.5;
    Kchain(nBath+nImp/2-1, nBath+nImp/2-1) = -U/2;
    Kchain(L/2,            L/2)             = -U/2;
    Kchain(nBath, nBath+nImp/2-1) = Kchain(nBath+nImp/2-1, nBath) = V;
    Kchain(L/2,   L/2+nImp/2-1)  = Kchain(L/2+nImp/2-1,   L/2)   = V;

    mat UmatChain(nImp, nImp, fill::zeros);
    UmatChain(nImp/2-1, nImp/2) = U;

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

    // ---- 2) spin/fb side: interleaved K -> ImpuritySpin (calls toStar internally) ----
    // Convention 2: impPos = {buf_up, imp_up, imp_dw, buf_dw} = {2, 0, 1, 3}
    mat Kinter(L, L, fill::zeros);
    for (int i = 0; i < L-2; i++) Kinter(i,i+2) = Kinter(i+2,i) = 0.5;
    Kinter(0,0) = Kinter(1,1) = -U/2;
    Kinter(0,2) = Kinter(2,0) = Kinter(1,3) = Kinter(3,1) = V;

    mat Umat(L, L, fill::zeros);
    Umat(0,1) = U;

    ImpuritySpin model {{.Kmat=Kinter, .Umat=Umat, .impPos={2,0,1,3}}};

    Fb_mps_spin<cmpx> fb;
    {
        vec ek = model.param.Kmat.diag();
        ek[L/2-1] = ek[L/2]   = -10;   // imp_up, imp_dw occupied
        ek[L/2-2] = ek[L/2+1] =  10;   // buf_up, buf_dw empty
        fb = Fb_mps_spin<cmpx>::from_slater(model.param.rot*cmpx(1,0), ek,
                                             model.param.nPart(), model.param.nImp());
    }

    Impurity_dyn_spin solver(model, fb, dt);
    solver.fb.tol = tol;

    // ---- sanity: initial correlators agree ----
    {
        cx_mat C0_chain = chainCorr(psi_chain, sites_chain);
        cx_mat C0_spin  = toSplitBasis(solver.correlator_all(), L, nBath);
        double init_diff = arma::abs(C0_chain - C0_spin).max();
        cout << "# initial |chain - spin|_inf = " << init_diff << "\n";
    }

    // Impurity block in split basis: sites [L/2-nImp/2 .. L/2+nImp/2-1]
    const int imp_a = L/2 - nImp/2;           // = nBath = 14
    const int imp_b = L/2 + nImp/2 - 1;       // = L/2 + nImp/2 - 1 = 17
    const int impUp_chain = nBath + nImp/2 - 1;  // = L/2 - 1 = 15 (spin-up imp)

    cout << "# t   max|C_imp_chain-C_imp_spin|   max|C_chain-C_spin|   "
            "n_imp_up_chain   n_imp_up_spin   bondDim_chain   bondDim_spin\n"
         << setprecision(10);

    for (int step = 1; step*dt <= tMax + 1e-9; step++) {
        doTdvp(psi_chain, mpo_chain, dt, maxdim, tol);
        solver.iterate({.max_bond_dim = maxdim, .nIter_diag = 16, .epsilonM = 1e-4});

        cx_mat Cc = chainCorr(psi_chain, sites_chain);
        cx_mat Cs = toSplitBasis(solver.correlator_all(), L, nBath);
        double diff_imp = arma::abs(Cc.submat(imp_a,imp_a,imp_b,imp_b)
                                   - Cs.submat(imp_a,imp_a,imp_b,imp_b)).max();
        double diff_all = arma::abs(Cc - Cs).max();
        double n_chain = std::real(Cc(impUp_chain, impUp_chain));
        double n_spin  = std::real(Cs(impUp_chain, impUp_chain));

        cout << step*dt    << "  "
             << diff_imp   << "  "
             << diff_all   << "  "
             << n_chain    << "  "
             << n_spin     << "  "
             << itensor::maxLinkDim(psi_chain)   << "  "
             << itensor::maxLinkDim(solver.fb.psi)
             << "\n" << flush;
    }
    return 0;
}
