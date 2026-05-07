#include <itensor/all.h>
#include <tdvp.h>
#include <basisextension.h>
#include <armadillo>

#include <iostream>
#include <iomanip>
#include <complex>

using namespace std;
using namespace arma;

/// return the kinetic energy in star geometry and the rotation to get it.
/// Layout: [spin-up bath | spin-up imp | spin-down imp | spin-down bath]
/// For spin-up the impurity is at the right end; for spin-down at the left end.
auto computeKstar(mat K, int nImp)
{
    int L=K.n_rows;
    int nBath=L/2-nImp/2;

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);

    auto pos_up=regspace<uvec>(0,L/2-1);
    auto pos_dw=regspace<uvec>(L/2,L-1);

    for(int s : {0,1})
    {
        uvec pos      = s==0 ? pos_up : pos_dw;
        uvec pos_bath = s==0 ? pos.head(nBath)   : pos.tail(nBath);
        uvec pos_impu = s==0 ? pos.tail(nImp/2)  : pos.head(nImp/2);

        mat Kbath=K.submat(pos_bath,pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1,evec1,Kbath);
        uvec iek = s==0 ? sort_index(abs(ek1),"descend") : sort_index(abs(ek1));
        arma::mat evec=evec1.cols(iek);
        arma::vec ek=ek1.rows(iek);

        arma::mat vk=K.submat(pos_impu,pos_bath).eval()*evec;
        Kstar.submat(pos_impu,pos_impu)=K.submat(pos_impu,pos_impu);

        for(auto j=0u;j<ek.size();j++) {
            int jj=pos_bath[iek[j]];
            Kstar(jj,jj)=ek[j];
            for(auto i=0u;i<pos_impu.size();i++) {
                int ii=pos_impu[i];
                Kstar(ii,jj)=Kstar(jj,ii)=vk(i,j);
            }
        }
        rot.cols(pos_bath)=rot.cols(pos_bath).eval()*evec;
    }

    return make_pair(Kstar,rot);
}

void doTdvp(itensor::MPS &psi, itensor::MPO const mpo, double dt, double tol=1e-12)
{
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024;
    sweeps.cutoff() = tol;
    sweeps.niter() = 16;
    sweeps.noise() = 0e-8;

    std::vector<double> epsilonK(15, 1e-4);
    itensor::addBasis(psi, mpo, epsilonK,
                      {"Cutoff", 1e-4,
                       "Method", "DensityMatrix",
                       "KrylovOrd", 15,
                       "DoNormalize", true,
                       "Quiet", true,
                       "Silent", true});

    using cmpx=complex<double>;
    itensor::tdvp(psi,mpo,-cmpx(0,1)*dt,sweeps,
                  {"Truncate", true,
                   "DoNormalize", true,
                   "Quiet", true,
                   "Silent", true,
                   "NumCenter", 2,
                   "ErrGoal", 1e-8});
}

/// Build MPO for H^(2): complex quadratic part K2 + Hubbard Umat.
itensor::MPO getHamiltonian2(itensor::Fermion sites,
                              arma::cx_mat const& K2,
                              arma::mat   const& Umat,
                              int nBath)
{
    double tol=1e-12;
    int L=K2.n_rows;
    int nImp=Umat.n_rows;
    using cmpx=complex<double>;

    itensor::AutoMPO h(sites);

    // Hubbard U (real, unchanged)
    for(auto i=0; i<nImp; i++)
        for(auto j=0; j<nImp; j++) {
            int ii=nBath+i;
            int jj=nBath+j;
            if(std::abs(Umat(i,j))>1e-15)
                h += Umat(i,j),"N",ii+1,"N",jj+1;
        }

    // Complex hopping (imp-imp and imp-bath only; bath-bath block is zero)
    for(auto i=0; i<L; i++)
        for(auto j=0; j<L; j++) {
            cmpx kij=K2(i,j);
            if(std::abs(kij)>tol)
                h += kij,"Cdag",i+1,"C",j+1;
        }

    return itensor::toMPO(h);
}

/// Build the K2 matrix for H^(2) at the current step.
///
/// Interaction-picture derivation (star geometry, H_bath = sum_i eps_i n_i):
///
///   The IP hybridization at time t is K_eff = K_star_hyb * Ubath,
///   where Ubath accumulates both the bath phase evolution exp(-i eps dt) at
///   each step and any additional intra-bath rotations R (via Ubath <- R * Ubath).
///
///   The second-order Trotter correction [H_IP, H_bath] adds a factor
///   (1 - i*dt/2 * eps_j) to each bath column j in the ORIGINAL star basis:
///
///     K2_hyb = K_star_hyb * diag(1 - i*dt/2 * eps) * Ubath
///
///   This formula stays correct after any intra-bath rotation because H_bath is
///   always diagonal in the original star basis with eigenvalues eps_j.
///
///   Bath-bath block: zero (H_bath subtracted in H^(2) = H - H_bath - correction).
arma::cx_mat buildK2(arma::mat   const& Kstar,
                      arma::uvec  const& impIdx,
                      arma::uvec  const& bathIdx,
                      arma::vec   const& bathEk,    // original star bath energies
                      arma::cx_mat const& Ubath,     // accumulated bath unitary (nBath x nBath)
                      double dt)
{
    using cmpx=complex<double>;
    int L=Kstar.n_rows;

    // Trotter factor: diag(1 - i*dt/2 * eps_j)  [one entry per bath site]
    cx_vec tfac(bathIdx.size());
    for(auto j=0u; j<bathIdx.size(); j++)
        tfac[j] = cmpx(1.0, -dt/2.0 * bathEk[j]);

    // K_star_hyb: imp x bath submatrix (real, cast to complex)
    cx_mat Khyb(impIdx.size(), bathIdx.size());
    for(auto a=0u; a<impIdx.size(); a++)
        for(auto j=0u; j<bathIdx.size(); j++)
            Khyb(a,j) = Kstar(impIdx[a], bathIdx[j]);

    // K2_hyb = Khyb * diag(tfac) * Ubath
    cx_mat K2_hyb = Khyb * arma::diagmat(tfac) * Ubath;

    // Assemble full L x L K2
    cx_mat K2(L, L, arma::fill::zeros);

    // Imp-imp block (unchanged)
    for(auto a=0u; a<impIdx.size(); a++)
        for(auto b=0u; b<impIdx.size(); b++)
            K2(impIdx[a], impIdx[b]) = Kstar(impIdx[a], impIdx[b]);

    // Hybridization + its Hermitian conjugate
    for(auto a=0u; a<impIdx.size(); a++)
        for(auto j=0u; j<bathIdx.size(); j++) {
            K2(impIdx[a], bathIdx[j]) = K2_hyb(a,j);
            K2(bathIdx[j], impIdx[a]) = conj(K2_hyb(a,j));
        }

    // Bath-bath block: remains zero (H_bath removed from H^(2))
    return K2;
}

int main()
{
    int L=100;
    int nImp=4;
    double dt=0.1;
    int nBath=L/2-nImp/2;

    mat Kstar, Umat;
    mat rot;
    {
        double U=0.2;
        double V=0.1;
        arma::mat K(L,L,arma::fill::zeros);
        {
            for(auto i=0; i<L/2-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            for(auto i=L/2; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            K(nBath+nImp/2-1,nBath+nImp/2-1)=-U/2;
            K(L/2,L/2)=-U/2;
            K(nBath,nBath+nImp/2-1)=K(nBath+nImp/2-1,nBath)=V;
            K(L/2,L/2+nImp/2-1)=K(L/2+nImp/2-1,L/2)=V;
        }
        Umat.zeros(nImp,nImp);
        Umat(nImp/2-1,nImp/2)=U;

        tie(Kstar,rot)=computeKstar(K,nImp);
    }

    // --- Site index sets (0-indexed) ---
    // Layout: [0..nBath-1 = bath_up | nBath..nBath+1 = imp_up |
    //          L/2..L/2+1 = imp_dw  | L/2+2..L-1    = bath_dw ]
    uvec pos_up  = regspace<uvec>(0,   L/2-1);
    uvec pos_dw  = regspace<uvec>(L/2, L-1);
    uvec bath_up = pos_up.head(nBath);    // spin-up  bath:  0 .. nBath-1
    uvec bath_dw = pos_dw.tail(nBath);   // spin-down bath:  L/2+nImp/2 .. L-1
    uvec imp_up  = pos_up.tail(nImp/2);  // spin-up  imp:   nBath .. nBath+nImp/2-1
    uvec imp_dw  = pos_dw.head(nImp/2);  // spin-down imp:  L/2   .. L/2+nImp/2-1

    uvec bathIdx = join_vert(bath_up, bath_dw);   // 0-indexed, length = 2*nBath
    uvec impIdx  = join_vert(imp_up,  imp_dw);    // 0-indexed, length = nImp

    // Original star bath energies (diagonal of Kstar at bath sites)
    vec bathEk = Kstar.diag()(bathIdx);           // length 2*nBath

    // --- Accumulated bath unitary (interaction picture) ---
    //
    // Ubath tracks the single-particle propagator for bath modes.
    // Invariant: the effective hybridization in the current IP frame is
    //            K_eff = K_star_hyb * Ubath.
    //
    // Update rule at each time step (bath phase, never applied to MPS):
    //   Ubath <- diag(exp(-i*eps*dt)) * Ubath   (left-multiply, row-wise)
    //
    // To insert an intra-bath rotation R later, simply do:
    //   Ubath <- R * Ubath
    // and the rest of the code picks it up automatically via buildK2.
    using cmpx=complex<double>;
    int nBathTotal = (int)bathIdx.size();          // = 2*nBath
    cx_mat Ubath   = arma::eye<cx_mat>(nBathTotal, nBathTotal);

    // --- Initial state (same as original) ---
    itensor::Fermion sites=itensor::Fermion(L,{"ConserveNf",true});
    itensor::MPS psi;
    {
        auto ek=arma::vec{Kstar.diag()};
        ek[nBath+nImp/2-1]=ek[L/2]=-10;    // physical imp: force occupied
        ek[nBath]=ek[L/2+nImp/2-1]=10;     // buffer sites: force empty

        int nPart=L/2;
        sites=itensor::Fermion(ek.size(),{"ConserveNf",true});
        auto state=itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        for(int j=0; j<nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
        }
        psi=itensor::MPS(state);
    }

    // --- Time evolution (pure interaction picture) ---
    //
    // At each step the Trotter identity gives:
    //   exp(-i H dt) = exp(-i H_bath dt) * exp(-i H^(2) dt) + O(dt^3)
    //
    // We apply only exp(-i H^(2) dt) to the MPS (via TDVP).
    // exp(-i H_bath dt) is tracked analytically in Ubath and never touches the MPS.
    // The MPS therefore always represents the state in the interaction picture of H_bath.
    //
    // Impurity observables (n_dw, n_dw_buf below) are unaffected by this choice
    // because the bath-only unitary exp(i H_bath t) commutes with c†_imp c_imp.
    cout<<"time m n_dw n_dw_buf\n"<<setprecision(12);
    for(auto i=0; i*dt<L; i++) {

        // Build H^(2) for this step using current Ubath
        auto K2   = buildK2(Kstar, impIdx, bathIdx, bathEk, Ubath, dt);
        auto mpo2 = getHamiltonian2(sites, K2, Umat, nBath);

        // TDVP step: apply exp(-i H^(2) dt) to the MPS
        doTdvp(psi, mpo2, dt);

        // Advance Ubath by the (postponed) bath phase exp(-i eps dt):
        // Ubath <- diag(exp(-i*eps*dt)) * Ubath  (left-multiply row by row)
        for(auto bi=0u; bi<(size_t)nBathTotal; bi++)
            Ubath.row(bi) *= exp(-cmpx(0,1)*bathEk[bi]*dt);

        // *** Insert any intra-bath rotation R here: Ubath = R * Ubath ***

        // Impurity observables (same in IP and Schrödinger picture)
        double n_dw   =itensor::expectC(psi,sites,"N",{nBath+nImp/2+1})[0].real();
        double n_dw_bf=itensor::expectC(psi,sites,"N",{nBath+nImp/2+2})[0].real();
        cout<<(i+1)*dt<<" "<<itensor::maxLinkDim(psi)<<" "<<n_dw<<" "<<n_dw_bf<<endl;
    }

    return 0;
}
