#include "fbr/givens_rotation.h"
#include "fbr/itensor_utils.h"
#include <itensor/all.h>
#include <tdvp.h>
#include <basisextension.h>
#include <armadillo>

#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using namespace fbr;
using cmpx= std::complex<double>;

/// return the kinetic energy in star geometry and the rotation to get it.
/// Layout: [spin-up bath | spin-up imp | spin-down imp | spin-down bath]
/// For spin-up the impurity is at the right end; for spin-down at the left end.
auto computeKstar(mat K, int n_imp)
{
    int L=K.n_rows;
    int nBath=L/2-n_imp/2;  // bath sites per spin

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);
    arma::mat Kstar_bath(L,L, fill::zeros);

    auto pos_up=regspace<uvec>(0,L/2-1);
    auto pos_dw=regspace<uvec>(L/2,L-1);

    for(int s : {0,1})
    {
        uvec pos   = s==0 ? pos_up : pos_dw;
        uvec pos_bath = s==0 ? pos.head(nBath)   : pos.tail(nBath);   // bath on left (up) or right (dw)
        uvec pos_impu = s==0 ? pos.tail(n_imp/2)  : pos.head(n_imp/2);  // imp on right (up) or left (dw)

        mat Kbath=K.submat(pos_bath,pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1,evec1,Kbath);
        uvec iek = s==0 ? sort_index(abs(ek1),"descend") : sort_index(abs(ek1));   // revert left bath
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
        Kstar_bath(pos_bath,pos_bath)=Kstar(pos_bath,pos_bath);
    }
    return make_pair(Kstar,Kstar_bath);
}

auto computeKip(arma::mat const& Kstar, int n_imp, double dt)
{
    using namespace arma;
    int L=Kstar.n_rows;
    int nBath=L/2-n_imp/2;

    uvec bathIdx;
    {
        uvec pos_up  = regspace<uvec>(0,   L/2-1);
        uvec pos_dw  = regspace<uvec>(L/2, L-1);
        uvec bath_up = pos_up.head(nBath);
        uvec bath_dw = pos_dw.tail(nBath);
        bathIdx = join_vert(bath_up, bath_dw);
    }

    //  H⁽²⁾ = H − H_bath − (idt/2)[H,H_bath]
    cx_mat Kbath(L,L,arma::fill::zeros);
    Kbath(bathIdx,bathIdx) = Kstar(bathIdx,bathIdx) * cmpx(1,0);
    cx_mat commutator = Kstar*Kbath - Kbath*Kstar;
    cx_mat Kip = Kstar - Kbath - cmpx(0,0.5*dt)*commutator;
    return Kip;
}

void do_tdvp(itensor::MPS &psi, itensor::MPO const mpo, double dt, double tol=1e-12)
{
    TdvpParam args;
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = args.max_bond_dim;
    sweeps.cutoff() = tol;
    sweeps.niter() = args.n_iter_diag;
    sweeps.noise() = args.noise;

    std::vector<double> epsilon_K(args.n_krylov, args.epsilon_K);
    itensor::addBasis(psi, mpo, epsilon_K,
                      {"Cutoff", args.epsilon_M,
                       "Method", "DensityMatrix",
                       "KrylovOrd", args.n_krylov,
                       "DoNormalize", true,
                       "Quiet", true,
                       "Silent", true});

    itensor::tdvp(psi,mpo, -cmpx(0,1)*dt, sweeps,          // TDVP sweep
                  {"Truncate", true,
                   "DoNormalize", true,
                   "Quiet", true,
                   "Silent", true,
                   "NumCenter", 2,
                   "ErrGoal", args.err_goal});
}

itensor::MPO getHamiltonian(itensor::Fermion sites, cx_mat const& K, mat const& Umat)
{
    double tol=1e-12;
    int L=K.n_rows;
    int n_imp=Umat.n_rows;
    int nBath=L/2-n_imp/2;  // impurity cluster occupies sites [nBath, nBath+n_imp)
    itensor::AutoMPO h(sites);
    for(auto i=0; i<n_imp; i++)
        for(auto j=0; j<n_imp; j++) {
            int ii=nBath+i;  // these positions change with n_imp and nBath
            int jj=nBath+j;
            if (std::abs(Umat(i,j))>1e-15)
                h += Umat(i,j), "N", ii+1, "N", jj+1;
        }

    for(auto i=0; i<L; i++)
        for(auto j=0; j<L; j++)
            if (std::abs(K(i,j))>tol)
                h += K(i,j),"Cdag",i+1,"C",j+1;
    return itensor::toMPO(h);
}

int main()
{
    int L=100;
    int n_imp=4;
    double dt=0.1;
    int nBath=L/2-n_imp/2;  // =4 for L=12, n_imp=4

    mat Kstar, Umat; // define the Hamiltonian
    mat Kbath;
    {
        double U=0.2;
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        {
            // spin-up chain (sites 0..L/2-1) and spin-down chain (sites L/2..L-1)
            for(auto i=0; i<L/2-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            for(auto i=L/2; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            // impurity on-site energies: spin-up imp at site nBath+n_imp/2-1, spin-down at L/2
            K(nBath+n_imp/2-1, nBath+n_imp/2-1)=-U/2;
            K(L/2, L/2)=-U/2;
            // hybridization V: overwrites the chain hop between buffer and physical impurity
            K(nBath, nBath+n_imp/2-1)=K(nBath+n_imp/2-1, nBath)=V;      // spin-up
            K(L/2, L/2+n_imp/2-1)=K(L/2+n_imp/2-1, L/2)=V;              // spin-down
        }
        Umat.zeros(n_imp,n_imp);
        Umat(n_imp/2-1,n_imp/2)=U;  // Hubbard U between spin-up imp (cluster idx n_imp/2-1) and spin-down imp (n_imp/2)

        std::tie(Kstar,Kbath) = computeKstar(K, n_imp);
    }

    itensor::Fermion sites=itensor::Fermion(L, {"ConserveNf",true});
    itensor::MPS psi;  // should be  bath--|0110|--bath
    {
        auto ek=arma::vec {Kstar.diag()};
        // force impurity occupation: physical imp sites occupied, buffer sites empty
        ek[nBath+n_imp/2-1]=ek[L/2]=-10;    // spin-up and spin-down physical impurities
        ek[nBath]=ek[L/2+n_imp/2-1]=10;     // spin-up and spin-down buffers

        int n_part=L/2;
        sites=itensor::Fermion(ek.size(), {"ConserveNf",true});
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        for(int j = 0; j < n_part; j++) {
            int k=iek[j];
            state.set(k+1,"1");
        }
        psi=itensor::MPS(state);
    }


    cout<<"time m n_up n_dw\n"<<setprecision(12);
    arma::cx_mat Kip = computeKip(Kstar,n_imp,dt);

    // arma::real(Kip).eval().print("Kip ok");

    for(auto i=0; i*dt<L; i++){
        arma::cx_mat expBath=expmat(-cmpx(0,1)*Kbath*dt*i);
        cx_mat Kip_n = expBath.t() * Kip * expBath;

        // arma::real(expBath).eval().print("exp_n ok");
        // arma::real(Kip).eval().print("Kip_n ok");

        auto mpo=getHamiltonian(sites,Kip_n,Umat);
        do_tdvp(psi,mpo,dt);        
        double n_dw=itensor::expectC(psi,sites,"N",{nBath+n_imp/2+1})[0].real();       // spin-down physical imp (1-indexed)
        double n_dw_bf=itensor::expectC(psi,sites,"N",{nBath+n_imp/2+2})[0].real();    // spin-down buffer site (1-indexed)
        cout<<(i+1)*dt<<" "<<itensor::maxLinkDim(psi)<<" "<<n_dw<<" "<<n_dw_bf<<endl;
    }
    return 0;
}
