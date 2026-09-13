#include "fbr/fb_mps.h"
#include "fbr/impurity_param.h"
#include "fbr/itensor_utils.h"
#include "tdvp.h"
#include "basisextension.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using namespace fbr;

/// return the kinetic energy in star geometry and the rotation to get it.
auto computeKstar(mat K, int n_imp)
{
    int L=K.n_rows;
    auto pos_up = regspace<uvec>(0,2,L-1);
    auto pos_dw = regspace<uvec>(1,2,L-1);

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);

    for(auto pos : {pos_up,pos_dw})
    {
        uvec pos_bath = pos.subvec(n_imp/2,L/2-1); // these two uvec can change if the impurity is in the center
        uvec pos_impu = pos.subvec(0,n_imp/2-1);

        mat Kbath = K.submat(pos_bath,pos_bath);
        mat evec1;
        vec ek1;
        eig_sym(ek1,evec1,Kbath);
        arma::uvec iek=arma::stable_sort_index(arma::abs(ek1));
        arma::mat evec=evec1.cols(iek);
        arma::vec ek=ek1.rows(iek);

        arma::mat vk=K.submat(pos_impu,pos_bath)*evec;

        Kstar.submat(pos_impu,pos_impu)=K.submat(pos_impu,pos_impu);
        for(auto j=0u;j<ek.size();j++) {
            int jj=pos_bath[j];
            Kstar(jj,jj)=ek[j];
            for(auto i=0u; i<pos_impu.size(); i++) {
                int ii=pos_impu[i];
                Kstar(ii,jj)=Kstar(jj,ii)=vk(i,j);
            }
        }
        rot.cols(pos_bath)=rot.cols(pos_bath).eval()*evec;
    } // pos

    return make_pair(Kstar,rot);
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

    itensor::tdvp(psi,mpo, -imag_1*dt, sweeps,          // TDVP sweep
                  {"Truncate", true,
                   "DoNormalize", true,
                   "Quiet", true,
                   "Silent", true,
                   "NumCenter", 2,
                   "ErrGoal", args.err_goal});
}

itensor::MPO getHamiltonian(itensor::Fermion sites, mat const& K, mat const& Umat)
{
    itensor::MPO mpo;
    double tol=1e-10;
    int L=K.n_rows;
    int n_imp=Umat.n_rows;
    itensor::AutoMPO h(sites);
    for(auto i=0; i<n_imp; i++)
        for(auto j=0; j<n_imp; j++) {
            int ii=i; //these two positions can change if the impurity is the center
            int jj=j;
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

    mat Kstar, Umat; // define the Hamiltonian
    mat rot;         // define the orbitals
    {
        double U=0.2;
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        {
            for(auto i=0; i<L-2; i++)
                K(i,i+2)=K(i+2,i)=0.5;
            K(0,0)=-U/2;
            K(1,1)=-U/2;
            K(0,2)=K(2,0)=K(1,3)=K(3,1)=V;
        }
        Umat.zeros(n_imp,n_imp);
        Umat(0,1)=U;

        std::tie(Kstar,rot) = computeKstar(K, n_imp);
    }

    Fb_mps<cmpx> fb;
    {
        auto ek=arma::vec {Kstar.diag()};
        // force impurity occupation |1100>
        ek[0]=ek[1]=-10;
        ek[2]=ek[3]=10;
        fb=Fb_mps<cmpx>::from_slater(rot*cmpx(1,0), ek, L/2, n_imp, leading);
    }

    auto mpo=getHamiltonian(fb.sites,Kstar,Umat);

    cout<<"iteration m 0 energy time\n"<<setprecision(12);
    for(auto i=0;i*dt<L;i++){
        do_tdvp(fb.psi,mpo,dt);
        double n0=itensor::expectC(fb.psi,fb.sites,"N",{1})[0].real();
        double n1=itensor::expectC(fb.psi,fb.sites,"N",{3})[0].real();
        cout<<(i+1)*dt<<" "<<itensor::maxLinkDim(fb.psi)<<" "<<n0<<" "<<n1<<endl;
    }
    return 0;
}
