#include "fbr/fb_mps.h"
#include "fbr/itensor_utils.h"
#include "tdvp.h"
#include "basisextension.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using namespace fbr;

/// return the kinetic energy in star geometry and the rotation to get it.
/// Layout: [spin-up bath | spin-up imp | spin-down imp | spin-down bath]
/// For spin-up the impurity is at the right end; for spin-down at the left end.
auto computeKstar(mat K, int nImp)
{
    int L=K.n_rows;
    int nBath=L/2-nImp/2;  // bath sites per spin

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);

    auto pos_up=regspace<uvec>(0,L/2-1);
    auto pos_dw=regspace<uvec>(L/2,L-1);

    for(int s : {0,1})
    {
        uvec pos   = s==0 ? pos_up : pos_dw;
        uvec pos_bath = s==0 ? pos.head(nBath)   : pos.tail(nBath);   // bath on left (up) or right (dw)
        uvec pos_impu = s==0 ? pos.tail(nImp/2)  : pos.head(nImp/2);  // imp on right (up) or left (dw)

        mat Kbath=K.submat(pos_bath,pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1,evec1,Kbath);
        uvec iek = s==0 ? sort_index(abs(ek1),"descend") : sort_index(abs(ek1));   // revert left bath
        arma::mat evec=evec1.cols(iek);
        arma::vec ek=ek1.rows(iek);

        arma::mat vk=K.submat(pos_impu,pos_bath)*evec;

        Kstar.submat(pos_impu,pos_impu)=K.submat(pos_impu,pos_impu);
        for(auto j=0u;j<ek.size();j++) {
            int jj=pos_bath[j];
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
    TdvpParam args;
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = args.max_bond_dim;
    sweeps.cutoff() = tol;
    sweeps.niter() = args.nIter_diag;
    sweeps.noise() = args.noise;

    std::vector<double> epsilonK(args.nKrylov, args.epsilonK);
    itensor::addBasis(psi, mpo, epsilonK,
                      {"Cutoff", args.epsilonM,
                       "Method", "DensityMatrix",
                       "KrylovOrd", args.nKrylov,
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
    double tol=1e-12;
    int L=K.n_rows;
    itensor::AutoMPO h(sites);
    for(int i=0; i<L; i++)
        for(int j=0; j<L; j++)
            if (std::abs(Umat(i,j))>1e-15)
                h += Umat(i,j), "N", i+1, "N", j+1;

    for(auto i=0; i<L; i++)
        for(auto j=0; j<L; j++)
            if (std::abs(K(i,j))>tol)
                h += K(i,j),"Cdag",i+1,"C",j+1;
    return itensor::toMPO(h);
}

int main()
{
    int L=100;
    int nImp=4;
    double dt=0.1;

    mat Kstar, Umat;
    mat rot;
    {
        double U=0.2;
        double V=0.1;
        int nBath=L/2-nImp/2;  // =4 for L=12, nImp=4
        arma::mat K(L,L, arma::fill::zeros);
        {
            for(auto i=0; i<L/2-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            for(auto i=L/2; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            K(nBath+nImp/2-1, nBath+nImp/2-1)=-U/2;
            K(L/2, L/2)=-U/2;
            K(nBath, nBath+nImp/2-1)=K(nBath+nImp/2-1, nBath)=V;
            K(L/2, L/2+nImp/2-1)=K(L/2+nImp/2-1, L/2)=V;
        }
        Umat.zeros(L, L);
        Umat(nBath+nImp/2-1, L/2) = U;

        std::tie(Kstar,rot) = computeKstar(K, nImp);
    }

    Fb_mps<cmpx> fb;
    {
        int nBath=L/2-nImp/2;
        auto ek=arma::vec {Kstar.diag()};
        // force impurity occupation: physical imp sites occupied, buffer sites empty
        ek[nBath+nImp/2-1]=ek[L/2]=-10;    // spin-up and spin-down physical impurities
        ek[nBath]=ek[L/2+nImp/2-1]=10;     // spin-up and spin-down buffers
        fb=Fb_mps<cmpx>::from_slater(rot*cmpx(1,0), ek, L/2, nImp, spin_symmetric);
    }

    // Construct model from pre-computed star geometry (bypassing toStar)
    Impurity model;
    {
        int nBath=L/2-nImp/2;
        model.param.Kmat = Kstar;
        model.param.Umat = Umat;
        model.param.rot  = rot;
        // convention 2: spatial order outer-up..inner-up..inner-dw..outer-dw
        model.param.impPos = {nBath, nBath+nImp/2-1, L/2, L/2+nImp/2-1};
        model.param.layout = spin_symmetric;
    }

    auto mpo=getHamiltonian(fb.sites,model.param.Kmat,model.param.Umat);

    int nBath=L/2-nImp/2;
    cout<<"time m n_up n_dw\n"<<setprecision(12);
    for(auto i=0;i*dt<L;i++){
        doTdvp(fb.psi,mpo,dt);
        double n_up=itensor::expectC(fb.psi,fb.sites,"N",{nBath+nImp/2+1})[0].real();    // spin-up physical imp (1-indexed)
        double n_dw=itensor::expectC(fb.psi,fb.sites,"N",{nBath+nImp/2+2})[0].real();  // spin-down physical imp (1-indexed)
        cout<<(i+1)*dt<<" "<<itensor::maxLinkDim(fb.psi)<<" "<<n_up<<" "<<n_dw<<endl;
    }
    return 0;
}
