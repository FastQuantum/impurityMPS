#include <itensor/all.h>
#include <tdvp.h>
#include <basisextension.h>
#include <armadillo>

#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;

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

    std::vector<double> epsilonK(15, 1e-4);   // match epsilonM from impurity_dyn
    itensor::addBasis(psi, mpo, epsilonK,
                      {"Cutoff", 1e-4,
                       "Method", "DensityMatrix",
                       "KrylovOrd", 15,
                       "DoNormalize", true,
                       "Quiet", true,
                       "Silent", true});

    using cmpx=complex<double>;
    itensor::tdvp(psi,mpo, -cmpx(0,1)*dt, sweeps,          // TDVP sweep
                  {"Truncate", true,
                   "DoNormalize", true,
                   "Quiet", true,
                   "Silent", true,
                   "NumCenter", 2,
                   "ErrGoal", 1e-8});
}

itensor::MPO getHamiltonian(itensor::Fermion sites, mat const& K, mat const& Umat)
{
    double tol=1e-12;
    int L=K.n_rows;
    int nImp=Umat.n_rows;
    int nBath=L/2-nImp/2;  // impurity cluster occupies sites [nBath, nBath+nImp)
    itensor::AutoMPO h(sites);
    for(auto i=0; i<nImp; i++)
        for(auto j=0; j<nImp; j++) {
            int ii=nBath+i;  // these positions change with nImp and nBath
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
    int nImp=4;
    double dt=0.1;
    int nBath=L/2-nImp/2;  // =4 for L=12, nImp=4

    mat Kstar, Umat; // define the Hamiltonian
    mat rot;         // define the orbitals
    {
        double U=0.2;
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        {
            // spin-up chain (sites 0..L/2-1) and spin-down chain (sites L/2..L-1)
            for(auto i=0; i<L/2-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            for(auto i=L/2; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            // impurity on-site energies: spin-up imp at site nBath+nImp/2-1, spin-down at L/2
            K(nBath+nImp/2-1, nBath+nImp/2-1)=-U/2;
            K(L/2, L/2)=-U/2;
            // hybridization V: overwrites the chain hop between buffer and physical impurity
            K(nBath, nBath+nImp/2-1)=K(nBath+nImp/2-1, nBath)=V;      // spin-up
            K(L/2, L/2+nImp/2-1)=K(L/2+nImp/2-1, L/2)=V;              // spin-down
        }
        Umat.zeros(nImp,nImp);
        Umat(nImp/2-1,nImp/2)=U;  // Hubbard U between spin-up imp (cluster idx nImp/2-1) and spin-down imp (nImp/2)

        std::tie(Kstar,rot) = computeKstar(K, nImp);
    }

    itensor::Fermion sites=itensor::Fermion(L, {"ConserveNf",true});
    itensor::MPS psi;  // should be  bath--|0110|--bath
    {
        auto ek=arma::vec {Kstar.diag()};
        // force impurity occupation: physical imp sites occupied, buffer sites empty
        ek[nBath+nImp/2-1]=ek[L/2]=-10;    // spin-up and spin-down physical impurities
        ek[nBath]=ek[L/2+nImp/2-1]=10;     // spin-up and spin-down buffers

        int nPart=L/2;
        sites=itensor::Fermion(ek.size(), {"ConserveNf",true});
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
        }
        psi=itensor::MPS(state);
    }

    auto mpo=getHamiltonian(sites,Kstar,Umat);

    cout<<"time m n_up n_dw\n"<<setprecision(12);
    for(auto i=0;i*dt<L;i++){
        doTdvp(psi,mpo,dt);
        double n_dw=itensor::expectC(psi,sites,"N",{nBath+nImp/2+1})[0].real();       // spin-down physical imp (1-indexed)
        double n_dw_bf=itensor::expectC(psi,sites,"N",{nBath+nImp/2+2})[0].real();    // spin-down buffer site (1-indexed)
        cout<<(i+1)*dt<<" "<<itensor::maxLinkDim(psi)<<" "<<n_dw<<" "<<n_dw_bf<<endl;
    }
    return 0;
}
