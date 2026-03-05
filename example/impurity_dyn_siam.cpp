#include "impurityMPS/impurity_dyn.h"
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    int L=1000;
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
    // arma::mat K(L,L, arma::fill::zeros);
    // {
    //     for(auto i=1; i<L-1; i++)
    //         K(i,i+1)=K(i+1,i)=0.5;
    //     K(0,1)=K(1,0)=0.1;
    //     K(0,0)=-U/2;
    //     K(1,1)=-U/2;
    // }
    Fb_mps<cmpx> fb;
    {
        arma::mat Umat(4,4,arma::fill::zeros);
        Umat(0,1)=U;
        auto model = Impurity {{.Kmat=K, .Umat=Umat}};

        auto ek=arma::vec {model.param.Kmat.diag()};
        // force impurity ocupation |1100>
        ek[0]=ek[1]=-10;
        ek[2]=ek[3]=10;
        fb=Fb_mps<cmpx>::from_slater(model.param.rot*cmpx(1,0), ek, model.param.nPart(), model.param.nImp());
    }

    arma::mat Umat(4,4,arma::fill::zeros);
    Umat(0,1)=U;
    auto model = Impurity {{.Kmat=K, .Umat=Umat}};


    double dt=0.1;
    auto solver=Impurity_dyn(model,fb,dt);

    cout<<"time m <n0> <cd>  nActive\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i*dt<L; i++){
        solver.iterate({.max_bond_dim=2048, .nIter_diag=16/*,.epsilonM=0*/});
        double n0 = solver.fb.correlator(0,0).real();
        double cd=2*solver.fb.correlator(0,1).real();
        cout<<(i+1)*solver.dt<<" "<<maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<cd<<" "<<solver.fb.nActive<<endl;
        t0.mark();
    }
    return 0;
}
