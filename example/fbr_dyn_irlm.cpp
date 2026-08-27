#include "fbr/fbr_dyn.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace fbr;

int main()
{
    int L=100;
    double U=0.2;
    arma::mat K(L,L, arma::fill::zeros);
    {
        for(auto i=1; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=0.5;
        K(0,1)=K(1,0)=0.1;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
    }
    arma::mat Umat(L,L,arma::fill::zeros);
    Umat(0,1)=U;

    auto model = Impurity {{.Kmat=K, .Umat=Umat, .impPos={0,1}}};

    auto ek=arma::vec {model.param.Kmat.diag()};
    // force impurity occupation |10>
    ek[0]=-10;
    ek[1]=10;
    auto fb=slater<cmpx>(model, ek);
    fb.tol=1e-10;

    double dt=0.1;
    auto solver=Fbr_dyn(model,fb,dt);

    cout<<"time energy <n0> <cd> nActive\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i*dt<L; i++){
        solver.iterate({.max_bond_dim=2048, .epsilonM=1e-4});
        double n0 = solver.correlator(0,0).real();
        double cd = 2*solver.correlator(0,1).real();
        cout<<(i+1)*solver.dt<<" "<<solver.energy<<" "<<n0<<" "<<cd<<" "<<solver.fb.nActive()<<endl;
        t0.mark();
    }
    return 0;
}
