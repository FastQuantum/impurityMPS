#include "impurityMPS/impurity_dyn.h"
#include <iostream>
#include <iomanip>

using namespace std;

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
    arma::mat Umat={{0,U},{0,0}};
    auto model = Impurity {{.Kmat=K, .Umat=Umat}};

    auto ek=arma::vec {model.param.Kmat.diag()};
    // force impurity ocupation |10>
    ek[0]=-10;
    ek[1]=10;
    auto fb=Fb_mps<cmpx>::from_slater(model.param.rot*cmpx(1,0), ek, model.param.nPart(), model.param.nImp());
    fb.tol=1e-10;

    double dt=0.1;
    auto solver=Impurity_dyn(model,fb,dt);

    cout<<"time nActive energy <n0> time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i*dt<L; i++){
        solver.iterate(/*{.epsilonM=0}*/);
        double n0 = solver.fb.correlator(0,0).real();
        double cd=2*solver.fb.correlator(0,1).real();
        cout<<(i+1)*solver.dt<<" "<<solver.energy<<" "<<n0<<" "<<cd<<" "<<solver.fb.nActive<<endl;
        t0.mark();
    }
    return 0;
}
