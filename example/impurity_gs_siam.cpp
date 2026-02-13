#include "impurityMPS/impurity_gs.h"
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    int L=1000;
    double U=1.0;
    arma::mat K(L,L, arma::fill::zeros);
    {
        for(auto i=0; i<L-2; i++)
            K(i,i+2)=K(i+2,i)=0.5;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
    }
    arma::mat Umat={{0,U},{0,0}};
    auto model = Impurity {{.Kmat=K, .Umat=Umat}};

    auto ek=arma::vec {model.param.Kmat.diag()};
    // optional: force impurity ocupation |10>
    ek[0]=-10;
    ek[1]=10;
    auto fb=Fb_mps<double>::from_slater(model.param.rot, ek, model.param.nPart(), model.param.nImp());
    fb.natOrbDepth=20;

    auto solver=Impurity_gs(model,fb);

    cout<<"iteration m nActive energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<30;i++){
        solver.iterate();
        cout<<i+1<<" "<<maxLinkDim(solver.fb.psi)<<" "<<solver.fb.nActive<<" "<<solver.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
