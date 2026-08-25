#include "fbr/fbr_gs.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace fbr;

int main()
{
    int L=1000;
    double U=0.5;
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
    auto fb=Fb_mps<double>::from_slater(model.param.rot, ek, model.param.nPart(), model.param.nImp(), leading, false);
    fb.tol=1e-10;

    auto solver=Fbr_gs(model,fb);

    cout<<"iteration nActive energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<100;i++){
        solver.iterate();
        cout<<i+1<<" "<<solver.fb.nActive()<<" "<<solver.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
