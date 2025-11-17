#include "impurityMPS/impurity_gs.h"
#include <iostream>
#include <iomanip>

using namespace std;

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
    arma::mat Umat={{0,U},{0,0}};
    auto model = Impurity ({.Kmat=K, .Umat=Umat});
    auto solver=Impurity_gs(model);

    { // optional: force impurity ocupation |10>
        auto ek=arma::vec {solver.param.Kmat.diag()};
        ek[0]=-10;
        ek[1]=10;
        solver.prepareSlaterGs(ek);
    }

    cout<<"iteration nActive energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<30;i++){
        solver.extract_representative(0);
        solver.extract_representative(1);
        solver.doDmrg();
        solver.rotateToNaturalOrbitals();
        cout<<i+1<<" "<<solver.fb.nActive<<" "<<solver.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
