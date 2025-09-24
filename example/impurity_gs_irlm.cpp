#include "impurityMPS/impurity_gs.h"
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    int L=1000;
    double U=0.5;
    arma::sp_mat K(L,L);
    {
        for(auto i=1; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=0.5;
        K(0,1)=K(1,0)=0.1;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
    }
    arma::sp_mat Umat(2,2);
    Umat(0,1)=U;
    auto Kstar=ImpurityParam::to_star_kin_tridiag(K, Umat.n_rows);
    auto model0=Impurity_gs({.Kstar=Kstar, .Umat=Umat});

    { // optional: force impurity ocupation |10>
        auto ek=arma::vec {Kstar.diag()};
        ek[0]=-10;
        ek[1]=10;
        model0.prepareSlaterGs(ek,L/2);
    }

    cout<<"iteration nActive energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<30;i++){
        model0.extract_f(0.0);
        model0.extract_f(1.0);
        model0.doDmrg();
        model0.rotateToNaturalOrbitals();
        cout<<i+1<<" "<<model0.nActive<<" "<<model0.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
