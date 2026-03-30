#include "impurityMPS/impurity_gs_spin.h"
#include "impurityMPS/impurity_param_spin.h"
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    int L=1000;
    double U=2.0;
    bool spin=false;
    arma::mat K(L,L, arma::fill::zeros);
    {
        for(auto i=0; i<L-2; i++)
            K(i,i+2)=K(i+2,i)=0.5;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
        K(0,2)=K(2,0)=K(1,3)=K(3,1)=0.5;
    }
    arma::mat Umat={{0,U},{0,0}};
    auto model = ImpuritySpin {{.Kmat=K, .Umat=Umat}};

    auto ek=arma::vec {model.param.Kmat.diag()};
    // optional: force impurity ocupation |10>
    ek[0]=-10;
    ek[1]=10;
    auto fb=Fb_mps_spin<double>::from_slater(model.param.rot, ek, model.param.nPart(), model.param.nImp());
    // fb.natOrbDepth=10;
    fb.tol=1e-10;

    auto solver=Impurity_gs_spin(model,fb);
    // solver.param.Kmat.print("Kmat");

    cout<<"iteration m nActive energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<100;i++){
        solver.iterate(/*{.max_bond_dim=128}*/);
        double n0 = solver.fb.correlator(0,0);
        double cd=2*solver.fb.correlator(0,1);
        cout<<i+1<<" "<<itensor::maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<cd<<" "<<solver.fb.p2-solver.fb.p1<<" "<<solver.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
