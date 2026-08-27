#include "fbr/fbr_gs_spin.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace fbr;

int main()
{
    int L=1000;
    double U=2.0;
    arma::mat K(L,L, arma::fill::zeros);
    {
        for(auto i=0; i<L-2; i++)
            K(i,i+2)=K(i+2,i)=0.5;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
        K(0,2)=K(2,0)=K(1,3)=K(3,1)=0.5;
    }
    // Umat: L×L, indexed by site in input Kmat layout. U on (site 0 = imp_up, site 1 = imp_dw).
    arma::mat Umat(L, L, arma::fill::zeros);
    Umat(0, 1) = U;
    // impPos in convention 2 (outer up, inner up=imp_up, inner dw=imp_dw, outer dw).
    // Only physical impurities here (no buffer): impPos = {0, 1}.
    auto model = Impurity {{.Kmat=K, .Umat=Umat, .impPos={0,1}, .layout=spin_symmetric}};

    auto ek=arma::vec {model.param.Kmat.diag()};
    // optional: force impurity ocupation |10>
    ek[0]=-10;
    ek[1]=10;
    auto fb=slater<double>(model, ek);
    fb.tol=1e-10;

    auto solver=Fbr_gs_spin(model,fb);

    cout<<"iteration m nActive energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    double n0 = solver.fb.correlator(0,0);
    double cd=2*solver.fb.correlator(0,1);
    cout<<0<<" "<<itensor::maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<cd<<" "<<solver.fb.p2-solver.fb.p1<<" "<<solver.energy<<" "<<t0.sincemark().wall<<endl;

    for(auto i=0;i<100;i++){
        auto [a,b]=solver.fb.interval_active_full();
        solver.fb.occupations_ni().as_row().eval().cols(a,b-1).eval().print("ni");
        solver.iterate(/*{.max_bond_dim=128}*/);
        double n0 = solver.fb.correlator(0,0);
        double cd=2*solver.fb.correlator(0,1);
        cout<<i+1<<" "<<itensor::maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<cd<<" "<<solver.fb.p2-solver.fb.p1<<" "<<solver.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
