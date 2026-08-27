#include "fbr/fbr_dyn.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace fbr;

int main()
{
    int L=100;
    Impurity model;
    {
        double U=0.2;
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        for(auto i=0; i<L-2; i++)
            K(i,i+2)=K(i+2,i)=0.5;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
        K(0,2)=K(2,0)=K(1,3)=K(3,1)=V;
        arma::mat Umat(L, L, arma::fill::zeros);
        Umat(0,1) = U;  // SIAM: U on (imp_up site 0, imp_dw site 1)
        std::vector<int> impPos = {2, 0, 1, 3};  // {buf_up, imp_up, imp_dw, buf_dw}
        model = Impurity {{.Kmat=K, .Umat=Umat, .impPos=impPos, .layout=spin_block}};
    }
    Fb_mps<cmpx> fb;
    {
        auto ek=arma::vec {model.param.Kmat.diag()};
        ek[L/2-1]=ek[L/2]=-10;
        ek[L/2-2]=ek[L/2+1]=10;
        fb=slater<cmpx>(model, ek);
    }

    double dt=0.1;
    auto solver=Fbr_dyn(model,fb,dt);
    solver.fb.tol=1e-12;

    arma::real(solver.K*1).eval().clean(1e-11).print("K initial");

    cout<<"time m <n0> <n1>  nActive\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i*dt<L; i++){
        solver.iterate({.epsilonM=0});  // epsilonM=0 -> no expansion; nKrylov inert, err_goal from default
        double n0= solver.fb.occupations_ni()(L/2);
        double n1= solver.fb.occupations_ni()(L/2+1);
        cout<<(i+1)*solver.dt<<" "<<maxLinkDim(solver.fb.psi)<<" "
            <<n0<<" "<<n1<<" "<<solver.fb.p2-solver.fb.p1<<endl;
        t0.mark();
    }
    return 0;
}
