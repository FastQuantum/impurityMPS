#include "fbr/fbr_dyn.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using namespace fbr;

int main()
{
    int L=100;
    Impurity model;
    {
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
        arma::mat Umat(L,L,arma::fill::zeros);
        Umat(0,1)=U;

        model = Impurity {{.Kmat=K, .Umat=Umat, .impPos={0,1,2,3}}};

        K.print("Kmat before star ns");
    }
    Fb_mps<cmpx> fb;
    {
        auto ek=arma::vec {model.param.Kmat.diag()};
        // force impurity ocupation |1100>
        ek[0]=ek[1]=-10; //TODO: the ek change the Hamiltonian
        ek[2]=ek[3]=10;
        fb=Fb_mps<cmpx>::from_slater(model.param.rot*cmpx(1,0), ek, model.param.nPart(), model.param.nImp(), leading, false);
        // fb.occupations_ni().as_row().eval().print("ni");
        // fb.occupations_ni2().as_row().eval().print("ni2");
    }

    double dt=0.1;
    auto solver=Fbr_dyn(model,fb,dt);
    solver.fb.tol=1e-12;

    // arma::real(fb.rot*1).eval().clean(1e-11).print("fb.rot");
    // arma::real(model.param.rot*1).eval().clean(1e-11).print("param.rot");
    // auto Q=solver.param.rot;
    arma::real(solver.K*1).eval().clean(1e-11).print("K inicial ns");
    // arma::real(solver.param.Kmat*1).eval().clean(1e-11).print("Kmat original");
    // arma::real(solver.Kip0*1).eval().clean(1e-11).print("Kip0 before main() iterations");
    // terminate();

    cout<<"time m <n0> <cd> nActive\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i*dt<L; i++){
        solver.iterate({.max_bond_dim=2048, .epsilonM=1e-4});
        double n0= solver.fb.occupations_ni2()(0);
        double n1= solver.fb.occupations_ni2()(2);
        cout<<(i+1)*solver.dt<<" "<<itensor::maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<n1<<" "<<solver.fb.nActive()<<endl;
        t0.mark();
    }
    return 0;
}
