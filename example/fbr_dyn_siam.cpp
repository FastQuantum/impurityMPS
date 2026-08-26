#include "fbr/fbr_dyn.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace fbr;

int main()
{
    int L=1000;
    Impurity model;
    {
        double U=0.1;
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        {
            for(auto i=0; i<L-2; i++)
                K(i,i+2)=K(i+2,i)=0.5;
            K(0,0)=-U/2;
            K(1,1)=-U/2;
            K(0,2)=K(2,0)=K(1,3)=K(3,1)=V;
        }
        // L×L Umat, site-indexed. SIAM Coulomb between physical up imp (site 0) and dw imp (site 1).
        arma::mat Umat(L, L, arma::fill::zeros);
        Umat(0,1) = U;
        // Convention 2 (outer..inner..outer): {buf_up, imp_up, imp_dw, buf_dw} = {2, 0, 1, 3}.
        std::vector<int> impPos = {2, 0, 1, 3};

        model = Impurity {{.Kmat=K, .Umat=Umat, .impPos=impPos, .layout=spin_symmetric}};
    }
    Fb_mps<cmpx> fb;
    {
        auto ek=arma::vec {model.param.Kmat.diag()};
        // force impurity ocupation |1100>
        ek[L/2-1]=ek[L/2]=-10;
        ek[L/2-2]=ek[L/2+1]=10;
        arma::cx_mat rot(L,L,arma::fill::eye);
        fb=model.slater<cmpx>(ek);
        // fb.occupations_ni().as_row().eval().print("ni");
    }
    // fb.tol=1e-10;

    double dt=0.1;
    auto solver=Fbr_dyn(model,fb,dt);

    // arma::real(fb.rot*1).eval().clean(1e-11).print("fb.rot");
    // arma::real(model.param.rot*1).eval().clean(1e-11).print("param.rot");
    // auto Q=solver.param.rot;
    // arma::real(solver.K*1).eval().clean(1e-11).print("K inicial");
    // arma::real(solver.param.Kmat*1).eval().clean(1e-11).print("Kmat original");
    // arma::real(solver.Kip0*1).eval().clean(1e-11).print("Kip0 before main() iterations");
    // terminate();

    cout<<"time m <n0> <cd>  nActive\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i*dt<L; i++){
        // arma::real(solver.K*1).eval().clean(1e-11).print("K");
        // auto [a,b]=solver.fb.interval_active_full();
        // solver.fb.occupations_ni().as_row().eval().cols(a,b-1).eval().print("ni");

        solver.iterate({.epsilonM=0e-8});
        // double n0 = solver.fb.correlator(1,1).real();
        double n0= solver.fb.occupations_ni()(L/2);
        double n1= solver.fb.occupations_ni()(L/2+1);
        cout<<(i+1)*solver.dt<<" "<<maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<n1<<" "<<solver.fb.p2-solver.fb.p1
             <<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
