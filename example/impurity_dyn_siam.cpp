#include "impurityMPS/impurity_dyn_spin.h"
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    int L=12;
    ImpuritySpin model;
    {
        double U=2;
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        {
            for(auto i=0; i<L-2; i++)
                K(i,i+2)=K(i+2,i)=0.5;
            K(0,0)=-U/2;
            K(1,1)=-U/2;
            K(0,2)=K(2,0)=K(1,3)=K(3,1)=V;
        }
        arma::mat Umat(4,4,arma::fill::zeros);
        Umat(0,1)=U;

        // arma::real(K*1).eval().clean(1e-11).print("K original");

        model = ImpuritySpin {{.Kmat=K, .Umat=Umat}};
    }
    Fb_mps_spin<cmpx> fb;
    {
        auto ek=arma::vec {model.param.Kmat.diag()};
        // force impurity ocupation |1100>
        ek[L/2-1]=ek[L/2]=-10;
        ek[L/2-2]=ek[L/2+1]=10;
        arma::cx_mat rot(L,L,arma::fill::eye);
        fb=Fb_mps_spin<cmpx>::from_slater(model.param.rot*cmpx(1,0), ek, model.param.nPart(), model.param.nImp());
        // fb.occupations_ni().as_row().eval().print("ni");
    }

    double dt=0.1;
    auto solver=Impurity_dyn_spin(model,fb,dt);
    solver.fb.tol=1e-12;

    // arma::real(fb.rot*1).eval().clean(1e-11).print("fb.rot");
    // arma::real(model.param.rot*1).eval().clean(1e-11).print("param.rot");
    // auto Q=solver.param.rot;
    // arma::real(K*1).eval().clean(1e-11).print("K original");
    // arma::real(solver.param.Kmat*1).eval().clean(1e-11).print("Kmat original");
    // arma::real(solver.Kip0*1).eval().clean(1e-11).print("Kip0 before main() iterations");
    // terminate();

    cout<<"time m <n0> <cd>  nActive\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i<4/*i*dt<L*/; i++){
        // arma::real(solver.K*1).eval().clean(1e-11).print("K");
        auto [a,b]=solver.fb.interval_active_full();
        // solver.fb.occupations_ni().as_row().eval().cols(a,b-1).eval().print("ni");

        solver.iterate({.max_bond_dim=2048, .nIter_diag=16,.epsilonM=1e-4});
        // double n0 = solver.fb.correlator(1,1).real();
        double n02= solver.fb.occupations_ni()(L/2);
        double cd=2*solver.fb.correlator(0,1).real();
        cout<<(i+1)*solver.dt<<" "<<maxLinkDim(solver.fb.psi)<<" "<<n02<<" "<<cd<<" "<<solver.fb.p2-solver.fb.p1<<endl;
        t0.mark();
    }
    return 0;
}
