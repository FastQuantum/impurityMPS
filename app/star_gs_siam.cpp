#include "fbr/impurity_param.h"
#include "fbr/initial_state.h"
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
    arma::mat Umat(L,L,arma::fill::zeros);
    Umat(0,1)=U;
    auto model = ImpurityParam{.Kmat=K, .Umat=Umat, .imp_pos={0,1}};
    model.to_star();

    auto ek=arma::vec {model.Kmat.diag()};
    // optional: force impurity ocupation |10>
    ek[0]=-10;
    ek[1]=10;
    auto fb=slater<double>(model, ek);
    itensor::AutoMPO h(fb.sites);
    for(auto i=0; i<model.n_imp(); i++)
        for(auto j=0; j<model.n_imp(); j++)
            if (std::abs(model.Umat(i,j))>1e-15)
                h += model.Umat(i,j), "N", i+1, "N", j+1;
    for(auto i=0; i<model.Kmat.n_rows; i++)
        for(auto j=0; j<model.Kmat.n_cols; j++)
            if (std::abs(model.Kmat(i,j))>fb.tol)
                h += model.Kmat(i,j),"Cdag",i+1,"C",j+1;
    auto mpo = itensor::toMPO(h);

    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024;
    sweeps.cutoff() = 1e-10;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-8;

    cout<<"iteration m 0 energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<100;i++){
        double energy=itensor::dmrg(fb.psi,mpo,sweeps, {/*"MaxSite",fb.n_active,*/"Quiet", true, "Silent", true});
        double n0=itensor::expect(fb.psi,fb.sites,"N",{1})[0];
        cout<<i+1<<" "<<itensor::maxLinkDim(fb.psi)<<" "<<n0<<" "<<energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
