#include "fbr/fb_mps.h"
#include "fbr/fbr_param.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace fbr;

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
    auto model = Fbr {{.Kmat=K, .Umat=Umat}};

    auto ek=arma::vec {model.param.Kmat.diag()};
    // optional: force impurity ocupation |10>
    ek[0]=-10;
    ek[1]=10;
    auto fb=Fb_mps<double>::from_slater(model.param.rot, ek, model.param.nPart(), model.param.nImp(), spin);
    // fb.natOrbDepth=10;
    itensor::AutoMPO h(fb.sites);
    for(auto i=0; i<model.param.nImp(); i++)
        for(auto j=0; j<model.param.nImp(); j++)
            if (std::abs(model.param.Umat(i,j))>1e-15)
                h += model.param.Umat(i,j), "N", i+1, "N", j+1;
    for(auto i=0; i<model.param.Kmat.n_rows; i++)
        for(auto j=0; j<model.param.Kmat.n_cols; j++)
            if (std::abs(model.param.Kmat(i,j))>fb.tol)
                h += model.param.Kmat(i,j),"Cdag",i+1,"C",j+1;
    auto mpo = itensor::toMPO(h);

    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024;
    sweeps.cutoff() = 1e-10;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-8;

    cout<<"iteration m 0 energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<100;i++){
        double energy=itensor::dmrg(fb.psi,mpo,sweeps, {/*"MaxSite",fb.nActive,*/"Quiet", true, "Silent", true});
        double n0=itensor::expect(fb.psi,fb.sites,"N",{1})[0];
        cout<<i+1<<" "<<itensor::maxLinkDim(fb.psi)<<" "<<n0<<" "<<energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
