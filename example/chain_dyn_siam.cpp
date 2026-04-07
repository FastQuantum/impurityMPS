#include "impurityMPS/fb_mps_spin.h"
#include "impurityMPS/impurity_param_spin.h"
#include "impurityMPS/it_tdvp.h"
#include "impurityMPS/it_tdvp.h"
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    int L=12;
    ImpuritySpin model;
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
        arma::mat Umat(4,4,arma::fill::zeros);
        Umat(0,1)=U;

        model = ImpuritySpin {{.Kmat=K, .Umat=Umat}};
    }
    Fb_mps_spin<cmpx> fb;
    {
        auto ek=arma::vec {model.param.Kmat.diag()};
        // force impurity ocupation |1100>
        ek[L/2-1]=ek[L/2]=-10;
        ek[L/2-2]=ek[L/2+1]=10;
        fb=Fb_mps_spin<cmpx>::from_slater(model.param.rot*cmpx(1,0), ek, model.param.nPart(), model.param.nImp());
        fb.occupations_ni().as_row().eval().print("ni");
        fb.occupations_ni2().as_row().eval().print("ni2");
    }

    itensor::MPO mpo;
    {
        auto &param=model.param;
        itensor::AutoMPO h(fb.sites);
        auto impPos=param.impPos();
        for(auto i=0; i<param.nImp(); i++)
            for(auto j=0; j<param.nImp(); j++) {
                int ii=impPos[i];
                int jj=impPos[j];
                if (std::abs(param.Umat(i,j))>1e-15)
                    h += param.Umat(i,j), "N", ii+1, "N", jj+1;
            }

        for(auto i=0; i<L; i++)
            for(auto j=0; j<L; j++)
                if (std::abs(param.Kmat(i,j))>fb.tol)
                    h += param.Kmat(i,j),"Cdag",i+1,"C",j+1;
        mpo=itensor::toMPO(h);
    }

    double dt=0.1;
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024;
    sweeps.cutoff() = fb.tol;
    sweeps.niter() = 16;
    sweeps.noise() = 1e-8;
    {
        std::vector<double> epsilonK(15,1E-8);  // Global subspace expansion
        itensor::addBasis(fb.psi,mpo,epsilonK,
                          {"Cutoff", 1e-4,
                           "Method", "DensityMatrix",
                           "KrylovOrd", 15,
                           "DoNormalize", true,
                           "Quiet", true,
                           "Silent", true});
    }

    cout<<"iteration m 0 energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<10;i++){
        double energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,          // TDVP sweep
                               {"Truncate", true,
                                "DoNormalize", true,
                                "Quiet", true,
                                "Silent", true,
                                "NumCenter", 2,
                                "ErrGoal", 1e-8});


        double n0=itensor::expectC(fb.psi,fb.sites,"N",{L/2})[0].real();
        double n02=fb.occupations_ni2()[L/2-1];
        fb.occupations_ni2().as_row().eval().print("ni2");
        cout<<(i+1)*dt<<" "<<itensor::maxLinkDim(fb.psi)<<" "<<n0<<" "<<n02<<" "<<energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
