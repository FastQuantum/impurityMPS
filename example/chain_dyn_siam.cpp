#include "impurityMPS/fb_mps.h"
#include "impurityMPS/impurity_param.h"
#include "impurityMPS/it_tdvp.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;


auto computeKstar(mat K, int nImp)
{
    int L=K.n_rows;
    auto pos_up = regspace<uvec>(0,2,L-1);
    auto pos_dw = regspace<uvec>(1,2,L-1);

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);

    for(auto pos : {pos_up,pos_dw})
    {
        uvec pos_bath = pos.subvec(nImp/2,L/2-1);
        uvec pos_impu = pos.subvec(0,nImp/2-1);

        mat Kbath = K.submat(pos_bath,pos_bath);
        mat evec1;
        vec ek1;
        eig_sym(ek1,evec1,Kbath);
        arma::uvec iek=arma::stable_sort_index(arma::abs(ek1));
        arma::mat evec=evec1.cols(iek);
        arma::vec ek=ek1.rows(iek);

        arma::mat vk=K.submat(pos_impu,pos_bath)*evec;

        Kstar.submat(pos_impu,pos_impu)=K.submat(pos_impu,pos_impu);
        for(auto j=0u;j<ek.size();j++) {
            int jj=pos_bath[j];
            Kstar(jj,jj)=ek[j];
            for(auto i=0u; i<pos_impu.size(); i++) {
                int ii=pos_impu[i];
                Kstar(ii,jj)=Kstar(jj,ii)=vk(i,j);
            }
        }
        rot.cols(pos_bath)=rot.cols(pos_bath).eval()*evec;
    } // pos

    return make_pair(Kstar,rot);
}

int main()
{
    int L=12;
    int nImp;
    mat Kstar, Umat, rot;
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
        Umat.zeros(4,4);
        Umat(0,1)=U;

        nImp=Umat.n_rows;
        std::tie(Kstar,rot) = computeKstar(K, nImp);
    }
    Fb_mps<cmpx> fb;
    {
        auto ek=arma::vec {Kstar.diag()};
        // force impurity ocupation |1100>
        ek[0]=ek[1]=-10; //TODO: the ek change the Hamiltonian
        ek[2]=ek[3]=10;
        fb=Fb_mps<cmpx>::from_slater(rot*cmpx(1,0), ek, L/2, nImp, false);
        // fb.occupations_ni().as_row().eval().print("ni");
        // fb.occupations_ni2().as_row().eval().print("ni2");
    }

    itensor::MPO mpo;
    {
        itensor::AutoMPO h(fb.sites);
        for(auto i=0; i<nImp; i++)
            for(auto j=0; j<nImp; j++) {
                int ii=i;
                int jj=j;
                if (std::abs(Umat(i,j))>1e-15)
                    h += Umat(i,j), "N", ii+1, "N", jj+1;
            }

        for(auto i=0; i<L; i++)
            for(auto j=0; j<L; j++)
                if (std::abs(Kstar(i,j))>fb.tol)
                    h += Kstar(i,j),"Cdag",i+1,"C",j+1;
        mpo=itensor::toMPO(h);
    }

    double dt=0.1;
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024;
    sweeps.cutoff() = fb.tol;
    sweeps.niter() = 16;
    sweeps.noise() = 0e-8;

    cout<<"iteration m 0 energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i*dt<L;i++){
        {
            std::vector<double> epsilonK(15, 1e-4);   // match epsilonM from impurity_dyn
            itensor::addBasis(fb.psi, mpo, epsilonK,
                              {"Cutoff", 1e-4,
                               "Method", "DensityMatrix",
                               "KrylovOrd", 15,
                               "DoNormalize", true,
                               "Quiet", true,
                               "Silent", true});
        }
        double energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,          // TDVP sweep
                               {"Truncate", true,
                                "DoNormalize", true,
                                "Quiet", true,
                                "Silent", true,
                                "NumCenter", 2,
                                "ErrGoal", 1e-8});


        double n0=itensor::expectC(fb.psi,fb.sites,"N",{1})[0].real();
        double n1=itensor::expectC(fb.psi,fb.sites,"N",{3})[0].real();
        // fb.occupations_ni2().as_row().eval().print("ni2");
        cout<<(i+1)*dt<<" "<<itensor::maxLinkDim(fb.psi)<<" "<<n0<<" "<<n1<<" "<<energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
