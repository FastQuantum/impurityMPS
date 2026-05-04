#include "impurityMPS/impurity_dyn.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;

/// return the kinetic energy in star geometry and the rotation to get it.
auto computeKstar(mat K, int nImp)
{
    int L=K.n_rows;
    auto pos_up = regspace<uvec>(0,2,L-1);
    auto pos_dw = regspace<uvec>(1,2,L-1);

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);

    for(auto pos : {pos_up,pos_dw})
    {
        uvec pos_bath = pos.subvec(nImp/2,L/2-1); // these two uvec can change if the impurity is in the center
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
            int jj=pos_bath[iek[j]];
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
    int L=100;
    int nImp=4;
    double dt=0.1;

    mat Kstar, Umat;
    mat rot;
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
        Umat.zeros(nImp,nImp);
        Umat(0,1)=U;

        std::tie(Kstar,rot) = computeKstar(K, nImp);
    }

    Fb_mps<cmpx> fb;
    {
        auto ek=arma::vec {Kstar.diag()};
        // force impurity occupation: physical imp sites occupied, buffer sites empty
        ek[0]=ek[1]=-10;    // spin-up and spin-down physical impurities
        ek[2]=ek[3]=10;     // spin-up and spin-down buffers
        fb=Fb_mps<cmpx>::from_slater(rot*cmpx(1,0), ek, L/2, nImp,false);
    }

    // Construct model from pre-computed star geometry (bypassing toStar)
    Impurity model;
    {
        model.param.Kmat = Kstar;
        model.param.Umat = Umat;
        model.param.rot  = rot;//arma::mat(L,L,arma::fill::eye);
        // impurity cluster sits at the same positions in Kstar as in K (computeKstar does not move them)
        model.param.impPos = iota(nImp);
    }

    auto solver=Impurity_dyn(model,fb,dt);
    solver.fb.tol=1e-12;

    // arma::real(fb.rot*1).eval().clean(1e-11).print("fb.rot");
    // arma::real(model.param.rot*1).eval().clean(1e-11).print("param.rot");
    // auto Q=solver.param.rot;
    arma::real(solver.K*1).eval().clean(1e-11).print("K inicial ns");
    // arma::real(solver.param.Kmat*1).eval().clean(1e-11).print("Kmat original");
    // arma::real(solver.Kip0*1).eval().clean(1e-11).print("Kip0 before main() iterations");
    // terminate();

    cout<<"time m <n0> <cd>  nActive\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0; i*dt<L; i++){
        // arma::real(solver.K*1).eval().clean(1e-11).print("K");
        // auto [a,b]=solver.fb.interval_active_full();
        // solver.fb.occupations_ni().as_row().eval().cols(a,b-1).eval().print("ni");

        solver.iterate({.max_bond_dim=2048, .nIter_diag=16,.epsilonM=1e-4});
        // double n0 = solver.fb.correlator(1,1).real();
        double n0= solver.fb.occupations_ni2()(0);
        double n1= solver.fb.occupations_ni2()(2);
        cout<<(i+1)*solver.dt<<" "<<itensor::maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<n1<<" "<<solver.fb.nActive<<endl;
        t0.mark();
    }
    return 0;
}
