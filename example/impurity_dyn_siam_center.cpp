#include "impurityMPS/impurity_dyn_spin.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;

/// return the kinetic energy in star geometry and the rotation to get it.
/// Layout: [spin-up bath | spin-up imp | spin-down imp | spin-down bath]
/// For spin-up the impurity is at the right end; for spin-down at the left end.
auto computeKstar(mat K, int nImp)
{
    int L=K.n_rows;
    int nBath=L/2-nImp/2;  // bath sites per spin

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);

    auto pos_up=regspace<uvec>(0,L/2-1);
    auto pos_dw=regspace<uvec>(L/2,L-1);

    for(int s : {0,1})
    {
        uvec pos   = s==0 ? pos_up : pos_dw;
        uvec pos_bath = s==0 ? pos.head(nBath)   : pos.tail(nBath);   // bath on left (up) or right (dw)
        uvec pos_impu = s==0 ? pos.tail(nImp/2)  : pos.head(nImp/2);  // imp on right (up) or left (dw)

        mat Kbath=K.submat(pos_bath,pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1,evec1,Kbath);
        uvec iek = s==0 ? sort_index(abs(ek1),"descend") : sort_index(abs(ek1));   // revert left bath
        arma::mat evec=evec1.cols(iek);
        arma::vec ek=ek1.rows(iek);

        arma::mat vk=K.submat(pos_impu,pos_bath).eval()*evec;
        Kstar.submat(pos_impu,pos_impu)=K.submat(pos_impu,pos_impu);

        for(auto j=0u;j<ek.size();j++) {
            int jj=pos_bath[j];
            Kstar(jj,jj)=ek[j];
            for(auto i=0u;i<pos_impu.size();i++) {
                int ii=pos_impu[i];
                Kstar(ii,jj)=Kstar(jj,ii)=vk(i,j);
            }
        }
        rot.cols(pos_bath)=rot.cols(pos_bath).eval()*evec;
    }

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
        int nBath=L/2-nImp/2;  // =4 for L=12, nImp=4
        arma::mat K(L,L, arma::fill::zeros);
        {
            for(auto i=0; i<L/2-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            for(auto i=L/2; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            K(nBath+nImp/2-1, nBath+nImp/2-1)=-U/2;
            K(L/2, L/2)=-U/2;
            K(nBath, nBath+nImp/2-1)=K(nBath+nImp/2-1, nBath)=V;
            K(L/2, L/2+nImp/2-1)=K(L/2+nImp/2-1, L/2)=V;
        }
        // L×L site-indexed Umat: SIAM Coulomb between innermost up (site nBath+nImp/2-1)
        // and innermost dw (site L/2).
        Umat.zeros(L, L);
        Umat(nBath+nImp/2-1, L/2) = U;

        std::tie(Kstar,rot) = computeKstar(K, nImp);
    }

    Fb_mps_spin<cmpx> fb;
    {
        int nBath=L/2-nImp/2;
        auto ek=arma::vec {Kstar.diag()};
        // force impurity occupation: physical imp sites occupied, buffer sites empty
        ek[nBath+nImp/2-1]=ek[L/2]=-10;    // spin-up and spin-down physical impurities
        ek[nBath]=ek[L/2+nImp/2-1]=10;     // spin-up and spin-down buffers
        fb=Fb_mps_spin<cmpx>::from_slater(rot*cmpx(1,0), ek, L/2, nImp);
    }

    // Construct model from pre-computed star geometry (bypassing toStar)
    ImpuritySpin model;
    {
        model.param.Kmat = Kstar;
        model.param.Umat = Umat;
        model.param.rot  = arma::mat(L,L, arma::fill::eye);
        // convention 2: {outer_up, inner_up, inner_dw, outer_dw} in Kstar layout.
        model.param.impPos = {L/2-2, L/2-1, L/2, L/2+1};
    }

    auto solver=Impurity_dyn_spin(model,fb,dt);
    solver.fb.tol=1e-12;

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

        solver.iterate({.max_bond_dim=2048, .nIter_diag=16,.epsilonM=1e-4});
        // double n0c = solver.fb.correlator(1,1).real();
        double n0= solver.fb.occupations_ni()(L/2);
        double n1= solver.fb.occupations_ni()(L/2+1);
        cout<<(i+1)*solver.dt<<" "<<maxLinkDim(solver.fb.psi)<<" "<<n0<<" "<<n1<<" "<<solver.fb.p2-solver.fb.p1<<endl;
        t0.mark();
    }
    return 0;
}
