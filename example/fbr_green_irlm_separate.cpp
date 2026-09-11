// Greater Green function of the spinless IRLM, evolving the two states in
// SEPARATE frames and aligning only at the measurement -- checked against the
// exact non-interacting result.
//
//     G(i,j,t) = -i <psi0| c_i(t) c_j^dag(0) |psi0>,   c_i(t)=e^{iHt}c_i e^{-iHt}
//
// With A=|psi0> and B_j=c_j^dag|psi0> evolved by the same H,
//     G(i,j,t) = -i <A(t)| c_i |B_j(t)>.
//
// fbr_green_irlm.cpp puts A and B in ONE shared basis (Fbr_dyn_shared) so the
// matrix element is a bare MPS overlap; the shared window then has to hold the
// union of both states' natural orbitals at every step and grows fast. Here each
// state is evolved by its OWN Fbr_dyn, in its own small window, and the two are
// brought into a common basis only at the measurement, by green_overlap.h's
// c_element(). The two runs share the model (hence the star frame and bath
// phase) and are stepped in lockstep (same n_iter), so the interaction-picture
// phase is common and cancels, leaving the plain relative frame A.rot^dag B.rot.
//
// Usage: fbr_green_irlm_separate [U [L [nStep]]]   (the reference is exact at U=0)

#include "fbr/fbr.h"
#include "fbr/green_overlap.h"

#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {

/// c_j^dag|psi0>, normalized, with the norm it had before normalizing.
std::pair<Fb_mps<cmpx>,double> add_particle(Fb_mps<cmpx> const& psi0, int j)
{
    auto state=psi0;
    state.apply_local_op("Cdag",j);
    double nrm=std::sqrt(std::real(itensor::innerC(state.psi,state.psi)));
    state.psi.normalize();
    state.update_cc();
    return {state,nrm};
}

} // namespace

int main(int argc, char** argv)
{
    double U = argc>1 ? std::stod(argv[1]) : 0.0;
    int L    = argc>2 ? std::stoi(argv[2]) : 40;
    int nStep= argc>3 ? std::stoi(argv[3]) : 40;
    double V=0.5;          // impurity-bath hybridization
    double dt=0.05;
    int n_part=L/2;

    arma::mat K(L,L, arma::fill::zeros);
    {
        for(auto i=1; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=0.5;
        K(0,1)=K(1,0)=V;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
    }
    arma::mat Umat(L,L,arma::fill::zeros);
    Umat(0,1)=U;

    // exact non-interacting reference, from the real-space K
    vec ek_exact;
    mat evec_exact;
    eig_sym(ek_exact,evec_exact,K);
    auto G_exact=[&](int i,int j,double t) {
        cmpx g=0;
        for(auto a=n_part; a<L; a++)   // unoccupied modes only
            g += std::exp(-imag_1*ek_exact[a]*t)*evec_exact(i,a)*evec_exact(j,a);
        return -imag_1*g;
    };

    auto model = ImpurityParam{.Kmat=K, .Umat=Umat, .imp_pos={0,1}};
    model.to_star();

    // ---- ground state ----
    auto gs=Fb_mps<double>::from_slater(model.rot,
                                        vec{model.Kmat.diag()},
                                        n_part, model.n_imp(), leading);
    gs.tol=1e-12;
    auto gs_solver=Fbr_gs(model,gs);
    for(auto i=0; i<60; i++) gs_solver.iterate({.max_bond_dim=256});
    cout<<setprecision(12)
        <<"# ground state energy: fbr="<<gs_solver.energy
        <<"  exact="<<arma::sum(ek_exact.head(n_part))<<endl;

    // ---- one run per matrix element, two INDEPENDENT solvers per run ----
    auto psi0=gs_solver.fb.to_complex();
    psi0.tol=1e-12;

    struct Run { Fbr_dyn A, B; double nrm; };
    auto run_for=[&](int j) {
        auto [Bj,nrm]=add_particle(psi0,j);
        Bj.tol=psi0.tol;
        return Run{ Fbr_dyn(model,psi0,dt), Fbr_dyn(model,Bj,dt), nrm };
    };
    auto run0=run_for(0);
    auto run1=run_for(1);

    // the impurity sites must be single MPS orbitals for c_element to be local
    for(auto i : {0,1}) {
        auto Q=run0.A.effective_rot();
        if (std::abs(std::abs(Q(i,i))-1)>1e-10)
            throw std::runtime_error("impurity site is not a single MPS orbital");
    }

    cout<<"# t  ReG00 ImG00  ReG00_exact ImG00_exact  ReG01 ImG01  ReG01_exact ImG01_exact"
        <<"   nActive(A0 B0 A1 B1)  chi_align\n"
        <<setprecision(6)<<fixed;
    // The rotated MPS is a throwaway (only the scalar matrix element is kept),
    // so the frame-alignment circuit is contracted at a loose cutoff.
    const double meas_cutoff=1e-4;
    double err=0;
    for(auto step=0; step<=nStep; step++) {
        double t=step*dt;
        cmpx G00=-imag_1*run0.nrm*c_element(run0.A.fb,run0.B.fb,0,meas_cutoff);
        cmpx G01=-imag_1*run1.nrm*c_element(run1.A.fb,run1.B.fb,0,meas_cutoff);
        cmpx G00e=G_exact(0,0,t), G01e=G_exact(0,1,t);
        err=std::max({err,std::abs(G00-G00e),std::abs(G01-G01e)});

        if (step%5==0) {
            // bond dimension the loose alignment circuit carries at the measurement
            auto Bl=run1.B.fb; align_to_frame(Bl,run1.A.fb.rot,meas_cutoff);
            cout<<t
                <<"  "<<G00.real()<<" "<<G00.imag()
                <<"  "<<G00e.real()<<" "<<G00e.imag()
                <<"  "<<G01.real()<<" "<<G01.imag()
                <<"  "<<G01e.real()<<" "<<G01e.imag()
                <<"   "<<run0.A.fb.n_active()<<" "<<run0.B.fb.n_active()
                <<" "<<run1.A.fb.n_active()<<" "<<run1.B.fb.n_active()
                <<"   chi_align="<<itensor::maxLinkDim(Bl.psi)<<endl;
        }

        if (step<nStep) {
            run0.A.iterate({.epsilon_M=0}); run0.B.iterate({.epsilon_M=0});
            run1.A.iterate({.epsilon_M=0}); run1.B.iterate({.epsilon_M=0});
        }
    }
    cout<<"# max |G - G_exact| = "<<scientific<<err
        <<(U==0 ? "  (U=0: the reference is exact)"
                : "  (U!=0: the non-interacting reference does not apply)")<<endl;
    return 0;
}
