// Greater Green function of the spinless IRLM, checked against the exact
// non-interacting result.
//
//     G(i,j,t) = -i <psi0| c_i(t) c_j^dag(0) |psi0>,   c_i(t)=e^{iHt}c_i e^{-iHt}
//
// Writing A=|psi0> and B_j=c_j^dag|psi0> and evolving both with the same H,
//     <A(t)| c_i |B_j(t)> = <psi0| e^{iHt} c_i e^{-iHt} c_j^dag |psi0>,
// so G is a matrix element between two states, which have to share the SAME
// orbital basis at every time. That is what Fbr_dyn_shared gives: the orbital
// rotations are found once and applied to every state.
//
// Each element of G needs exactly two states, so run one Fbr_dyn_shared per
// element -- {psi0, c_0^dag psi0} here, and {psi0, c_1^dag psi0} there -- rather
// than putting all three in one basis. A shared window must be wide enough for
// every state in it, so a third state can only widen it, and a wider window is
// more entanglement for every state to carry. Measured at L=40, U=0.5: the
// three-state window reaches 28 active orbitals by t=1 and 26 at t=1.5, where
// the two pairs need 13 and 12. The price is evolving psi0 twice, which is two
// independent runs that could as well be two processes.
//
// The impurity orbitals are never rotated, so the real-space site i is exactly
// the MPS orbital i (checked below) and c_i is a local operator there:
//     <A| c_i |B> = <c_i^dag A | B>
// is then a plain MPS overlap. (An MPO for c_i would carry one unit of Nf flux,
// which ITensor's AutoMPO cannot build with particle-number conservation.)
//
// For U=0 the Hamiltonian is quadratic, c_i(t)=sum_k [e^{-iKt}]_ik c_k, and with
// the ground state filling the n_part lowest modes of K = V diag(e) V^T,
//     G(i,j,t) = -i sum_{a unoccupied} e^{-i e_a t} V_ia V_ja,
// which is the reference printed next to the computed values.
//
// At U=0, where the reference is exact, G agrees with it to ~1e-5 over the
// whole run. The ground state comes from Fbr_gs, so its frame is not the star
// frame any more; the dynamics takes the star frame from the model itself, so
// that is fine.
//
// Usage: fbr_green_irlm [U]     (the reference is only exact at U=0)

#include "fbr/fbr.h"

#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {

/// <A| c_i |B>, for i a non-rotating impurity site.
cmpx c_element(Fb_mps<cmpx> const& A, Fb_mps<cmpx> const& B, int i)
{
    auto Ai=A;
    Ai.apply_local_op("Cdag",i);          // |c_i^dag A>
    return itensor::innerC(Ai.psi,B.psi);
}

/// c_j^dag|psi0>, normalized, together with the norm it had before normalizing.
/// The states of one Fbr_dyn_shared share their Slater determinant, so the state
/// has to be normalized; its norm goes back into G afterwards.
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
    int L=40;
    double U = argc>1 ? std::stod(argv[1]) : 0.0;
    double V=0.5;          // impurity-bath hybridization
    double dt=0.05;
    int nStep=40;
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

    // ---- one run per matrix element: {psi0, c_j^dag psi0} ----
    // The two states of a run share one active window, which has to hold every
    // orbital where they differ; Fbr_dyn_shared widens it for that. A tight
    // tolerance keeps the orbitals the extra particle leaks into inside it.
    auto psi0=gs_solver.fb.to_complex();
    psi0.tol=1e-12;
    auto pair_for=[&](int j) {
        auto [B,nrm]=add_particle(psi0,j);
        B.tol=psi0.tol;
        return std::make_pair(Fbr_dyn_shared(model,std::vector{psi0,B},dt),nrm);
    };
    auto [solver0,nrm0]=pair_for(0);
    auto [solver1,nrm1]=pair_for(1);

    // the impurity sites must be single MPS orbitals for c_element to be local
    for(auto i : {0,1}) {
        auto Q=solver0.effective_rot();
        if (std::abs(std::abs(Q(i,i))-1)>1e-10)
            throw std::runtime_error("impurity site is not a single MPS orbital");
    }

    cout<<"# t  ReG00 ImG00  ReG00_exact ImG00_exact  ReG01 ImG01  ReG01_exact ImG01_exact\n"
        <<setprecision(6)<<fixed;
    double err=0;
    for(auto step=0; step<=nStep; step++) {
        double t=step*dt;
        cmpx G00=-imag_1*nrm0*c_element(solver0.states[0],solver0.states[1],0);
        cmpx G01=-imag_1*nrm1*c_element(solver1.states[0],solver1.states[1],0);
        cmpx G00e=G_exact(0,0,t), G01e=G_exact(0,1,t);
        err=std::max({err,std::abs(G00-G00e),std::abs(G01-G01e)});

        if (step%5==0)
            cout<<t
                <<"  "<<G00.real()<<" "<<G00.imag()
                <<"  "<<G00e.real()<<" "<<G00e.imag()
                <<"  "<<G01.real()<<" "<<G01.imag()
                <<"  "<<G01e.real()<<" "<<G01e.imag()<<endl;

        if (step<nStep) {
            solver0.iterate({.epsilon_M=0});
            solver1.iterate({.epsilon_M=0});
        }
    }
    cout<<"# max |G - G_exact| = "<<scientific<<err
        <<(U==0 ? "  (U=0: the reference is exact)"
                : "  (U!=0: the non-interacting reference does not apply)")<<endl;
    return 0;
}
