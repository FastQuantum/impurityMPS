#ifndef FBR_DYN_H
#define FBR_DYN_H

#include "graph.h"
#include "itensor_utils.h"
#include "impurity_param.h"
#include "fb_mps.h"
#include "initial_state.h"

#include "tdvp.h"
#include "basisextension.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace fbr {

namespace detail {

/// Rank of a coupling block: number of singular values above tol (relative).
inline int sv_rank(arma::mat const& block, double tol)
{
    if (block.n_rows==0 || block.n_cols==0) return 0;
    arma::vec s=arma::svd(block);
    if (s.empty() || s[0]==0) return 0;
    return (int)arma::find(s>tol*s[0]).eval().size();
}

/// Rank of the impurity-bath coupling block of Kmat: the largest over the spin
/// sectors, so both grow the window by the same amount even when their coupling
/// ranks differ (spin_block). An extra, weakly coupled representative costs one
/// orbital; two different window widths would cost a special case everywhere.
inline int coupling_rank(Fb_mps<cmpx> const& fb, arma::mat const& Kmat)
{
    int rank=0;
    for (Spin s : {up,dw}) {
        auto [a_imp,b_imp]=fb.interval_impurity(s);
        auto [a_sla,b_sla]=fb.interval_slater(s);
        if (a_imp>=b_imp || a_sla>=b_sla) continue;
        rank=std::max(rank,sv_rank(Kmat.submat(a_imp,a_sla,b_imp-1,b_sla-1),fb.tol));
    }
    return rank;
}

/// Machinery shared by the single-state (Fbr_dyn) and multi-state (Fbr_dyn_shared)
/// dynamics solvers: the interaction picture of the diagonal bath Hamiltonian
/// and the frame algebra built on top of it. The MPS never sees the bath
/// phases, fb.rot tracks only the natural-orbital basis change, and
/// Schrödinger-picture correlators are recovered by dressing with
/// exp(-i Kbath t) (effective_rot).
struct DynCommon {
    using State = Fb_mps<cmpx>;

    ImpurityParam param;
    double dt;
    arma::cx_mat Kbath;      ///< diagonal bath Hamiltonian (star geometry)
    arma::cx_mat Kip0;       ///< second-order interaction-picture Hamiltonian at t=0
    arma::uvec imp_pos;      ///< impurity positions (star layout)
    arma::uvec bath_pos;     ///< bath positions (star layout)
    arma::cx_mat rot_star;       ///< the star frame, the basis Kip0 and Kbath are written in
    int n_sv=0;               ///< fixed rank of the impurity–bath coupling

    /// these quantities are updated during the iterations
    arma::cx_mat K;          ///< the current Hamiltonian
    int n_iter=0;

    DynCommon(ImpurityParam const& param_, State const& first, double dt_)
        : param(param_)
        , dt(dt_)
    {
        param.validate();   // a model built directly in star geometry never saw to_star()
        int L=param.length();
        if (first.sites.length()!=L)
            throw std::invalid_argument("Fbr_dyn: state length does not match the model");
        imp_pos = arma::conv_to<arma::uvec>::from(param.imp_pos);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L,param.imp_pos));

        // The state layout must put its impurity where the model does: this is
        // what tells a centered layout apart from a leading one.
        auto [a_imp,b_imp]=first.interval_impurity_full();
        if (imp_pos.empty() || b_imp-a_imp!=param.n_imp()
            || (int)imp_pos.min()!=a_imp || (int)imp_pos.max()!=b_imp-1)
            throw std::invalid_argument("Fbr_dyn: the state layout does not match the impurity positions of the model");

        arma::mat Kstar=param.Kmat;
        if (!bath_pos.empty()) {
            arma::mat Kb=Kstar.submat(bath_pos,bath_pos);
            if (arma::abs(Kb-arma::diagmat(Kb.diag())).max()>1e-12)
                throw std::invalid_argument("Fbr_dyn: the bath block of Kmat must be diagonal (star geometry, see to_star)");
        }

        // Star geometry: the bath-bath block of Kstar is diagonal. Let d be
        // that diagonal embedded in an L-vector (zero on impurity sites).
        // With D=diag(d), the commutator entries are
        //   (Kstar*D - D*Kstar)(i,j) = Kstar(i,j)*(d(j)-d(i)),
        // so column/row scaling gives it in O(L^2) instead of the two dense
        // O(L^3) products Kstar*Kbath_full and Kbath_full*Kstar.
        arma::vec d(L,arma::fill::zeros);
        d(bath_pos)=arma::vec(Kstar.diag())(bath_pos);
        arma::mat c1=Kstar; c1.each_row() %= d.t();   // Kstar*D
        arma::mat c2=Kstar; c2.each_col() %= d;       // D*Kstar

        Kip0 = Kstar * cmpx(1,0);
        Kip0(bath_pos,bath_pos).zeros();              // Kstar - Kbath_full (arrow)
        Kip0 -= cmpx(0,0.5*dt) * (c1-c2);

        // The star frame, which is what Kip0 and Kbath are written in. It is
        // the model's own frame, NOT the initial state's: the two agree when the
        // state comes straight from from_slater(param.rot,...), but not when it
        // has already been rotated (a ground state from Fbr_gs, say). Taking it
        // from the model makes rot_star^dag * fb.rot express the current orbitals in
        // the star basis whatever frame the state starts in.
        rot_star = param.rot * cmpx(1,0);
        Kbath = Kstar.submat(bath_pos,bath_pos) * cmpx(1,0);

        // Fix n_sv = rank of the impurity–bath coupling block at construction.
        // The same value is used for every representative plan thereafter.
        n_sv = coupling_rank(first,param.Kmat);
    }

    /// Diagonal of the interaction-picture phase exp(-i H_bath * n*dt).
    /// In star geometry H_bath is diagonal, so this is computed in O(L_bath),
    /// avoiding the O(L^3) dense matrix exponential.
    arma::cx_vec ip_phase(int n) const
    {
        arma::cx_vec d(param.length(),arma::fill::ones);
        if (n > 0)
            d(bath_pos) = arma::exp(-imag_1 * Kbath.diag() * (static_cast<double>(n)*dt));
        return d;
    }

    /// Interaction-picture Hamiltonian K = rot^dag * Kip0 * rot, with
    /// rot = diag(ip_phase) * rot_star^dag * fb.rot.
    /// O(L^2): Kip0 is Hermitian and its bath-bath block is exactly zero (a "cross"
    /// matrix), so only the n_imp impurity rows of rot are ever needed. Writing
    /// Kip0 = e_I M + M^dag e_I^dag - e_I D e_I^dag (I = impurity indices,
    /// M = Kip0.rows(I), D = Kip0(I,I)) gives the rank-2*n_imp update
    ///   K = A^dag B + B^dag A - A^dag D A,   A = rot.rows(I),  B = M*rot.
    arma::cx_mat build_K(State const& fb) const
    {
        arma::cx_vec d = ip_phase(n_iter);
        // A = rot.rows(imp_pos); exp_ih is identity on impurity rows, so it drops out.
        arma::cx_mat A = rot_star.cols(imp_pos).t() * fb.rot;   // n_imp x L
        // B = M * rot, evaluated left-to-right to keep every factor n_imp x L.
        arma::cx_mat B = Kip0.rows(imp_pos);                // M (n_imp x L)
        B.each_row() %= d.st();                             // M * diag(exp_ih)
        B = B * rot_star.t();                                   // n_imp x L
        B = B * fb.rot;                                     // n_imp x L
        arma::cx_mat D = Kip0.submat(imp_pos, imp_pos);     // n_imp x n_imp
        return A.t()*B + B.t()*A - A.t()*(D*A);
    }

    /// Rotate the current K into the basis proposed by an orbital update.
    void apply_plan_to_K(State const& fb, OrbitalUpdate<cmpx> const& update)
    {
        update.apply_as_basis(K);
        fb.ensure_symmetry(K);
    }

    /// MPO of the interacting Hamiltonian: the full Umat plus the kinetic
    /// block [a,b) of the current K
    itensor::MPO full_hamiltonian(State const& fb, int a,int b) const
    {
        itensor::AutoMPO h(fb.sites);
        int L = param.length();
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (std::abs(param.Umat(i,j)) > 1e-15)
                    h += param.Umat(i,j), "N", i+1, "N", j+1;

        for(auto i=a; i<b; i++)
            for(auto j=a; j<b; j++)
                if (std::abs(K(i,j))>fb.tol)
                    h += K(i,j),"Cdag",i+1,"C",j+1;

        return itensor::toMPO(h);
    }

    /// One TDVP timestep of a single state; returns its energy.
    double evolve_one(State& fb, itensor::MPO const& mpo, TdvpParam args) const
    {
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.n_iter_diag;
        sweeps.noise() = args.noise;

        if (args.epsilon_M != 0)
        {
            std::vector<double> epsilon_K(args.n_krylov,args.epsilon_K);
            itensor::addBasis(fb.psi,mpo,epsilon_K,
                              {"Cutoff", args.epsilon_M,
                               "Method", "DensityMatrix",
                               "KrylovOrd", args.n_krylov,
                               "DoNormalize", true,
                               "Quiet", true,
                               "Silent", true});
        }

        // Sites beyond the active window are Slater orbitals with no
        // Hamiltonian support in the interaction picture: skip them.
        auto [a,b]=fb.interval_active_full();
        double energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,
                                      {"MaxSite",b,
                                       "Truncate", true,
                                       "DoNormalize", true,
                                       "Quiet", true,
                                       "Silent", true,
                                       "NumCenter", 2,
                                       "ErrGoal", args.err_goal});
        energy += fb.slater_energy(K);
        fb.update_cc();
        return energy;
    }

    /// Effective MPS->real rotation in the Schrödinger picture.
    /// The MPS lives in the interaction picture of H_bath, so fb.rot tracks only the
    /// natural-orbital basis change. To recover real-space Schrödinger-picture
    /// correlators, we dress the bath block with exp(-i Kbath * t).
    arma::cx_mat effective_rot(State const& fb) const
    {
        arma::cx_mat M = rot_star.t() * fb.rot;
        M.each_col() %= ip_phase(n_iter);   // exp_ih * M, with exp_ih=diag(ip_phase)
        return rot_star * M;
    }

    /// The whole Schrödinger-picture real-space <c_i^dag c_j> matrix.
    arma::cx_mat correlator(State const& fb) const
    {
        arma::cx_mat Q = effective_rot(fb);
        return arma::conj(Q) * fb.cc * Q.st();
    }

    /// Schrödinger-picture real-space <c_i^dag c_j>.
    cmpx correlator(State const& fb, int i, int j) const
    {
        arma::cx_mat Q = effective_rot(fb);
        arma::cx_vec ccQj = fb.cc * Q.row(j).st();
        return arma::cdot(Q.row(i).st(), ccQj);
    }

    /// Column j of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed j, all i.
    arma::cx_vec correlator_col(State const& fb, int j) const
    {
        arma::cx_mat Q = effective_rot(fb);
        arma::cx_vec ccQj = fb.cc * Q.row(j).st();
        return arma::conj(Q) * ccQj;
    }

    /// Row i of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed i, all j.
    arma::cx_vec correlator_row(State const& fb, int i) const
    {
        arma::cx_mat Q = effective_rot(fb);
        arma::cx_rowvec v = arma::conj(Q.row(i)) * fb.cc;
        return Q * v.st();
    }
};

} // namespace detail

/// Real-time evolution of one few-body MPS. The orbital layout (spinless,
/// spin-flip symmetric or generic spin) comes from the state and the model,
/// which share it through ImpurityParam::layout:
///   auto solver = Fbr_dyn(model, fb, dt);
/// For several states evolving in one common orbital basis, see Fbr_dyn_shared.
struct Fbr_dyn : detail::DynCommon {
    using Common = detail::DynCommon;
    using State = typename Common::State;

    /// these quantities are updated during the iterations
    State fb;               ///< the current few body MPS
    double energy=-1000;

    explicit Fbr_dyn(ImpurityParam const& param_, State const& fb_, double dt_=0.1)
        : Common(param_,fb_,dt_)
        , fb { fb_ }
    {
        fb.n_sv=this->n_sv;
        this->K=Common::build_K(fb);
    }

    void iterate(TdvpParam args={})
    {
        this->K=Common::build_K(fb);   // interaction-picture Hamiltonian, O(L^2)
        this->n_iter++;

        apply_plan(fb.plan_representative(this->K,0));
        apply_plan(fb.plan_representative(this->K,1));
        apply_plan(fb.plan_active_representative(this->K));
        do_tdvp(args);
        apply_plan(fb.plan_natural_orbitals(fb.cc));
    }

    void apply_plan(OrbitalUpdate<cmpx> const& update)
    {
        this->apply_plan_to_K(fb,update);
        fb.apply(update);
    }

    void do_tdvp(TdvpParam args={})
    {
        auto [a,b]=fb.interval_active_full();
        auto mpo=Common::full_hamiltonian(fb,a,b);
        energy=this->evolve_one(fb,mpo,args);
    }

    arma::cx_mat build_K() const { return Common::build_K(fb); }
    arma::cx_mat effective_rot() const { return Common::effective_rot(fb); }

    arma::cx_mat correlator() const { return Common::correlator(fb); }
    cmpx correlator(int i, int j) const { return Common::correlator(fb,i,j); }
    arma::cx_vec correlator_col(int j) const { return Common::correlator_col(fb,j); }
    arma::cx_vec correlator_row(int i) const { return Common::correlator_row(fb,i); }
};

/// Few-body real-time evolution of several states in one common orbital basis.
///
/// The orbital transformations are found once from the collection of states
/// (natural orbitals come from their averaged correlation matrix) and applied
/// identically to every MPS. Each state is nevertheless evolved by its own
/// TDVP call, since the TDVP projection and truncation are state-dependent.
struct Fbr_dyn_shared : detail::DynCommon {
    using Common = detail::DynCommon;
    using State = typename Common::State;

    /// these quantities are updated during the iterations
    std::vector<State> states;    ///< the current few body MPS states
    std::vector<double> energies; ///< energy of every state

    explicit Fbr_dyn_shared(ImpurityParam const& param_, std::vector<State> states_, double dt_=0.1)
        : Common(param_,first_of(states_),dt_)
        , states(std::move(states_))
    {
        check_common_orbitals();
        energies.assign(states.size(),-1000.0);
        for (auto& state : states) state.n_sv=this->n_sv;
        this->K=Common::build_K(states.front());
    }

    void iterate(TdvpParam args={})
    {
        check_common_orbitals();
        this->K=Common::build_K(states.front());
        this->n_iter++;

        apply_plan(states.front().plan_representative(this->K,0));
        apply_plan(states.front().plan_representative(this->K,1));
        apply_plan(states.front().plan_active_representative(this->K));
        do_tdvp(args);
        apply_plan(widen_to_all_states(states.front().plan_natural_orbitals(combined_cc())));
    }

    void apply_plan(OrbitalUpdate<cmpx> const& update)
    {
        this->apply_plan_to_K(states.front(),update);
        for (auto& state : states)
            state.apply(update);
    }

    void do_tdvp(TdvpParam args={})
    {
        auto [a,b]=states.front().interval_active_full();
        auto mpo=Common::full_hamiltonian(states.front(),a,b);
        for (std::size_t n=0; n<states.size(); ++n)
            energies[n]=this->evolve_one(states[n],mpo,args);
    }

    arma::cx_mat effective_rot(std::size_t n=0) const { return Common::effective_rot(states.at(n)); }

    arma::cx_mat correlator(std::size_t n=0) const { return Common::correlator(states.at(n)); }
    cmpx correlator(int i, int j, std::size_t n=0) const { return Common::correlator(states.at(n),i,j); }
    arma::cx_vec correlator_col(int j, std::size_t n=0) const { return Common::correlator_col(states.at(n),j); }
    arma::cx_vec correlator_row(int i, std::size_t n=0) const { return Common::correlator_row(states.at(n),i); }

private:
    static State const& first_of(std::vector<State> const& states)
    {
        if (states.empty())
            throw std::invalid_argument("Fbr_dyn_shared: at least one state is required");
        return states.front();
    }

    /// All states must share the orbital layout, the rotation frame, the
    /// ITensor site indices, and the Slater part of the correlation matrix.
    void check_common_orbitals() const
    {
        auto const& first=states.front();
        int L=this->param.length();
        if (first.sites.length()!=L)
            throw std::invalid_argument("Fbr_dyn_shared: state length does not match the model");

        auto [a,b]=first.interval_active_full();
        for (std::size_t n=1; n<states.size(); ++n) {
            auto const& state=states[n];
            if (state.sites.length()!=L || state.interval_active_full()!=std::pair{a,b}
                || state.imp_size!=first.imp_size)
                throw std::invalid_argument("Fbr_dyn_shared: states do not share the same orbital layout");
            if (arma::norm(state.rot-first.rot,"fro")>10*first.tol)
                throw std::invalid_argument("Fbr_dyn_shared: states do not share the same orbital rotation");
            for (int i=1; i<=L; ++i)
                if (state.sites(i)!=first.sites(i))
                    throw std::invalid_argument("Fbr_dyn_shared: states must share the same ITensor site indices");
        }

        double tolerance=slater_tol();
        for (int i=0; i<L; ++i) {
            if (i>=a && i<b) continue;
            double occupation=std::real(first.cc(i,i))>0.5 ? 1.0 : 0.0;
            for (auto const& state : states)
                if (std::abs(state.cc(i,i)-occupation)>tolerance)
                    throw std::invalid_argument("Fbr_dyn_shared: states do not share the same Slater state");
            for (std::size_t n=1; n<states.size(); ++n)
                for (int j=0; j<L; ++j) {
                    if (j>=a && j<b) continue;
                    if (std::abs(states[n].cc(i,j)-first.cc(i,j))>tolerance)
                        throw std::invalid_argument("Fbr_dyn_shared: states do not share the same Slater correlator");
                }
        }
    }

    /// How much the Slater part of two states may differ before they count as
    /// incompatible. Used both to check the states and to keep the window wide
    /// enough that they stay compatible.
    double slater_tol() const { return std::max(100*states.front().tol,1e-10); }

    /// The active window has to hold every orbital where the states differ: an
    /// orbital may join the Slater part only if it is empty (or full) in all of
    /// them. plan_natural_orbitals decides that from the averaged correlation
    /// matrix, where a difference between states is divided by their number and
    /// can fall below the tolerance, so widen its window to what each state
    /// needs. Rotating the orbitals is unaffected: the average is the right
    /// choice there, and only the resulting interval is widened.
    OrbitalUpdate<cmpx> widen_to_all_states(OrbitalUpdate<cmpx> update) const
    {
        if (states.size()<2) return update;
        int L=this->param.length();
        auto [lo,hi]=update.active;

        std::vector<arma::cx_mat> cc;   // every correlator in the proposed basis
        cc.reserve(states.size());
        for (auto const& state : states) {
            cc.push_back(state.cc);
            for (auto const& gate : update.gates)
                gate.apply_as_correlator(cc.back());
        }

        for (std::size_t n=0; n<states.size(); ++n) {
            // an orbital that is neither empty nor full is active, as usual
            for (int i=0; i<L; i++) {
                double ni=std::real(cc[n](i,i));
                if (ni>states[n].tol && ni<1-states[n].tol) {
                    lo=std::min(lo,i);
                    hi=std::max(hi,i+1);
                }
            }
            // and so is an orbital where this state differs from the first one.
            // Occupations alone are not enough: a coherence goes like the square
            // root of an occupation, so orbitals far too empty to look active
            // still carry a difference well above the tolerance below.
            if (n==0) continue;
            arma::mat diff=arma::abs(cc[n]-cc[0]);
            for (int i=0; i<L; i++)
                if (diff.row(i).max()>slater_tol()) {
                    lo=std::min(lo,i);
                    hi=std::max(hi,i+1);
                }
        }
        if (states.front().layout==spin_symmetric) {   // keep the window centered
            int d=std::max(hi-L/2,L/2-lo);
            lo=L/2-d;
            hi=L/2+d;
        }
        update.active={lo,hi};
        return update;
    }

    /// Natural orbitals are found from the states' average correlation matrix.
    arma::cx_mat combined_cc() const
    {
        arma::cx_mat cc(states.front().cc.n_rows,states.front().cc.n_cols,arma::fill::zeros);
        for (auto const& state : states)
            cc+=state.cc;
        cc/=static_cast<double>(states.size());
        return cc;
    }
};

} // namespace fbr

#endif // FBR_DYN_H
