#ifndef FBR_DYN_H
#define FBR_DYN_H

#include "graph.h"
#include "itensor_utils.h"
#include "impurity_param.h"
#include "impurity_param_spin.h"
#include "fb_mps.h"

#include "tdvp.h"
#include "basisextension.h"

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace fbr {

namespace detail {

/// Rank of a coupling block: number of singular values above tol (relative).
inline int svRank(arma::mat const& block, double tol)
{
    if (block.n_rows==0 || block.n_cols==0) return 0;
    arma::vec s=arma::svd(block);
    if (s.empty() || s[0]==0) return 0;
    return (int)arma::find(s>tol*s[0]).eval().size();
}

/// Rank of the impurity-bath coupling block of Kmat, per spin sector.
inline int couplingRank(Fb_mps<cmpx> const& fb, arma::mat const& Kmat)
{
    int rank=0;
    for (Spin s : {up,dw}) {
        auto [a_imp,b_imp]=fb.interval_impurity(s);
        auto [a_sla,b_sla]=fb.interval_slater(s);
        if (a_imp>=b_imp || a_sla>=b_sla) continue;
        rank=std::max(rank,svRank(Kmat.submat(a_imp,a_sla,b_imp-1,b_sla-1),fb.tol));
    }
    return rank;
}

/// Machinery shared by the single-state (Fbr_dyn) and multi-state (Fbr_ns_dyn)
/// dynamics solvers: the interaction picture of the diagonal bath Hamiltonian
/// and the frame algebra built on top of it. The MPS never sees the bath
/// phases, fb.rot tracks only the natural-orbital basis change, and
/// Schrödinger-picture correlators are recovered by dressing with
/// exp(-i Kbath t) (effective_rot).
template<class Model>
struct DynCommon {
    using State = Fb_mps<cmpx>;
    using Param = std::decay_t<decltype(std::declval<Model const&>().param)>;

    Param param;
    double dt;
    arma::cx_mat Kbath;      ///< diagonal bath Hamiltonian (star geometry)
    arma::cx_mat Kip0;       ///< second-order interaction-picture Hamiltonian at t=0
    arma::uvec imp_pos;      ///< impurity positions (star layout)
    arma::uvec bath_pos;     ///< bath positions (star layout)
    arma::cx_mat rotS;       ///< the star frame, the basis Kip0 and Kbath are written in
    int nSv=0;               ///< fixed rank of the impurity–bath coupling

    /// these quantities are updated during the iterations
    arma::cx_mat K;          ///< the current Hamiltonian
    int nIter=0;

    DynCommon(Model const& imp, State const& first, double dt_)
        : param(imp.param)
        , dt(dt_)
    {
        int L=param.length();
        if (first.sites.length()!=L)
            throw std::invalid_argument("Fbr_dyn: state length does not match the model");
        imp_pos = arma::conv_to<arma::uvec>::from(param.impPos);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L,param.impPos));

        // The state layout must put its impurity where the model does: this is
        // what tells a centered layout apart from a leading one.
        auto [a_imp,b_imp]=first.interval_impurity_full();
        if (imp_pos.empty() || b_imp-a_imp!=param.nImp()
            || (int)imp_pos.min()!=a_imp || (int)imp_pos.max()!=b_imp-1)
            throw std::invalid_argument("Fbr_dyn: the state layout does not match the impurity positions of the model");

        arma::mat Kstar=param.Kmat;
        if (!bath_pos.empty()) {
            arma::mat Kb=Kstar.submat(bath_pos,bath_pos);
            if (arma::abs(Kb-arma::diagmat(Kb.diag())).max()>1e-12)
                throw std::invalid_argument("Fbr_dyn: the bath block of Kmat must be diagonal (star geometry, see toStar)");
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
        // from the model makes rotS^dag * fb.rot express the current orbitals in
        // the star basis whatever frame the state starts in.
        rotS = param.rot * cmpx(1,0);
        Kbath = Kstar.submat(bath_pos,bath_pos) * cmpx(1,0);

        // Fix nSv = rank of the impurity–bath coupling block at construction.
        // The same value is used for every representative plan thereafter.
        nSv = couplingRank(first,param.Kmat);
    }

    /// Diagonal of the interaction-picture phase exp(-i H_bath * n*dt).
    /// In star geometry H_bath is diagonal, so this is computed in O(L_bath),
    /// avoiding the O(L^3) dense matrix exponential.
    arma::cx_vec ipPhase(int n) const
    {
        arma::cx_vec d(param.length(),arma::fill::ones);
        if (n > 0)
            d(bath_pos) = arma::exp(-imag_1 * Kbath.diag() * (static_cast<double>(n)*dt));
        return d;
    }

    /// Interaction-picture Hamiltonian K = rot^dag * Kip0 * rot, with
    /// rot = diag(ipPhase) * rotS^dag * fb.rot.
    /// O(L^2): Kip0 is Hermitian and its bath-bath block is exactly zero (a "cross"
    /// matrix), so only the nImp impurity rows of rot are ever needed. Writing
    /// Kip0 = e_I M + M^dag e_I^dag - e_I D e_I^dag (I = impurity indices,
    /// M = Kip0.rows(I), D = Kip0(I,I)) gives the rank-2*nImp update
    ///   K = A^dag B + B^dag A - A^dag D A,   A = rot.rows(I),  B = M*rot.
    arma::cx_mat buildK(State const& fb) const
    {
        arma::cx_vec d = ipPhase(nIter);
        // A = rot.rows(imp_pos); exp_ih is identity on impurity rows, so it drops out.
        arma::cx_mat A = rotS.cols(imp_pos).t() * fb.rot;   // nImp x L
        // B = M * rot, evaluated left-to-right to keep every factor nImp x L.
        arma::cx_mat B = Kip0.rows(imp_pos);                // M (nImp x L)
        B.each_row() %= d.st();                             // M * diag(exp_ih)
        B = B * rotS.t();                                   // nImp x L
        B = B * fb.rot;                                     // nImp x L
        arma::cx_mat D = Kip0.submat(imp_pos, imp_pos);     // nImp x nImp
        return A.t()*B + B.t()*A - A.t()*(D*A);
    }

    /// Reference O(L^3) full conjugation. Numerically identical to buildK();
    /// kept only for validation tests.
    arma::cx_mat buildK_reference(State const& fb) const
    {
        int L = param.length();
        arma::cx_mat exp_ih(L, L, arma::fill::eye);
        if (nIter > 0)
            exp_ih.submat(bath_pos,bath_pos) = expIH<cmpx>(Kbath * (static_cast<double>(nIter)*dt));
        arma::cx_mat rot = exp_ih * rotS.t() * fb.rot;
        return rot.t() * Kip0 * rot;
    }

    /// Rotate the current K into the basis proposed by an orbital update.
    void applyPlanToK(State const& fb, OrbitalUpdate<cmpx> const& update)
    {
        update.applyAsBasis(K);
        fb.ensure_symmetry(K);
    }

    /// MPO of the interacting Hamiltonian: the full Umat plus the kinetic
    /// block [a,b) of the current K
    itensor::MPO fullHamiltonian(State const& fb, int a,int b) const
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
    double evolveOne(State& fb, itensor::MPO const& mpo, TdvpParam args) const
    {
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;

        if (args.epsilonM != 0)
        {
            std::vector<double> epsilonK(args.nKrylov,args.epsilonK);
            itensor::addBasis(fb.psi,mpo,epsilonK,
                              {"Cutoff", args.epsilonM,
                               "Method", "DensityMatrix",
                               "KrylovOrd", args.nKrylov,
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
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
        return energy;
    }

    /// Effective MPS->real rotation in the Schrödinger picture.
    /// The MPS lives in the interaction picture of H_bath, so fb.rot tracks only the
    /// natural-orbital basis change. To recover real-space Schrödinger-picture
    /// correlators, we dress the bath block with exp(-i Kbath * t).
    arma::cx_mat effective_rot(State const& fb) const
    {
        arma::cx_mat M = rotS.t() * fb.rot;
        M.each_col() %= ipPhase(nIter);   // exp_ih * M, with exp_ih=diag(ipPhase)
        return rotS * M;
    }

    /// Schrödinger-picture real-space <c_i^dag c_j> matrix.
    arma::cx_mat correlator_all(State const& fb) const
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

    /// Row of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed j, all i.
    arma::cx_vec correlator_all_i(State const& fb, int j) const
    {
        arma::cx_mat Q = effective_rot(fb);
        arma::cx_vec ccQj = fb.cc * Q.row(j).st();
        return arma::conj(Q) * ccQj;
    }

    /// Column of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed i, all j.
    arma::cx_vec correlator_all_j(State const& fb, int i) const
    {
        arma::cx_mat Q = effective_rot(fb);
        arma::cx_rowvec v = arma::conj(Q.row(i)) * fb.cc;
        return Q * v.st();
    }
};

} // namespace detail

/// Real-time evolution of one few-body MPS. The state type selects the orbital
/// layout (Fb_mps: spinless, Fb_mps_spin: spin-flip symmetric,
/// Fb_mps_spin_block: generic spin); class template argument deduction makes
/// the usual spelling simply
///   auto solver = Fbr_dyn(model, fb, dt);
/// For several states evolving in one common orbital basis, see Fbr_ns_dyn.
template<class Model>
struct Fbr_dyn : detail::DynCommon<Model> {
    using Common = detail::DynCommon<Model>;
    using State = typename Common::State;

    /// these quantities are updated during the iterations
    State fb;               ///< the current few body MPS
    double energy=-1000;

    explicit Fbr_dyn(Model const& imp, State const& fb_, double dt_=0.1)
        : Common(imp,fb_,dt_)
        , fb { fb_ }
    {
        fb.nSv=this->nSv;
        this->K=Common::buildK(fb);
    }

    void iterate(TdvpParam args={})
    {
        this->K=Common::buildK(fb);   // interaction-picture Hamiltonian, O(L^2)
        this->nIter++;

        applyPlan(fb.planRepresentative(this->K,0));
        applyPlan(fb.planRepresentative(this->K,1));
        applyPlan(fb.planActiveRepresentative(this->K));
        doTdvp(args);
        applyPlan(fb.planNaturalOrbitals(fb.cc));
    }

    void applyPlan(OrbitalUpdate<cmpx> const& update)
    {
        this->applyPlanToK(fb,update);
        fb.applyUpdate(update);
    }

    void doTdvp(TdvpParam args={})
    {
        auto [a,b]=fb.interval_active_full();
        auto mpo=Common::fullHamiltonian(fb,a,b);
        energy=this->evolveOne(fb,mpo,args);
    }

    arma::cx_mat buildK() const { return Common::buildK(fb); }
    arma::cx_mat buildK_reference() const { return Common::buildK_reference(fb); }
    arma::cx_mat effective_rot() const { return Common::effective_rot(fb); }

    arma::cx_mat correlator_all() const { return Common::correlator_all(fb); }
    cmpx correlator(int i, int j) const { return Common::correlator(fb,i,j); }
    arma::cx_vec correlator_all_i(int j) const { return Common::correlator_all_i(fb,j); }
    arma::cx_vec correlator_all_j(int i) const { return Common::correlator_all_j(fb,i); }
};

/// Few-body real-time evolution of several states in one common orbital basis.
///
/// The orbital transformations are found once from the collection of states
/// (natural orbitals come from their averaged correlation matrix) and applied
/// identically to every MPS. Each state is nevertheless evolved by its own
/// TDVP call, since the TDVP projection and truncation are state-dependent.
template<class Model>
struct Fbr_ns_dyn : detail::DynCommon<Model> {
    using Common = detail::DynCommon<Model>;
    using State = typename Common::State;

    /// these quantities are updated during the iterations
    std::vector<State> states;    ///< the current few body MPS states
    std::vector<double> energies; ///< energy of every state

    explicit Fbr_ns_dyn(Model const& imp, std::vector<State> states_, double dt_=0.1)
        : Common(imp,firstOf(states_),dt_)
        , states(std::move(states_))
    {
        checkCommonOrbitals();
        energies.assign(states.size(),-1000.0);
        for (auto& state : states) state.nSv=this->nSv;
        this->K=Common::buildK(states.front());
    }

    void iterate(TdvpParam args={})
    {
        checkCommonOrbitals();
        this->K=Common::buildK(states.front());
        this->nIter++;

        applyPlan(states.front().planRepresentative(this->K,0));
        applyPlan(states.front().planRepresentative(this->K,1));
        applyPlan(states.front().planActiveRepresentative(this->K));
        doTdvp(args);
        applyPlan(widenToAllStates(states.front().planNaturalOrbitals(combinedCc())));
    }

    void applyPlan(OrbitalUpdate<cmpx> const& update)
    {
        this->applyPlanToK(states.front(),update);
        for (auto& state : states)
            state.applyUpdate(update);
    }

    void doTdvp(TdvpParam args={})
    {
        auto [a,b]=states.front().interval_active_full();
        auto mpo=Common::fullHamiltonian(states.front(),a,b);
        for (std::size_t n=0; n<states.size(); ++n)
            energies[n]=this->evolveOne(states[n],mpo,args);
    }

    arma::cx_mat effective_rot(std::size_t n=0) const { return Common::effective_rot(states.at(n)); }

    arma::cx_mat correlator_all(std::size_t n=0) const { return Common::correlator_all(states.at(n)); }
    cmpx correlator(int i, int j, std::size_t n=0) const { return Common::correlator(states.at(n),i,j); }
    arma::cx_vec correlator_all_i(int j, std::size_t n=0) const { return Common::correlator_all_i(states.at(n),j); }
    arma::cx_vec correlator_all_j(int i, std::size_t n=0) const { return Common::correlator_all_j(states.at(n),i); }

private:
    static State const& firstOf(std::vector<State> const& states)
    {
        if (states.empty())
            throw std::invalid_argument("Fbr_ns_dyn: at least one state is required");
        return states.front();
    }

    /// All states must share the orbital layout, the rotation frame, the
    /// ITensor site indices, and the Slater part of the correlation matrix.
    void checkCommonOrbitals() const
    {
        auto const& first=states.front();
        int L=this->param.length();
        if (first.sites.length()!=L)
            throw std::invalid_argument("Fbr_ns_dyn: state length does not match the model");

        auto [a,b]=first.interval_active_full();
        for (std::size_t n=1; n<states.size(); ++n) {
            auto const& state=states[n];
            if (state.sites.length()!=L || state.interval_active_full()!=std::pair{a,b}
                || state.imp_size!=first.imp_size)
                throw std::invalid_argument("Fbr_ns_dyn: states do not share the same orbital layout");
            if (arma::norm(state.rot-first.rot,"fro")>10*first.tol)
                throw std::invalid_argument("Fbr_ns_dyn: states do not share the same orbital rotation");
            for (int i=1; i<=L; ++i)
                if (state.sites(i)!=first.sites(i))
                    throw std::invalid_argument("Fbr_ns_dyn: states must share the same ITensor site indices");
        }

        double slater_tol=slaterTol();
        for (int i=0; i<L; ++i) {
            if (i>=a && i<b) continue;
            double occupation=std::real(first.cc(i,i))>0.5 ? 1.0 : 0.0;
            for (auto const& state : states)
                if (std::abs(state.cc(i,i)-occupation)>slater_tol)
                    throw std::invalid_argument("Fbr_ns_dyn: states do not share the same Slater state");
            for (std::size_t n=1; n<states.size(); ++n)
                for (int j=0; j<L; ++j) {
                    if (j>=a && j<b) continue;
                    if (std::abs(states[n].cc(i,j)-first.cc(i,j))>slater_tol)
                        throw std::invalid_argument("Fbr_ns_dyn: states do not share the same Slater correlator");
                }
        }
    }

    /// How much the Slater part of two states may differ before they count as
    /// incompatible. Used both to check the states and to keep the window wide
    /// enough that they stay compatible.
    double slaterTol() const { return std::max(100*states.front().tol,1e-10); }

    /// The active window has to hold every orbital where the states differ: an
    /// orbital may join the Slater part only if it is empty (or full) in all of
    /// them. planNaturalOrbitals decides that from the averaged correlation
    /// matrix, where a difference between states is divided by their number and
    /// can fall below the tolerance, so widen its window to what each state
    /// needs. Rotating the orbitals is unaffected: the average is the right
    /// choice there, and only the resulting interval is widened.
    OrbitalUpdate<cmpx> widenToAllStates(OrbitalUpdate<cmpx> update) const
    {
        if (states.size()<2) return update;
        int L=this->param.length();
        auto [lo,hi]=update.active;

        std::vector<arma::cx_mat> cc;   // every correlator in the proposed basis
        cc.reserve(states.size());
        for (auto const& state : states) {
            cc.push_back(state.cc);
            for (auto const& gate : update.gates)
                gate.applyAsCorrelator(cc.back());
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
                if (diff.row(i).max()>slaterTol()) {
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
    arma::cx_mat combinedCc() const
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
