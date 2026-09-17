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
#include <string>
#include <utility>
#include <vector>

namespace fbr {

namespace detail {

/// Largest impurity-to-Slater coupling rank across spin sectors. Dynamics fixes
/// this rank at construction; each representative plan caps it by the available
/// singular vectors. Rank selection uses the state's relative activity tolerance.
inline int coupling_rank(Fb_mps<cmpx> const& fb, arma::mat const& Kmat)
{
    int rank=0;
    for (Spin s : {up,dw}) {
        auto [a_imp,b_imp]=fb.range(Part::impurity,s);
        auto [a_sla,b_sla]=fb.range(Part::slater,s);
        if (a_imp>=b_imp || a_sla>=b_sla) continue;
        // Keep singular values above the relative orbital-activity tolerance.
        arma::vec singular_values=arma::svd(Kmat.submat(a_imp,a_sla,b_imp-1,b_sla-1));
        if (singular_values.empty() || singular_values[0]==0) continue;
        int sector_rank=(int)arma::find(singular_values>fb.act_tol()*singular_values[0]).eval().size();
        rank=std::max(rank,sector_rank);
    }
    return rank;
}

/// Check that the initial cc blocks agree under spin reflection. Orbital plans
/// use the down sector and mirror it onto up, so a polarized state needs spin_block.
/// The 1e-6 threshold allows DMRG error in a nominally symmetric initial state.
/// This is a one-particle symmetry check, not a test of the full many-body state.
inline void require_spin_sym_state(Fb_mps<cmpx> const& fb)
{
    if (fb.geometry!=spin_sym) return;
    constexpr double threshold=1e-6;
    arma::cx_mat mirrored=fb.cc;
    fb.ensure_symmetry(mirrored);
    double mismatch=arma::abs(mirrored-fb.cc).max();
    if (mismatch>threshold)
        throw std::invalid_argument("Fbr_dyn: spin_sym needs a spin-flip symmetric state, "
                                    "but the up and dw blocks of cc differ by "+std::to_string(mismatch)
                                    +" (threshold "+std::to_string(threshold)+"); use spin_block");
}

} // namespace detail

/// Real-time evolution of one few-body MPS. The chain geometry (standard,
/// spin-flip symmetric or generic spin) comes from the state and the model,
/// which share it through ImpurityParam::geometry:
///   auto solver = Fbr_dyn(model, fb, dt);
/// For several states evolving in one common orbital basis, see Fbr_dyn_shared.
struct Fbr_dyn {
    using State = Fb_mps<cmpx>;

    ImpurityParam param;
    double dt;
    arma::cx_mat Kbath;      ///< diagonal bath Hamiltonian (star geometry)
    arma::cx_mat Kip0;       ///< second-order interaction-picture Hamiltonian at t=0
    arma::uvec imp_pos;      ///< impurity positions (star geometry)
    arma::uvec bath_pos;     ///< bath positions (star geometry)
    arma::cx_mat rot_star;  ///< the star frame, the basis Kip0 and Kbath are written in
    int coupling_rank=0;    ///< fixed rank of the impurity–bath coupling

    /// these quantities are updated during the iterations
    arma::cx_mat K;          ///< kinetic matrix in the current orbital basis (interaction picture)
    int n_iter=0;            ///< completed timesteps between calls to iterate()

    State fb;               ///< the current few-body MPS
    double energy=-1000;

    explicit Fbr_dyn(ImpurityParam const& param_, State const& fb_, double dt_=0.1)
        : param(param_)
        , dt(dt_)
        , fb(fb_)
    {
        param.prepare();   // a model built directly in star geometry never saw to_star()
        int L=param.length();
        if (fb.sites.length()!=L)
            throw std::invalid_argument("Fbr_dyn: state length does not match the model");
        imp_pos = arma::conv_to<arma::uvec>::from(param.imp_pos);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L,param.imp_pos));

        // The state geometry must put its impurity where the model does: this is
        // what tells a centered geometry apart from a standard one.
        auto [a_imp,b_imp]=fb.range(Part::impurity);
        if (imp_pos.empty() || b_imp-a_imp!=param.n_imp()
            || (int)imp_pos.min()!=a_imp || (int)imp_pos.max()!=b_imp-1)
            throw std::invalid_argument("Fbr_dyn: the state geometry does not match the impurity positions of the model");
        detail::require_spin_sym_state(fb);

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

        // Fix coupling_rank = rank of the impurity–bath coupling block at construction.
        // The same value is used for every representative plan thereafter.
        coupling_rank = detail::coupling_rank(fb,param.Kmat);
        fb.coupling_rank=coupling_rank;
        K=build_K();
    }

    void iterate(TdvpParam args={})
    {
        K=build_K();
        ++n_iter;

        apply_plan(fb.plan_representative(K,0));
        apply_plan(fb.plan_representative(K,1));
        apply_plan(fb.plan_active_representative(K));
        do_tdvp(args);
        apply_plan(fb.plan_natural_orbitals(fb.cc));
    }

    void apply_plan(OrbitalUpdate<cmpx> const& update)
    {
        update.apply_as_basis(K);
        fb.ensure_symmetry(K);
        fb.apply(update);
    }

    /// Build the active-window Hamiltonian and evolve the MPS for one timestep.
    void do_tdvp(TdvpParam args={})
    {
        auto [a,b]=fb.range(Part::active);
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

        auto mpo=itensor::toMPO(h);

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
        energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,
                                      {"MaxSite",b,
                                       "Truncate", true,
                                       "DoNormalize", true,
                                       "Quiet", true,
                                       "Silent", true,
                                       "NumCenter", 2,
                                       "ErrGoal", args.err_goal});
        energy += fb.slater_energy(K);
        fb.update_cc();
    }

    /// Diagonal bath phases exp(-i H_bath * n*dt).
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
    arma::cx_mat build_K() const
    {
        arma::cx_vec d = ip_phase(n_iter);
        // A = rot.rows(imp_pos); bath phases are one on impurity rows.
        arma::cx_mat A = rot_star.cols(imp_pos).t() * fb.rot;   // n_imp x L
        // B = M * rot, evaluated left-to-right to keep every factor n_imp x L.
        arma::cx_mat B = Kip0.rows(imp_pos);                // M (n_imp x L)
        B.each_row() %= d.st();                             // M * diag(d)
        B = B * rot_star.t();                                   // n_imp x L
        B = B * fb.rot;                                     // n_imp x L
        arma::cx_mat D = Kip0.submat(imp_pos, imp_pos);     // n_imp x n_imp
        return A.t()*B + B.t()*A - A.t()*(D*A);
    }

    /// Effective MPS->real rotation in the Schrödinger picture.
    /// The MPS lives in the interaction picture of H_bath, so fb.rot tracks only the
    /// natural-orbital basis change. To recover real-space Schrödinger-picture
    /// correlators, we dress the bath block with exp(-i Kbath * t):
    ///   Q = rot_star * diag(ip_phase) * rot_star^dag * fb.rot.
    /// O(L^3) (two dense L x L products): the single elements, rows and columns
    /// below never form it.
    arma::cx_mat effective_rot() const
    {
        arma::cx_mat M = rot_star.t() * fb.rot;
        M.each_col() %= ip_phase(n_iter);   // diag(ip_phase) * M
        return rot_star * M;
    }

    /// Row k of effective_rot, as a column: Q.row(k)^T = fb.rot^T conj(rot_star) (d % rot_star.row(k)^T),
    /// d=ip_phase. Written with conjugated vectors so that every product is a
    /// plain or ^dag matrix-vector product: O(L^2), no L x L temporary.
    arma::cx_vec effective_rot_row(int k) const
    {
        arma::cx_vec x = arma::conj(rot_star.row(k).st() % ip_phase(n_iter));
        return arma::conj(fb.rot.t() * (rot_star * x));
    }

    /// Q * w, with Q = effective_rot, from right to left: O(L^2).
    arma::cx_vec effective_rot_times(arma::cx_vec const& w) const
    {
        arma::cx_vec y = rot_star.t() * (fb.rot * w);
        y %= ip_phase(n_iter);
        return rot_star * y;
    }

    /// The whole Schrödinger-picture real-space <c_i^dag c_j> matrix. O(L^3).
    arma::cx_mat correlator() const
    {
        arma::cx_mat Q = effective_rot();
        return arma::conj(Q) * fb.cc * Q.st();
    }

    /// Schrödinger-picture real-space <c_i^dag c_j> = sum_ab conj(Q(i,a)) cc(a,b) Q(j,b).
    /// Only rows i and j of Q are needed: O(L^2).
    cmpx correlator(int i, int j) const
    {
        arma::cx_vec ccQj = fb.cc * effective_rot_row(j);
        return arma::cdot(effective_rot_row(i), ccQj);
    }

    /// Column j of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed j, all i.
    /// conj(Q) * v = conj(Q * conj(v)): O(L^2).
    arma::cx_vec correlator_col(int j) const
    {
        arma::cx_vec ccQj = fb.cc * effective_rot_row(j);
        return arma::conj(effective_rot_times(arma::conj(ccQj)));
    }

    /// Row i of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed i, all j.
    /// Q * (conj(Q.row(i)) * cc)^T: O(L^2).
    arma::cx_vec correlator_row(int i) const
    {
        arma::cx_rowvec v = effective_rot_row(i).t() * fb.cc;
        return effective_rot_times(v.st());
    }
};

/// Few-body real-time evolution of several states in one common orbital basis.
///
/// The first state's natural orbitals define the basis for every state. Put the
/// state whose evolution is hardest to represent first (usually c^dag|psi0> for
/// a Green function). After rotating, widen_to_all_states keeps the window large
/// enough for every state. Each MPS gets its own TDVP sweep because projection
/// and truncation depend on the state.
struct Fbr_dyn_shared {
    using State = Fb_mps<cmpx>;

    ImpurityParam param;
    double dt;
    arma::cx_mat Kbath;      ///< diagonal bath Hamiltonian (star geometry)
    arma::cx_mat Kip0;       ///< second-order interaction-picture Hamiltonian at t=0
    arma::uvec imp_pos;      ///< impurity positions (star geometry)
    arma::uvec bath_pos;     ///< bath positions (star geometry)
    arma::cx_mat rot_star;  ///< the star frame, the basis Kip0 and Kbath are written in
    int coupling_rank=0;    ///< fixed rank of the impurity–bath coupling

    /// these quantities are updated during the iterations
    arma::cx_mat K;          ///< kinetic matrix in the current orbital basis (interaction picture)
    int n_iter=0;            ///< completed timesteps between calls to iterate()

    std::vector<State> states;    ///< the current few-body MPS states
    std::vector<double> energies; ///< energy of every state

    explicit Fbr_dyn_shared(ImpurityParam const& param_, std::vector<State> states_, double dt_=0.1)
        : param(param_)
        , dt(dt_)
        , states(std::move(states_))
    {
        if (states.empty())
            throw std::invalid_argument("Fbr_dyn_shared: at least one state is required");
        auto const& first=states.front();
        param.prepare();   // a model built directly in star geometry never saw to_star()
        int L=param.length();
        if (first.sites.length()!=L)
            throw std::invalid_argument("Fbr_dyn: state length does not match the model");
        imp_pos = arma::conv_to<arma::uvec>::from(param.imp_pos);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L,param.imp_pos));

        // The state geometry must put its impurity where the model does: this is
        // what tells a centered geometry apart from a standard one.
        auto [a_imp,b_imp]=first.range(Part::impurity);
        if (imp_pos.empty() || b_imp-a_imp!=param.n_imp()
            || (int)imp_pos.min()!=a_imp || (int)imp_pos.max()!=b_imp-1)
            throw std::invalid_argument("Fbr_dyn: the state geometry does not match the impurity positions of the model");
        detail::require_spin_sym_state(first);

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

        // Fix coupling_rank = rank of the impurity–bath coupling block at construction.
        // The same value is used for every representative plan thereafter.
        coupling_rank = detail::coupling_rank(first,param.Kmat);
        check_common_orbitals();
        for (auto const& state : states)
            detail::require_spin_sym_state(state);
        energies.assign(states.size(),-1000.0);
        for (auto& state : states) state.coupling_rank=coupling_rank;
        K=build_K();
    }

    void iterate(TdvpParam args={})
    {
        check_common_orbitals();
        K=build_K();
        ++n_iter;

        auto& first=states.front();
        apply_plan(first.plan_representative(K,0));
        apply_plan(first.plan_representative(K,1));
        apply_plan(first.plan_active_representative(K));
        do_tdvp(args);
        apply_plan(widen_to_all_states(first.plan_natural_orbitals(first.cc)));
    }

    void apply_plan(OrbitalUpdate<cmpx> const& update)
    {
        update.apply_as_basis(K);
        states.front().ensure_symmetry(K);
        for (auto& state : states) state.apply(update);
    }

    void do_tdvp(TdvpParam args={})
    {
        auto const& first=states.front();
        auto [a,b]=first.range(Part::active);
        itensor::AutoMPO h(first.sites);
        int L = param.length();
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (std::abs(param.Umat(i,j)) > 1e-15)
                    h += param.Umat(i,j), "N", i+1, "N", j+1;

        for(auto i=a; i<b; i++)
            for(auto j=a; j<b; j++)
                if (std::abs(K(i,j))>first.tol)
                    h += K(i,j),"Cdag",i+1,"C",j+1;

        auto mpo=itensor::toMPO(h);

        for (std::size_t n=0; n<states.size(); ++n) {
            auto& fb=states[n];
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
            energies[n] = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,
                                          {"MaxSite",b,
                                           "Truncate", true,
                                           "DoNormalize", true,
                                           "Quiet", true,
                                           "Silent", true,
                                           "NumCenter", 2,
                                           "ErrGoal", args.err_goal});
            energies[n] += fb.slater_energy(K);
            fb.update_cc();
        }
    }

    arma::cx_vec ip_phase(int n) const
    {
        arma::cx_vec d(param.length(),arma::fill::ones);
        if (n > 0)
            d(bath_pos) = arma::exp(-imag_1 * Kbath.diag() * (static_cast<double>(n)*dt));
        return d;
    }

    arma::cx_mat build_K() const
    {
        auto const& fb=states.front();
        arma::cx_vec d = ip_phase(n_iter);
        // A = rot.rows(imp_pos); bath phases are one on impurity rows.
        arma::cx_mat A = rot_star.cols(imp_pos).t() * fb.rot;   // n_imp x L
        // B = M * rot, evaluated left-to-right to keep every factor n_imp x L.
        arma::cx_mat B = Kip0.rows(imp_pos);                // M (n_imp x L)
        B.each_row() %= d.st();                             // M * diag(d)
        B = B * rot_star.t();                                   // n_imp x L
        B = B * fb.rot;                                     // n_imp x L
        arma::cx_mat D = Kip0.submat(imp_pos, imp_pos);     // n_imp x n_imp
        return A.t()*B + B.t()*A - A.t()*(D*A);
    }

    /// Effective MPS->real rotation in the Schrödinger picture.
    /// The MPS lives in the interaction picture of H_bath, so states.at(n).rot tracks only the
    /// natural-orbital basis change. To recover real-space Schrödinger-picture
    /// correlators, we dress the bath block with exp(-i Kbath * t):
    ///   Q = rot_star * diag(ip_phase) * rot_star^dag * states.at(n).rot.
    /// O(L^3) (two dense L x L products): the single elements, rows and columns
    /// below never form it.
    arma::cx_mat effective_rot(std::size_t n=0) const
    {
        arma::cx_mat M = rot_star.t() * states.at(n).rot;
        M.each_col() %= ip_phase(n_iter);   // diag(ip_phase) * M
        return rot_star * M;
    }

    /// Row k of effective_rot, as a column: Q.row(k)^T = states.at(n).rot^T conj(rot_star) (d % rot_star.row(k)^T),
    /// d=ip_phase. Written with conjugated vectors so that every product is a
    /// plain or ^dag matrix-vector product: O(L^2), no L x L temporary.
    arma::cx_vec effective_rot_row(int k, std::size_t n=0) const
    {
        arma::cx_vec x = arma::conj(rot_star.row(k).st() % ip_phase(n_iter));
        return arma::conj(states.at(n).rot.t() * (rot_star * x));
    }

    /// Q * w, with Q = effective_rot, from right to left: O(L^2).
    arma::cx_vec effective_rot_times(arma::cx_vec const& w, std::size_t n=0) const
    {
        arma::cx_vec y = rot_star.t() * (states.at(n).rot * w);
        y %= ip_phase(n_iter);
        return rot_star * y;
    }

    /// The whole Schrödinger-picture real-space <c_i^dag c_j> matrix. O(L^3).
    arma::cx_mat correlator(std::size_t n=0) const
    {
        arma::cx_mat Q = effective_rot(n);
        return arma::conj(Q) * states.at(n).cc * Q.st();
    }

    /// Schrödinger-picture real-space <c_i^dag c_j> = sum_ab conj(Q(i,a)) cc(a,b) Q(j,b).
    /// Only rows i and j of Q are needed: O(L^2).
    cmpx correlator(int i, int j, std::size_t n=0) const
    {
        arma::cx_vec ccQj = states.at(n).cc * effective_rot_row(j,n);
        return arma::cdot(effective_rot_row(i,n), ccQj);
    }

    /// Column j of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed j, all i.
    /// conj(Q) * v = conj(Q * conj(v)): O(L^2).
    arma::cx_vec correlator_col(int j, std::size_t n=0) const
    {
        arma::cx_vec ccQj = states.at(n).cc * effective_rot_row(j,n);
        return arma::conj(effective_rot_times(arma::conj(ccQj),n));
    }

    /// Row i of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed i, all j.
    /// Q * (conj(Q.row(i)) * cc)^T: O(L^2).
    arma::cx_vec correlator_row(int i, std::size_t n=0) const
    {
        arma::cx_rowvec v = effective_rot_row(i,n).t() * states.at(n).cc;
        return effective_rot_times(v.st(),n);
    }
private:
    /// All states must share the orbital geometry, the rotation frame, the
    /// ITensor site indices, and the Slater part of the correlation matrix.
    void check_common_orbitals() const
    {
        if (states.empty())
            throw std::invalid_argument("Fbr_dyn_shared: at least one state is required");
        auto const& first=states.front();
        int L=param.length();
        if (first.sites.length()!=L)
            throw std::invalid_argument("Fbr_dyn_shared: state length does not match the model");

        Range window=first.range(Part::active);
        for (std::size_t n=1; n<states.size(); ++n) {
            auto const& state=states[n];
            if (state.sites.length()!=L || state.range(Part::active)!=window
                || state.imp_size!=first.imp_size)
                throw std::invalid_argument("Fbr_dyn_shared: states do not share the same orbital geometry");
            if (arma::norm(state.rot-first.rot,"fro")>10*first.tol)
                throw std::invalid_argument("Fbr_dyn_shared: states do not share the same orbital rotation");
            for (int i=1; i<=L; ++i)
                if (state.sites(i)!=first.sites(i))
                    throw std::invalid_argument("Fbr_dyn_shared: states must share the same ITensor site indices");
        }

        double tolerance=slater_tol();
        for (int i=0; i<L; ++i) {
            if (window.contains(i)) continue;
            double occupation=std::real(first.cc(i,i))>0.5 ? 1.0 : 0.0;
            for (auto const& state : states)
                if (std::abs(state.cc(i,i)-occupation)>tolerance)
                    throw std::invalid_argument("Fbr_dyn_shared: states do not share the same Slater state");
            for (std::size_t n=1; n<states.size(); ++n)
                for (int j=0; j<L; ++j) {
                    if (window.contains(j)) continue;
                    if (std::abs(states[n].cc(i,j)-first.cc(i,j))>tolerance)
                        throw std::invalid_argument("Fbr_dyn_shared: states do not share the same Slater correlator");
                }
        }
    }

    /// How much the Slater part of two states may differ before they count as
    /// incompatible. Used both to check the states and to keep the window wide
    /// enough that they stay compatible.
    double slater_tol() const { return std::max(100*states.front().tol,1e-10); }

    /// The first state's natural orbitals define the shared basis. Widen its
    /// proposed window to retain every orbital needed by any of the other states.
    OrbitalUpdate<cmpx> widen_to_all_states(OrbitalUpdate<cmpx> update) const
    {
        if (states.size()<2) return update;
        int L=param.length();
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
        if (states.front().geometry==spin_sym) {   // keep the window centered
            int d=std::max(hi-L/2,L/2-lo);
            lo=L/2-d;
            hi=L/2+d;
        }
        update.active={lo,hi};
        return update;
    }

};

} // namespace fbr

#endif // FBR_DYN_H
