#ifndef FBR_DYN_FRAME_H
#define FBR_DYN_FRAME_H

#include "fbr_dyn.h"

namespace fbr {

/// EXPERIMENTAL / NEGATIVE RESULT -- kept for the record, not used by any test.
///
/// Co-moving-frame real-time evolution: a first-order Trotter split
///     exp(-iH dt) = exp(-i H_bath dt) exp(-i H_imp dt) + O(dt^2)
/// where, unlike the interaction-picture Fbr_dyn, the bath propagator is applied
/// to the ACTIVE WINDOW of the MPS as a genuine rotation of the state (not a
/// basis relabel), so the MPS stays in the Schrodinger picture. The idea was that
/// natural orbitals chosen from that Schrodinger cc would keep the window small
/// for a state Fbr_dyn's interaction-picture cc makes precess (e.g. the ground
/// state).
///
/// It does NOT work, for a reason that is itself the finding. This code drops the
/// bath rotation's Slater block ("empty/full orbitals, a phase changes nothing").
/// That is false: after the natural-orbital demotion step, the empty and full
/// Slater orbitals are natural orbitals in which H_bath is NOT diagonal -- it has
/// K_ij matrix elements coupling an empty orbital to a full one. exp(-i H_bath dt)
/// on that block hops charge empty<->full (an O(1) effect), so:
///   - dropping the block loses that hopping  -> wrong physics (large drift), and
///   - the same coupling partially fills Slater orbitals that must then be
///     promoted back -> the active window grows, not shrinks.
/// Measured on |gs> at U=0: arrow-only (bath step disabled) stays at n_active=4,
/// drift ~5e-8; enabling the bath rotation blows the window to the full system and
/// corrupts the state. So the ~20-orbital ground-state window of Fbr_dyn is not a
/// formulation artifact -- it is this physical empty<->full coupling, and the
/// naive state-rotating route cannot beat the interaction picture.
///
/// H_imp = H - H_bath is the impurity on-site + hybridization ("arrow") plus the
/// Hubbard U (in the MPO); H_bath is the diagonal bath in star geometry.
struct Fbr_dyn_frame : detail::DynCommon {
    using Common = detail::DynCommon;
    using State = typename Common::State;

    State fb;                ///< the current few-body MPS (Schrodinger frame)
    double energy=-1000;
    arma::cx_mat Karrow;     ///< H_imp kinetic part (Kstar with the bath-bath block zeroed), star basis
    arma::cx_mat Dbath;      ///< H_bath: bath energies on the diagonal, 0 on impurity, star basis

    explicit Fbr_dyn_frame(ImpurityParam const& param_, State const& fb_, double dt_=0.1)
        : Common(param_,fb_,dt_)
        , fb { fb_ }
    {
        fb.n_sv=this->n_sv;
        Karrow=arma::conv_to<arma::cx_mat>::from(param.Kmat);
        if (!bath_pos.empty()) Karrow.submat(bath_pos,bath_pos).zeros();   // remove H_bath
        int L=param.length();
        arma::vec d(L,arma::fill::zeros);
        d(bath_pos)=arma::vec(param.Kmat.diag())(bath_pos);
        Dbath=arma::diagmat(arma::conv_to<arma::cx_vec>::from(d));
        this->K=build_arrow_K();
    }

    /// H_imp kinetic part in the current MPS-orbital basis: M^dag Karrow M,
    /// M = rot_star^dag fb.rot (MPS orbitals expressed in the star basis).
    arma::cx_mat build_arrow_K() const
    {
        arma::cx_mat M = rot_star.t() * fb.rot;
        return M.t() * Karrow * M;
    }

    void iterate(TdvpParam args={})
    {
        this->K=build_arrow_K();

        apply_plan(fb.plan_representative(this->K,0));
        apply_plan(fb.plan_representative(this->K,1));
        apply_plan(fb.plan_active_representative(this->K));
        do_tdvp(args);                 // exp(-i H_imp dt) on the MPS
        apply_bath_frame();            // exp(-i H_bath dt) as a window rotation of the state
        apply_plan(fb.plan_natural_orbitals(fb.cc));   // re-select from the Schrodinger cc
    }

    /// Actively rotate the state's active window by exp(-i H_bath dt) (a genuine
    /// single-particle evolution of the MPS, not a basis relabel). H_bath
    /// restricted to the window is Hermitian; its impurity columns are unit
    /// vectors on impurity star sites where Dbath==0, so the impurity is never
    /// rotated. O(n_active): Uw is n_active x n_active and the Givens circuit
    /// stays inside the window.
    void apply_bath_frame()
    {
        auto [a,b]=fb.range(Part::active);
        if (b-a<=1) return;
        arma::cx_mat Mw = rot_star.t() * fb.rot.cols(a,b-1);   // L x n_active
        arma::cx_mat Ha = Mw.t() * Dbath * Mw;                 // n_active x n_active
        Ha = 0.5*(Ha+Ha.t());
        arma::cx_mat Uw = exp_iH((Ha*dt).eval());              // exp(-i Ha dt)

        auto givens = givens_for_rot_right(Uw);
        OrbitalUpdate<cmpx> update(a,b);
        update.append(arma::regspace<arma::uvec>(a,b-1), givens_transpose(givens));
        update.active={a,b};

        // fb.apply is a PASSIVE basis change: it rotates the MPS and compensates
        // in rot so the physical state is unchanged. We want the ACTIVE evolution
        // U|psi>: keep the MPS rotation, undo the rot compensation, and rebuild cc
        // from the rotated state. (Uw is zero on the impurity, so rot's impurity
        // columns are untouched and the impurity stays a single MPS orbital.)
        arma::cx_mat rot_before = fb.rot;
        fb.apply(update);
        fb.rot = rot_before;
        fb.update_cc();
    }

    void apply_plan(OrbitalUpdate<cmpx> const& update)
    {
        this->apply_plan_to_K(fb,update);
        fb.apply(update);
    }

    void do_tdvp(TdvpParam args={})
    {
        auto [a,b]=fb.range(Part::active);
        auto mpo=Common::full_hamiltonian(fb,a,b);
        energy=this->evolve_one(fb,mpo,args);
    }

    // n_iter stays 0, so effective_rot carries no ip_phase: fb.rot IS the frame.
    arma::cx_mat effective_rot() const { return Common::effective_rot(fb); }
    arma::cx_mat correlator() const { return Common::correlator(fb); }
    cmpx correlator(int i, int j) const { return Common::correlator(fb,i,j); }
};

} // namespace fbr

#endif // FBR_DYN_FRAME_H
