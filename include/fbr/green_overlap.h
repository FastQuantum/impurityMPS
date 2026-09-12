#ifndef FBR_GREEN_OVERLAP_H
#define FBR_GREEN_OVERLAP_H

#include "fb_mps.h"
#include "givens_rotation.h"
#include "itensor_utils.h"
#include "orbital_update.h"

#include <armadillo>
#include <itensor/all.h>
#include <stdexcept>

namespace fbr {

/// Overlaps between two few-body MPS states that live in DIFFERENT orbital
/// frames. Green functions need <A| c_i |B> with A and B evolved separately, so
/// each keeps its own small active window; only here, at the measurement, are
/// the two brought into one common basis.
///
/// The states must share the ITensor site indices (build one from a copy of the
/// other) and the same star frame and time step, so the interaction-picture bath
/// phase is common to both and cancels in the overlap. Then the relative frame
/// is the plain unitary  W = A.rot^dag B.rot,  and re-expressing B in A's frame
/// is a single-particle rotation applied to B's MPS.
namespace overlap_detail {

/// Givens reduction of `rot` to (phased) identity, LEFT-stair like
/// givens_for_rot_left, but skipping any pivot whose lower entry is already below
/// `tol`. This is what keeps the alignment O(n_active): the plain reduction turns
/// a pair of ~machine-eps entries (all the frozen/deep-bath columns of a
/// near-identity relative frame) into a full 45-degree rotation -- spurious gates
/// that only cancel in aggregate and that shred an entangled MPS. Skipping them
/// leaves only the genuinely non-trivial gates, in the region where the two
/// frames actually differ.
template<class T>
std::vector<GivensRot<T>> givens_align_left(arma::Mat<T> rot, double tol)
{
    std::vector<GivensRot<T>> givens;
    for (int j=0; j<(int)rot.n_cols; ++j) {
        arma::Col<T> v = rot.col(j);
        std::vector<GivensRot<T>> layer;
        for (int i=(int)v.size()-2; i>=j; --i) {
            if (std::abs(v[i+1]) < tol) continue;         // lower entry already zero
            layer.push_back(GivensRot<T>::create_from_pair(i, v[i], v[i+1], false, &v[i]));
        }
        apply_givens(layer, rot);
        for (auto const& g : layer) givens.push_back(g);
    }
    return givens;
}

/// Multiply the |1> amplitude of MPS site k (0-based) by `phase`. A diagonal,
/// even, single-site gate: no Jordan-Wigner string, no orthogonality-center move
/// needed beyond the local set.
inline void apply_site_phase(itensor::MPS& psi, itensor::Fermion const& sites,
                             int k, cmpx phase)
{
    auto s=sites(k+1);
    itensor::ITensor op(itensor::dag(s), prime(s));
    op.set(s(1), prime(s)(1), 1.0);
    op.set(s(2), prime(s)(2), phase);
    auto A=op*psi(k+1);
    A.noPrime();
    psi.set(k+1, A);
}

} // namespace overlap_detail

/// The contiguous band of orbitals [a,b) where two frames differ. Outside it the
/// two states share the frame column for column -- the deep bath is never rotated,
/// so the states are identical there and only the band needs aligning. Found by
/// growing out from the union active window while the boundary columns still
/// differ, which costs O(band * L), not O(L^2). It assumes the mismatch is
/// contiguous with the active window: true for the leading and centered layouts,
/// whose window sits in one block and whose demoted orbitals pile up at its edges.
template<class T>
Range mismatch_band(Fb_mps<T> const& A, Fb_mps<T> const& B, double tol)
{
    int L = A.length();
    int lo = std::min(A.active.a, B.active.a);
    int hi = std::max(A.active.b, B.active.b);
    auto differ = [&](int k){ return arma::norm(A.rot.col(k) - B.rot.col(k)) > tol; };
    while (lo > 0 && differ(lo - 1)) --lo;
    while (hi < L && differ(hi))     ++hi;
    return {lo, hi};
}

/// Rotate `fb` into the orbital frame `target` (an L x L unitary in the same
/// original basis as fb.rot), leaving the physical state unchanged. After the
/// call fb.rot == target (to `cutoff`), fb.cc and fb.psi are transformed to
/// match, so the MPS may be contracted against another state written in `target`.
///
/// `band` restricts the work to orbitals [band.a, band.b): the caller guarantees
/// fb.rot already equals target outside it (e.g. via mismatch_band), so the
/// relative rotation, its circuit and the phase clean-up all live in the band and
/// the cost is O(band^2 * L) instead of the O(L^3) of a full-frame alignment.
/// band = {-1,-1} (the default) means the whole chain.
///
/// `cutoff` sets the accuracy/cost trade-off: the reduction drops any Givens whose
/// entry is below it, gateTEvol truncates the MPS to it, and a residual phase
/// closer to 1 than it is left alone. cutoff<0 uses fb.tol. For a Green-function
/// overlap the rotated MPS is a throwaway, so a loose cutoff (1e-3) is plenty.
template<class T>
void align_to_frame(Fb_mps<T>& fb, arma::Mat<T> const& target, double cutoff=-1,
                    Range band={-1,-1})
{
    if (target.n_rows!=(arma::uword)fb.length())
        throw std::invalid_argument("align_to_frame: target size mismatch");
    if (cutoff<0) cutoff=fb.tol;
    int a = band.a<0 ? 0 : band.a;
    int b = band.b<0 ? fb.length() : band.b;

    // Relative rotation on the band only: fb.rot[:,a:b] * G = target[:,a:b]. The
    // band is closed under the mismatch (identical frames outside), so G is the
    // full band x band unitary and no weight leaks out.
    arma::Mat<T> G = fb.rot.cols(a,b-1).t() * target.cols(a,b-1);

    // givens_dagger(givens_align_left(G)) is a nearest-neighbour circuit whose
    // matrot equals G up to a diagonal column phase D; applying it as a frame
    // change gives fb.rot[:,a:b] -> target[:,a:b]*D, and D is cleaned up below.
    auto givens = givens_dagger(overlap_detail::givens_align_left(G, cutoff));
    OrbitalUpdate<T> update(fb.active.a, fb.active.b);
    update.append(arma::regspace<arma::uvec>(a, b-1), givens);

    std::vector<GivensRot<T>> circuit;
    for (auto const& gate : update.gates) {
        gate.apply_as_frame(fb.rot);
        gate.apply_as_correlator(fb.cc);
        if (std::abs(gate.s) > cutoff)                  // skip ~identity gates
            circuit.push_back(gate.givens(gate.a).transpose());
    }
    auto gates = gates_from_givens(fb.sites, circuit);
    if (!gates.empty())
        itensor::gateTEvol(gates,1,1,fb.psi,
                           {"Cutoff",cutoff,"Quiet",true,"Normalize",false,"ShowPercent",false});

    // Remove the residual column phase D = diag(target^dag fb.rot) on the band.
    for (int k=a; k<b; ++k) {
        T d = arma::cdot(target.col(k), fb.rot.col(k));   // (target^dag fb.rot)_kk
        if (std::abs(d-T(1)) > cutoff) {
            d /= std::abs(d);                              // unit phase
            overlap_detail::apply_site_phase(fb.psi, fb.sites, k, d);
            fb.rot.col(k) *= std::conj(d);                 // -> target.col(k)
            fb.cc.row(k)  *= std::conj(d);
            fb.cc.col(k)  *= d;
        }
    }
    fb.active = update.active;
}

/// <A|B>, with A and B in possibly different frames (shared sites, star frame and
/// time step). B is rotated into A's frame -- only within the band where the two
/// frames differ -- and the two MPS are contracted; see align_to_frame for
/// `cutoff`. Pass full=true to align the whole chain (the reference path).
template<class T>
cmpx overlap(Fb_mps<T> const& A, Fb_mps<T> const& B, double cutoff=-1, bool full=false)
{
    auto Bc = B;
    double tol = cutoff<0 ? A.tol : cutoff;
    align_to_frame(Bc, A.rot, cutoff, full ? Range{-1,-1} : mismatch_band(A, B, tol));
    return itensor::innerC(A.psi, Bc.psi);
}

/// <A| c_i |B> for i a non-rotating impurity site (see Fb_mps::apply_local_op),
/// with A and B in possibly different frames. Used for the Green function
///     G(i,j,t) = -i <psi0| c_i(t) c_j^dag |psi0> = -i <A(t)| c_i |B(t)>,
/// A=psi0, B=c_j^dag psi0, evolved in separate frames.
///
/// B is rotated into A's frame by a Givens circuit -- restricted to the band where
/// the two frames actually differ (see mismatch_band) -- then contracted and
/// discarded. `cutoff` may be loose (1e-3 is plenty). cutoff<0 uses A.tol; full=true
/// aligns the whole chain (the O(L^3) reference path, for validation).
template<class T>
cmpx c_element(Fb_mps<T> const& A, Fb_mps<T> const& B, int i, double cutoff=-1, bool full=false)
{
    auto Bc = B;
    double tol = cutoff<0 ? A.tol : cutoff;
    align_to_frame(Bc, A.rot, cutoff, full ? Range{-1,-1} : mismatch_band(A, B, tol));
    auto Ai = A;
    Ai.apply_local_op("Cdag", i);       // |c_i^dag A>
    return itensor::innerC(Ai.psi, Bc.psi);
}

} // namespace fbr

#endif // FBR_GREEN_OVERLAP_H
