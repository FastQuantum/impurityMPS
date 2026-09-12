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

/// The band of orbitals [a,b) where two frames differ: the hull of the columns k
/// with |A.rot.col(k) - B.rot.col(k)| > tol. That norm is the norm of column k of
/// W - I, W = A.rot^dag B.rot, so the band is the hull of the support of W - I.
/// Outside it W is the identity, hence block diagonal with a unitary band block:
/// the band is closed and aligning it alone is exact (to tol).
///
/// The mismatch is NOT contiguous with the active window: the representative
/// rotations reach Slater orbitals anywhere in the chain, so a column that happens
/// to agree may sit between two that do not. The hull is therefore found by
/// scanning inward from both chain ends, O((L - band) * L). An empty band (equal
/// frames) comes back with a == b.
template<class T>
Range mismatch_band(Fb_mps<T> const& A, Fb_mps<T> const& B, double tol)
{
    int lo = 0, hi = A.length();
    auto differ = [&](int k){ return arma::norm(A.rot.col(k) - B.rot.col(k)) > tol; };
    while (lo < hi && !differ(lo))     ++lo;
    while (hi > lo && !differ(hi - 1)) --hi;
    return {lo, hi};
}

/// Rotate `fb` into the orbital frame `target` (an L x L unitary in the same
/// original basis as fb.rot), leaving the physical state unchanged. After the
/// call fb.rot == target (to `cutoff`), fb.cc and fb.psi are transformed to
/// match, so the MPS may be contracted against another state written in `target`.
///
/// `band` restricts the work to orbitals [band.a, band.b), which must be closed:
/// fb.rot already equals target outside it (mismatch_band gives the smallest such
/// band). The relative rotation, its circuit and the phase clean-up then all live
/// in the band, and the cost is O(band^2 * L) instead of the O(L^3) of a
/// full-frame alignment. A band that is not closed -- some target column of the
/// band has weight outside the band of fb.rot -- is detected and the whole chain
/// is aligned instead. band = {-1,-1} (the default) means the whole chain; an
/// empty band means the frames are already equal and nothing is done.
///
/// `cutoff` sets the accuracy/cost trade-off, as an amplitude: the reduction drops
/// any Givens whose entry is below it, and a residual phase closer to 1 than it is
/// left alone. gateTEvol truncates the MPS to a discarded weight of cutoff^2 (but
/// no finer than fb.tol, the state's own truncation): a weight is a squared
/// amplitude, and a truncation at `cutoff` itself, repeated over the O(band^2)
/// gates, costs far more than `cutoff` (L=100 IRLM, cutoff 1e-4: G off by 8e-3,
/// against 1e-4 with cutoff^2 in a 1.4x longer run). cutoff<0 uses fb.tol. The
/// rotated MPS of a Green-function overlap is a throwaway, so 1e-4 is enough.
template<class T>
void align_to_frame(Fb_mps<T>& fb, arma::Mat<T> const& target, double cutoff=-1,
                    Range band={-1,-1})
{
    if (target.n_rows!=(arma::uword)fb.length())
        throw std::invalid_argument("align_to_frame: target size mismatch");
    if (cutoff<0) cutoff=fb.tol;
    int a = band.a<0 ? 0 : band.a;
    int b = band.b<0 ? fb.length() : band.b;
    if (a>=b) return;                                     // equal frames

    // Relative rotation on the band only: fb.rot[:,a:b] * G = target[:,a:b]. For a
    // closed band G is the full band x band unitary. Otherwise a column of G has
    // lost the weight target puts outside the band: align the whole chain.
    arma::Mat<T> G = fb.rot.cols(a,b-1).t() * target.cols(a,b-1);
    if (b-a < fb.length() &&
        arma::abs(1.0 - arma::sum(arma::square(arma::abs(G)), 0)).max() > cutoff) {
        a = 0; b = fb.length();
        G = fb.rot.t() * target;
    }

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
                           {"Cutoff",std::max(cutoff*cutoff,fb.tol),"Quiet",true,
                            "Normalize",false,"ShowPercent",false});

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
/// discarded. `cutoff` may be loose (1e-4 is enough). cutoff<0 uses A.tol; full=true
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
