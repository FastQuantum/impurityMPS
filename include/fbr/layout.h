#ifndef FBR_LAYOUT_H
#define FBR_LAYOUT_H

namespace fbr {

enum Spin{up, dw};

/// Arrangement of the orbitals along the chain, and so of the active window.
///
/// The chain always reads
///     |slater_up|active_up|imp_up|imp_dw|active_dw|slater_dw|
/// with the non-rotating impurity orbitals around the center. `leading` is the
/// degenerate case with an empty up sector, where the chain is simply
/// |imp|active|slater|.
///
/// Both the model (ImpurityParam, which produces this geometry in to_star) and
/// the state (Fb_mps, which lives in it) are described by the same Layout.
enum Layout {
    leading,        ///< |imp|active|slater|, a single sector (spinless)
    spin_symmetric, ///< centered window, spin up is the reflection of spin down
    spin_block      ///< centered window, the two spin blocks are independent
};

} // namespace fbr

#endif // FBR_LAYOUT_H
