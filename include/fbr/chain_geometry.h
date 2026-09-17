#ifndef FBR_CHAIN_GEOMETRY_H
#define FBR_CHAIN_GEOMETRY_H

namespace fbr {

enum Spin{up, dw};

/// Orbital ordering and spin symmetry used by the chain and active window.
/// standard is the default: impurity first, with no assumed spin symmetry.
/// It supports both spinless models and spinful models such as SIAM.
///
/// The chain always reads
///     |slater_up|active_up|imp_up|imp_dw|active_dw|slater_dw|
/// with the non-rotating impurity orbitals around the center. `standard` is the
/// degenerate case with an empty up sector, where the chain is simply
/// |imp|active|slater|.
///
/// Both the model (ImpurityParam, which produces this geometry in to_star) and
/// the state (Fb_mps, which lives in it) are described by the same Chain_geometry.
enum Chain_geometry {
    standard=0,  ///< |imp|active|slater|; no assumed spin symmetry
    spin_sym=1,  ///< centered window, spin up is the reflection of spin down
    spin_block=2 ///< centered window, the two spin blocks are independent
};

/// A half-open interval of chain positions, [a,b). Destructures, so
/// `auto [a,b] = fb.range(Part::active);` still reads the way it always did.
struct Range {
    int a=0, b=0;
    int size() const { return b-a; }
    bool empty() const { return b<=a; }
    bool contains(int i) const { return i>=a && i<b; }
    friend bool operator==(Range x, Range y) { return x.a==y.a && x.b==y.b; }
    friend bool operator!=(Range x, Range y) { return !(x==y); }
};

/// The parts of the chain a Range can name, from the center outwards:
/// the non-rotating impurity, the active window that holds it, the Slater
/// determinant beyond, everything that is not impurity (bath), and the
/// orbitals a rotation may touch (active minus impurity).
enum class Part { impurity, active, slater, bath, rotating };

} // namespace fbr

#endif // FBR_CHAIN_GEOMETRY_H
