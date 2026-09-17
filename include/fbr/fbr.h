#ifndef FBR_H
#define FBR_H

/// impurityMPS: impurity dynamics using a few-body MPS with orbital rotation.
///
/// This header pulls in everything a consumer needs. A run is always the same
/// four steps -- describe the model, transform it to star geometry, build the
/// state it starts from, iterate a solver:
///
///     ImpurityParam model {.Kmat=K, .Umat=Umat, .imp_pos={0,1}};
///     model.to_star();
///     auto fb = slater<double>(model);        // or slater<cmpx>(model,ek)
///     auto solver = Fbr_gs(model, fb);        // or Fbr_dyn(model, fb, dt)
///     for (int i=0; i<100; i++) solver.iterate();
///
/// The names that appear there, and the few that go with them:
///
///   ImpurityParam   Kmat, Umat, imp_pos, filling, geometry, and to_star()
///   Chain_geometry          standard (impurity first), spin_sym, spin_block
///                   (spin_sym needs a spin-flip symmetric state as well as
///                   model: a spin-polarized one such as c_up^dag|gs> needs spin_block)
///   slater<T>       the Slater state a model starts from
///   Fb_mps<T>       that state: rot, cc, active, and range(Part[,Spin])
///   Fbr_gs          ground state, iterate(DmrgParam)
///   Fbr_dyn         real-time evolution, iterate(TdvpParam)
///   Fbr_dyn_shared  several states evolving in one common orbital basis
///   overlap / c_element   <A|B> and <A|c_i|B> between states in DIFFERENT
///                   frames (Green functions from separately evolved states)
///
/// Fbr_gs exposes energy; measure its state through solver.fb.correlator(...).
/// Dynamics solvers provide correlator(), correlator(i,j), correlator_row(i),
/// and correlator_col(j) in the Schrodinger picture. Fbr_dyn_shared stores energies.
/// Fb_mps::occupations() reads cached orbital occupations; measure_occupations()
/// measures them from the MPS. Both use the current orbital basis.
///
/// Everything else -- the Givens rotations, the orbital-update plans, the
/// ITensor bridge -- is machinery these are built from, in the headers below.
///
/// One caveat, inherited from upstream: TDVP's basisextension.h defines
/// addBasis non-inline, so at most one translation unit per binary may include
/// this header (or fbr_dyn.h directly). Single-file programs -- every example
/// here -- are unaffected; a multi-TU target should include the specific
/// headers it needs and keep fbr_dyn.h to one file, which is what test/ does.

#include "chain_geometry.h"
#include "impurity_param.h"
#include "fb_mps.h"
#include "initial_state.h"
#include "fbr_gs.h"
#include "fbr_dyn.h"
#include "green_overlap.h"

#endif // FBR_H
