# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

impurityMPS is a C++17 header-only library for simulating quantum impurity dynamics using Matrix Product States (MPS). It combines ITensor (DMRG/TDVP tensor network framework) with custom orbital-rotation algorithms to efficiently handle large systems (up to ~1M orbitals) by keeping only a small "active" set in the full MPS.

Associated paper: https://arxiv.org/abs/2503.13706

## Build

**Prerequisites (must be manually installed to `$HOME/opt/`):**
- ITensor: `$HOME/opt/ITensor/` — MPS library
- TDVP: `$HOME/opt/TDVP/` — time-dependent variational principle header library

**Auto-fetched via CMake FetchContent:** Armadillo, Catch2, nlohmann_json

```bash
mkdir build && cd build
cmake ..
make -j4
```

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` when running executables — the code is not thread-safe and relies on single-threaded BLAS.

## Tests

```bash
cd build
ctest
# or directly:
./test/fbr_test
# run a single test tag:
./test/fbr_test "[givens]"
```

Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` for every test binary; running several of them at once on separate cores is fine.

Tests use Catch2 v2, in six executables. `fbr_test` holds the fast unit tests (`test_givens`, `test_graph`, `test_fb_mps`, `test_green_overlap`, ...). The other five need their own executable, because the TDVP headers define non-inline functions. Four compare a solver against a committed reference: `fbr_test_fbr` (spin-symmetric), `fbr_test_block` (generic spin), `fbr_test_ns` (spinless) and `fbr_test_green` (Green functions of the SIAM and the IRLM against their chain baselines, L=100). The fifth, `fbr_test_green_sep`, checks the separate-frame Green function against the exact U=0 result. The reference runs stop early by default (t=5 for the correlators, t=2 for the Green functions); configure with `-DFBR_ENABLE_LONG_TEST=ON` to go to t=20. The parameters covered are L=100 and V=0.1, with U=0.1 and 0.2 for the SIAM (correlators and Green function), and U=0.1, 0.2 and -0.2 for the IRLM Green functions. `fbr_test_fbr` additionally replays the first few steps of the L=1000 SIAM quench against `test/ref/output/fbr_dyn_siam_L1000_U*.txt` (tag `[large_l]`, regenerate with `example/fbr_dyn_siam.cpp`) as a smoke test that the solver still scales to L=1000.

## Where programs and data go

- `test/ref/`: **reference data only**, i.e. the baselines that new code and new experiments are measured against. `chain_*` is the real-space chain (the ground truth), `star_*` is the full-length star, and `fbr_*` is kept only when a test replays it. Data goes in `test/ref/output/`; build with `-DFBR_EXAMPLE_REF=ON`; see `test/ref/readme.txt`.
- `app/`: **numerical experiments** (method comparisons, benchmarks, tuning drivers, negative results). Run the programs from the repo root; they write `app/output/<program>_L<L>_U<U>.dat`, and only `.dat` files are versioned. Plotting and report scripts live in `app/plot/` (`python3 app/plot/report.py`). Build with `-DFBR_APP=ON`; see `app/README.md`.
- `example/`: short programs showing how to use the API.

## Architecture

### Core idea: active orbital window

The library tracks a small window of `n_active` orbitals in a full MPS (with entanglement) while the remaining orbitals are described by a Slater determinant. After each DMRG or TDVP sweep, the one-particle density matrix is diagonalized to find natural orbitals; the MPS is rotated into this basis to minimize entanglement, and orbitals near half-filling are promoted into the active window while those near 0 or 1 are demoted back to the Slater part.

### Key headers (`include/fbr/`)

All library types live in `namespace fbr`. Single-file programs include `#include "fbr/fbr.h"`, which pulls in everything and documents the short vocabulary; add `using namespace fbr;` in consumer code. A multi-TU target must include the specific headers instead and keep `fbr_dyn.h` to one file — upstream TDVP defines `addBasis` non-inline.

| Header | Purpose |
|---|---|
| `fbr.h` | The umbrella: the four-step recipe and the dozen names it uses |
| `fb_mps.h` | `Fb_mps<T>` — few-body MPS state with rotation matrix `rot`, correlation matrix `cc` and active window `active`. One class for the three orbital layouts, chosen by the `Layout` of the model (`ImpurityParam::layout`), or passed directly to `from_slater`: `leading` (spinless), `spin_symmetric`, `spin_block` |
| `layout.h` | `Spin`, `Layout`, `Part` and `Range` — the chain geometry vocabulary shared by the model and the state. `fb.range(Part::active)` or `fb.range(Part::slater,dw)` names any part of the chain |
| `impurity_param.h` | `ImpurityParam` — kinetic matrix `Kmat`, interaction `Umat`, impurity positions and the chain `layout`; `to_star()` transforms to star geometry (leading or centered, per `layout`), and the solvers `validate()` whatever they are handed |
| `initial_state.h` | `slater<T>(model,ek)` — the Slater state a model starts from, in its own frame, filling and layout (`ek` defaults to `Kmat.diag()`) |
| `fbr_gs.h` | `Fbr_gs` — ground state solver: DMRG loop + orbital rotation, for all three layouts |
| `fbr_dyn.h` | `Fbr_dyn` — dynamics: TDVP loop + orbital rotation: `Fbr_dyn(model,fb,dt)`. `Fbr_dyn_shared` evolves several states in one common orbital basis: `Fbr_dyn_shared(model,states,dt)`. The basis follows the first state (the master; for a Green function, the excitation `c†|psi0>`), and the window is widened to hold the others |
| `green_overlap.h` | `overlap(A,B)` and `c_element(A,B,i)` between states in *different* orbital frames (Green functions from separately evolved states); `align_to_frame` |
| `fbr_dyn_frame.h` | `Fbr_dyn_frame` — co-moving-frame dynamics. **Negative result**, kept for the record and not included by `fbr.h`; used only by `app/gs_frame_vs_ip_siam.cpp` |
| `graph.h` | Index/set utilities (`iota`, `regspace`, `set_diff`) and `fbr::graph::find_islands` for connected-component detection |
| `itensor_utils.h` | `DmrgParam`, `TdvpParam`; `gates_from_givens()` — converts Givens rotations to ITensor `BondGate`s |
| `givens_rotation.h` | Givens rotations applied to MPS: `givens_for_rot_left()`, `exp_iH()`, `ilog_matrix()` |

### Hamiltonian geometry

Input is a generic kinetic matrix `Kmat`. `to_star()` transforms it to star geometry (bath modes are orthogonalized so the bath Hamiltonian is diagonal), enabling efficient DMRG. After rotation, the system is a central impurity coupled to a set of bath orbitals.

### Spin variants

The `spin_symmetric` layout supports spin up/down having equivalent properties (spin flip commute with the Hamiltonian): only the down sector is computed and the up one is its mirror image. Use it with `ImpurityParam{.layout=spin_symmetric}`. The impurity is represented as -----spin-up-----xx XX------spin-down------- whre xx and XX are the non-rotating orbitals with spin up and down, respectively. The state must be spin-flip symmetric too, not just the model: a spin-polarized state such as the Green-function excitation c₀↑†|gs> needs `spin_block`, and `Fbr_dyn`/`Fbr_dyn_shared` throw if handed one under `spin_symmetric`. `spin_block` is the generic-spin variant (no spin-flip symmetry assumed), and `leading` is the spinless one, |imp|active|slater|, which is the same chain with an empty up sector (`mid()==0`).

### Examples (`example/`)

Each `.cpp` is a standalone executable. Key ones:
- `fbr_gs_irlm.cpp` — ground state, spinless IRLM
- `fbr_gs_siam.cpp` — ground state, SIAM (spin)
- `fbr_dyn_irlm.cpp` — real-time dynamics, spinless, complex MPS
- `fbr_dyn_siam.cpp` — dynamics with spin
- `fbr_green_irlm.cpp` — Green function from three states in one basis (`Fbr_dyn_shared`), checked against the exact non-interacting result
- `fbr_green_irlm_separate.cpp` — the same Green function, with each state in its own frame (`green_overlap.h`)

## Conventions

- The library is almost entirely header-only; `empty.cpp` exists only to generate the static archive `libimpurityMPS.a`.
- Template parameter `T` is `double` for ground state, `std::complex<double>` for dynamics.
- ITensor `Fermion` sites with particle-number conservation are used throughout; Jordan-Wigner strings are handled automatically by ITensor.
- `n_active` controls the MPS window size; orbital promotion/demotion uses occupancy thresholds relative to a tolerance `tol`.
