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
./test/impurityMPS_test
# run a single test tag:
./test/impurityMPS_test "[givens]"
```

Tests use Catch2 v2. The three test modules are `test_givens.cpp`, `test_graph.cpp`, and `test_itensor.cpp`.

## Architecture

### Core idea: active orbital window

The library tracks a small window of `nActive` orbitals in a full MPS (with entanglement) while the remaining orbitals are described by a Slater determinant. After each DMRG or TDVP sweep, the one-particle density matrix is diagonalized to find natural orbitals; the MPS is rotated into this basis to minimize entanglement, and orbitals near half-filling are promoted into the active window while those near 0 or 1 are demoted back to the Slater part.

### Key headers (`include/fbr/`)

All library types live in `namespace fbr`. Include as `#include "fbr/<header>.h"` and add `using namespace fbr;` in consumer code.

| Header | Purpose |
|---|---|
| `fb_mps.h` | `Fb_mps<T>` — few-body MPS state with rotation matrix `rot`, correlation matrix `cc` and active window `[p1,p2)`. One class for the three orbital layouts, chosen by the `Layout` passed to `from_slater`: `leading` (spinless), `spin_symmetric`, `spin_block` |
| `impurity_param.h` / `impurity_param_spin.h` | `ImpurityParam` / `ImpurityParamSpin` — kinetic matrix `Kmat`, interaction `Umat`, impurity positions; `toStar()` transforms to star geometry |
| `fbr_gs_spin.h` | `Fbr_gs_spin` — ground state solver: DMRG loop + orbital rotation |
| `fbr_dyn.h` | `Fbr_dyn<Model>` — dynamics: TDVP loop + orbital rotation, deduced from the model: `Fbr_dyn(model,fb,dt)`. `Fbr_ns_dyn<Model>` evolves several states in one common orbital basis: `Fbr_ns_dyn(model,states,dt)` |
| `graph.h` | Index/set utilities (`iota`, `regspace`, `set_diff`) and `fbr::graph::find_islands` for connected-component detection |
| `itensor_utils.h` | `DmrgParam`, `TdvpParam`; `NOGates()` — converts Givens rotations to ITensor `BondGate`s |
| `givens_rotation.h` | Givens rotations applied to MPS: `GivensRotForRot_left()`, `expIH()`, `my_svd()` |

### Hamiltonian geometry

Input is a generic kinetic matrix `Kmat`. `toStar()` transforms it to star geometry (bath modes are orthogonalized so the bath Hamiltonian is diagonal), enabling efficient DMRG. After rotation, the system is a central impurity coupled to a set of bath orbitals.

### Spin variants

The `spin_symmetric` layout supports spin up/down having equivalent properties (spin flip commute with the Hamiltonian): only the down sector is computed and the up one is its mirror image. Use it with `ImpurityParamSpin` and `Fbr_gs_spin`. The impurity is represented as -----spin-up-----xx XX------spin-down------- whre xx and XX are the non-rotating orbitals with spin up and down, respectively. `spin_block` is the generic-spin variant (no spin-flip symmetry assumed), and `leading` is the spinless one, |imp|active|slater|, which is the same chain with an empty up sector (`mid()==0`).

### Examples (`example/`)

Each `.cpp` is a standalone executable. Key ones:
- `fbr_gs_irlm.cpp` — ground state, spinless IRLM
- `fbr_gs_siam.cpp` — ground state, SIAM (spin)
- `fbr_dyn_irlm.cpp` — real-time dynamics, spinless, complex MPS
- `fbr_dyn_siam.cpp` — dynamics with spin
- `fbr_green_irlm.cpp` — Green function from three states in one basis (`Fbr_ns_dyn`), checked against the exact non-interacting result

## Conventions

- The library is almost entirely header-only; `empty.cpp` exists only to generate the static archive `libimpurityMPS.a`.
- Template parameter `T` is `double` for ground state, `std::complex<double>` for dynamics.
- ITensor `Fermion` sites with particle-number conservation are used throughout; Jordan-Wigner strings are handled automatically by ITensor.
- `nActive` controls the MPS window size; orbital promotion/demotion uses occupancy thresholds relative to a tolerance `tol`.
