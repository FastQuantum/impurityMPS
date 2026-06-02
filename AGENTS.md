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

### Core idea: active orbital window

The library tracks a small window of `nActive` orbitals in a full MPS (with entanglement) while the remaining orbitals are described by a Slater determinant. After each DMRG or TDVP sweep, the one-particle density matrix is diagonalized to find natural orbitals; the MPS is rotated into this basis to minimize entanglement, and orbitals near half-filling are promoted into the active window while those near 0 or 1 are demoted back to the Slater part.


### Hamiltonian geometry

Input is a generic kinetic matrix `Kmat`. `toStar()` transforms it to star geometry (bath modes are orthogonalized so the bath Hamiltonian is diagonal), enabling efficient DMRG. After rotation, the system is a central impurity coupled to a set of bath orbitals.

### Spin variants

Files ending in `_spin` support spin up/down having equivalent properties (spin flip commute with the Hamiltonian). The impurity is represented as -----spin-up-----xx XX------spin-down------- where xx and XX are the non-rotating orbitals with spin up and down, respectively
