# Impurity dynamics using MPS with orbital rotation 
This code is associated with the paper [Nunez2025](https://doi.org/10.48550/arXiv.2503.13706)

![](results/Banner.gif)

For the simpler **ground state** problem, we are able to solve 1 million orbitals in 1 hour of single core. See the branch [million](https://github.com/yurielnf/noip/tree/million)

## Dependencies
- [ITensor](https://github.com/ITensor/ITensor) for MPS manipulation
- [TDVP](https://github.com/ITensor/TDVP) for bechmarking our code
- [armadillo](http://arma.sourceforge.net/) for linear algebra. Armadillo depends on **blas**, **lapack**.
- [Catch2](https://github.com/catchorg/Catch2) for testing

## Compiling
- install your favorite `blas`/`lapack` libraries (for instansce `mkl`) including their `-dev` versions.
- download the [ITensor](https://github.com/ITensor/ITensor) library to `${HOME}/opt` (our cmake links to this place) and compile it following `INSTALL.md`
- download the [TDVP](https://github.com/ITensor/TDVP) library to `${HOME}/opt`
- compile our library:
```bash
git clone https://github.com/FastQuantum/impurityMPS.git
mkdir build
cd build
cmake ..
make -j4
``` 

## Running 
You will get a binary file per example, so you can type for instance
```bash
example/irlm_gs
```
We have tested the programs with one core, so we recommend before running
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential
```
Enjoy it!
