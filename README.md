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
1) Install your favorite `blas`/`lapack` libraries (for instance `mkl`) including their `-dev` versions.

2) Download the [ITensor](https://github.com/ITensor/ITensor) library to `${HOME}/opt` (our cmake links to this place) and compile it following `INSTALL.md`

3) Replace the line 371 of `itensor/mps/dmrg.h` saying `const int N = length(psi);` to
```c++
const int N = args.getInt("MaxSite",length(psi));
``` 

4) Download the [TDVP](https://github.com/ITensor/TDVP) library to `${HOME}/opt`

5) Compile our library:
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
build/example/impurity_gs_irlm
```
We have tested the programs with one core, so we recommend before running
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential
```
## Adding your application
Now you can add your own application in the folder `impurityMPS/example` and recompile
```bash
cd build
cmake ..
make -j4
```
The new binary will appear at `build/example`.

## Example code for ground state
Let's find the ground state of the IRLM model:
```c++
#include "impurityMPS/impurity_gs.h"
#include <iostream>
#include <iomanip>

using namespace std;

int main()
{
    int L=1000;
    double U=0.5;
    arma::sp_mat K(L,L);
    {
        for(auto i=1; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=0.5;
        K(0,1)=K(1,0)=0.1;
        K(0,0)=-U/2;
        K(1,1)=-U/2;
    }
    arma::sp_mat Umat(2,2);
    Umat(0,1)=U;
    auto Kstar=ImpurityParam::to_star_kin_tridiag(K, Umat.n_rows);
    auto model0=Impurity_gs({.Kstar=Kstar, .Umat=Umat});

    { // optional: force impurity ocupation |10>
        auto ek=arma::vec {Kstar.diag()};
        ek[0]=-10;
        ek[1]=10;
        model0.prepareSlaterGs(ek,L/2);
    }

    cout<<"iteration nActive energy time\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<30;i++){
        model0.extract_f(0.0);
        model0.extract_f(1.0);
        model0.doDmrg();
        model0.rotateToNaturalOrbitals();
        cout<<i+1<<" "<<model0.nActive<<" "<<model0.energy<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
    }
    return 0;
}
```
The output is
```bash
iteration nActive energy time(s)
...
25 12 -318.016525206 0.596198
26 12 -318.016525213 0.636912
27 12 -318.016525221 0.657342
28 12 -318.016525226 0.618605
29 12 -318.016525228 0.62662
30 12 -318.016525231 0.617154
