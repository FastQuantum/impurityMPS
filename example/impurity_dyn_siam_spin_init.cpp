#include "impurityMPS/impurity_dyn_spin.h"
#include "impurityMPS/impurity_spin_init.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;

int main()
{
    const int    L    = 100;
    const int    nImp = 4;
    const double U    = 0.2;
    const double V    = 0.1;
    const double dt   = 0.1;
    const int    nBath= L/2 - nImp/2;

    // K in alternating-pair ordering: arb[2*i] = block_up[i], arb[2*i+1] = block_dw[i]
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L-2; i += 2) K(i,i+2)=K(i+2,i)=0.5;    // up chain
    for (int i = 1; i < L-2; i += 2) K(i,i+2)=K(i+2,i)=0.5;    // dw chain
    int up_phys = 2*(nBath + nImp/2 - 1);   // arb index of up physical impurity (=98)
    int dw_phys = 1;                         // arb index of dw physical impurity
    K(up_phys, up_phys) = -U/2;
    K(dw_phys, dw_phys) = -U/2;
    K(up_phys-2, up_phys) = K(up_phys, up_phys-2) = V;  // up_buf–up_phys
    K(dw_phys,   dw_phys+2) = K(dw_phys+2, dw_phys) = V;  // dw_phys–dw_buf

    // physical sites listed first per spin (required by ImpuritySpinInit)
    vector<int> impPos = {up_phys, up_phys-2, dw_phys, dw_phys+2};

    mat Umat(nImp, nImp, fill::zeros);
    Umat(0, 2) = U;   // up-phys (index 0) -- dw-phys (index 2)

    auto init = ImpuritySpinInit(K, Umat, impPos);

    // force initial occupation: physical imps occupied, buffers empty
    vec ek = init.model.param.Kmat.diag();
    for (int p : init.model.param.impPos())
        ek[p] = (ek[p] < -U/4) ? -10.0 : +10.0;
    auto fb = Fb_mps_spin<cmpx>::from_slater(init.model.param.rot * cmpx(1,0), ek, L/2, nImp);

    auto solver = Impurity_dyn_spin(init.model, fb, dt);
    solver.fb.tol = 1e-12;

    cout << "time m <n0> <n1> nActive\n" << setprecision(12);
    for (int i = 0; i*dt < L; i++) {
        solver.iterate({.max_bond_dim=2048, .nIter_diag=16, .epsilonM=1e-4});
        double n0 = solver.fb.occupations_ni()(L/2);
        double n1 = solver.fb.occupations_ni()(L/2 + 1);
        cout << (i+1)*dt << " " << maxLinkDim(solver.fb.psi) << " " << n0 << " " << n1
             << " " << solver.fb.p2 - solver.fb.p1 << "\n";
    }
}
