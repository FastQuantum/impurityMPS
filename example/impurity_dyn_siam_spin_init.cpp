/// Compare ImpuritySpinInit (arbitrary-ordering input) against the manual
/// star-geometry setup from impurity_dyn_siam_center.cpp.
///
/// The same SIAM physics (L=100, nImp=4, U=0.2, V=0.1) is set up two ways:
///   - Reference : block ordering (up sites 0..49, dw sites 50..99), manual toStar.
///   - New       : alternating-pair ordering (arb[2i]=up_i, arb[2i+1]=dw_i),
///                 ImpuritySpinInit detects spin components automatically.
///
/// Expected: n0 (dw-physical) and n1 (dw-buffer) agree to numerical precision.

#include "impurityMPS/impurity_dyn_spin.h"
#include "impurityMPS/impurity_spin_init.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;

/// Diagonalise the bath per spin and return (Kstar, rot).
/// Layout: [up-bath | up-imp | dw-imp | dw-bath]
auto computeKstar(mat K, int nImp)
{
    int L=K.n_rows, nBath=L/2-nImp/2;
    mat Kstar(L,L,fill::zeros), rot(L,L,fill::eye);
    uvec pos_up=regspace<uvec>(0,L/2-1), pos_dw=regspace<uvec>(L/2,L-1);

    for(int s : {0,1})
    {
        uvec pos      = s==0 ? pos_up : pos_dw;
        uvec pos_bath = s==0 ? pos.head(nBath)  : pos.tail(nBath);
        uvec pos_impu = s==0 ? pos.tail(nImp/2) : pos.head(nImp/2);

        mat evec1; vec ek1;
        eig_sym(ek1,evec1,K.submat(pos_bath,pos_bath));
        uvec iek = s==0 ? sort_index(abs(ek1),"descend") : sort_index(abs(ek1));
        mat evec=evec1.cols(iek); vec ek=ek1.rows(iek);

        mat vk=K.submat(pos_impu,pos_bath)*evec;
        Kstar.submat(pos_impu,pos_impu)=K.submat(pos_impu,pos_impu);
        for(auto j=0u;j<ek.size();j++) {
            int jj=pos_bath[j]; Kstar(jj,jj)=ek[j];
            for(auto i=0u;i<pos_impu.size();i++)
                Kstar(pos_impu[i],jj)=Kstar(jj,pos_impu[i])=vk(i,j);
        }
        rot.cols(pos_bath)=rot.cols(pos_bath).eval()*evec;
    }
    return make_pair(Kstar,rot);
}

int main()
{
    const int    L    = 100;
    const int    nImp = 4;
    const double U    = 0.2;
    const double V    = 0.1;
    const double dt   = 0.1;
    const int    nBath= L/2-nImp/2;   // 48

    // Build K in block ordering: up chain 0..L/2-1, dw chain L/2..L-1
    mat K_block(L,L,fill::zeros);
    {
        for(int i=0; i<L/2-1; i++) K_block(i,i+1)=K_block(i+1,i)=0.5;
        for(int i=L/2; i<L-1; i++) K_block(i,i+1)=K_block(i+1,i)=0.5;
        K_block(nBath+nImp/2-1, nBath+nImp/2-1) = -U/2;  // up-physical on-site
        K_block(L/2, L/2)                         = -U/2;  // dw-physical on-site
        K_block(nBath, nBath+nImp/2-1) = K_block(nBath+nImp/2-1, nBath) = V;
        K_block(L/2, L/2+nImp/2-1)    = K_block(L/2+nImp/2-1, L/2)    = V;
    }

    // ================================================================
    // REFERENCE solver  (manual star geometry, same as impurity_dyn_siam_center.cpp)
    // Impurity-site order in impPos(): {up-buf=48, up-phys=49, dw-phys=50, dw-buf=51}
    // ================================================================
    mat Umat_ref(nImp,nImp,fill::zeros);
    Umat_ref(1,2)=U;   // Umat[up-phys index=1, dw-phys index=2] = U

    mat Kstar, rot_ref;
    tie(Kstar,rot_ref) = computeKstar(K_block, nImp);

    Fb_mps_spin<cmpx> fb_ref;
    {
        vec ek(Kstar.diag());
        ek[nBath+nImp/2-1] = ek[L/2]   = -10;  // up-phys(49) and dw-phys(50) occupied
        ek[nBath]           = ek[L/2+nImp/2-1] = +10;  // buffers(48,51) empty
        fb_ref = Fb_mps_spin<cmpx>::from_slater(rot_ref*cmpx(1,0), ek, L/2, nImp);
    }
    ImpuritySpin model_ref;
    {
        model_ref.param.Kmat    = Kstar;
        model_ref.param.Umat    = Umat_ref;
        model_ref.param.rot     = mat(L,L,fill::eye);
        model_ref.param.impPos0_up = model_ref.param.impPos1_up = {nBath,       nBath+nImp/2-1};
        model_ref.param.impPos0_dw = model_ref.param.impPos1_dw = {L/2,         L/2+nImp/2-1};
    }
    auto solver_ref = Impurity_dyn_spin(model_ref, fb_ref, dt);
    solver_ref.fb.tol = 1e-12;

    // ================================================================
    // NEW solver — ImpuritySpinInit with alternating-pair K ordering
    //
    // Permutation: arb[2*i] = block[i], arb[2*i+1] = block[L/2+i]
    //
    // Impurity sites in arb ordering (convention: physical FIRST per spin,
    // so that after toStar() the rotation is consistent with the Kmat):
    //   block[nBath+1=49]=up-phys  -> arb[98]    (index 0 in impPos_arb)
    //   block[nBath=48]=up-buf     -> arb[96]    (index 1)
    //   block[L/2=50]=dw-phys      -> arb[1]     (index 2)
    //   block[L/2+1=51]=dw-buf     -> arb[3]     (index 3)
    //
    // With physical sites first the Umat indices become:
    //   Umat[0,2] = U  (up-phys=index 0, dw-phys=index 2)
    // ================================================================
    uvec perm(L);
    for(int i=0; i<L/2; i++) { perm[2*i]=i; perm[2*i+1]=L/2+i; }
    mat K_arb = K_block.submat(perm,perm);

    vector<int> impPos_arb = {98, 96, 1, 3};  // up-phys, up-buf, dw-phys, dw-buf

    mat Umat_new(nImp,nImp,fill::zeros);
    Umat_new(0,2)=U;   // Umat[up-phys index=0, dw-phys index=2] = U

    auto init = ImpuritySpinInit(K_arb, Umat_new, impPos_arb);

    // Rebuild fb with forced occupation to match the reference initial state.
    // After toStar(): impPos() = {49,48,50,51} where K[49]=-U/2 (up-phys),
    // K[48]=0 (up-buf), K[50]=-U/2 (dw-phys), K[51]=0 (dw-buf).
    Fb_mps_spin<cmpx> fb_new;
    {
        vec ek(init.model.param.Kmat.diag());
        for (int p : init.model.param.impPos())
            ek[p] = (ek[p] < -U/4) ? -10.0 : +10.0;
        fb_new = Fb_mps_spin<cmpx>::from_slater(
            init.model.param.rot * cmpx(1,0), ek, L/2, nImp);
    }
    auto solver_new = Impurity_dyn_spin(init.model, fb_new, dt);
    solver_new.fb.tol = 1e-12;

    // ================================================================
    // Run and compare.
    // n0 = dw-physical at star site L/2=50   (identical in both solvers)
    // n1 = dw-buffer   at star site L/2+1=51 (identical in both solvers)
    // ================================================================
    const TdvpParam args{.max_bond_dim=512, .nIter_diag=16, .epsilonM=1e-4};

    cout << fixed << setprecision(9);
    cout << "time      n0_ref     n0_new     |diff0|    n1_ref     n1_new     |diff1|\n";
    for(int i=0; i*dt<20; i++)
    {
        solver_ref.iterate(args);
        solver_new.iterate(args);

        double n0_ref = solver_ref.fb.occupations_ni()(L/2);
        double n1_ref = solver_ref.fb.occupations_ni()(L/2+1);
        double n0_new = solver_new.fb.occupations_ni()(L/2);
        double n1_new = solver_new.fb.occupations_ni()(L/2+1);

        cout << (i+1)*dt
             << "  " << n0_ref << "  " << n0_new << "  " << abs(n0_ref-n0_new)
             << "  " << n1_ref << "  " << n1_new << "  " << abs(n1_ref-n1_new)
             << "\n";
    }
}
