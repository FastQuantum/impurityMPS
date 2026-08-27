// Scaling benchmark for the spin dynamics timestep.
//
// Reports per-timestep wall time vs L. With the star Hamiltonian given, every
// L-dependent operation inside iterate() (building K and planning/applying orbital
// updates) is O(L^2); the TDVP/MPS work is independent of L.
//
//   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./bench_buildK

#include "fbr/fbr_dyn.h"

#include <armadillo>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace arma;
using namespace fbr;

static auto makeSolver(int L, double dt)
{
    ImpurityParam model;
    double U = 0.2, V = 0.1;
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = -U / 2;
    K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {2, 0, 1, 3}, .layout=spin_symmetric};
    model.to_star();

    auto ek = vec{model.Kmat.diag()};
    ek[L / 2 - 1] = ek[L / 2] = -10;
    ek[L / 2 - 2] = ek[L / 2 + 1] = 10;
    auto fb = slater<cmpx>(model, ek);
    auto solver = Fbr_dyn(model, fb, dt);
    solver.fb.tol = 1e-10;
    return solver;
}

using clock_t_ = std::chrono::steady_clock;
static double ms_since(clock_t_::time_point t0)
{
    return std::chrono::duration<double, std::milli>(clock_t_::now() - t0).count();
}

int main()
{
    double dt = 0.1;
    TdvpParam args{.n_iter_diag = 6, .epsilon_M = 0};  // epsilon_M=0 -> no expansion; n_krylov inert

    std::printf("%8s %14s %14s %14s\n", "L", "buildK_ms", "iterate_ms", "iterate/L^2");
    std::printf("%8s %14s %14s %14s\n", "----", "--------", "----------", "-----------");

    for (int L : {500, 1000, 2000, 4000, 8000}) {
        auto solver = makeSolver(L, dt);

        // build_K on a random unitary frame (pure L-dependent linear algebra).
        arma_rng::set_seed(11);
        cx_mat G = cx_mat(L, L, fill::randn) + cmpx(0, 1) * cx_mat(L, L, fill::randn);
        cx_mat Q, R;
        qr(Q, R, G);
        solver.fb.rot = Q;
        solver.n_iter = 5;
        int rk = L <= 2000 ? 10 : 4;
        auto tb = clock_t_::now();
        for (int r = 0; r < rk; r++) { volatile double s = std::abs(solver.build_K()(0,0)); (void)s; }
        double buildK_ms = ms_since(tb) / rk;

        // Full iterate(): rebuild a fresh solver (build_K test trashed fb.rot/state).
        auto run = makeSolver(L, dt);
        run.iterate(args);                 // warm up (sets active window)
        run.iterate(args);
        const int nstep = 3;
        auto ti = clock_t_::now();
        for (int s = 0; s < nstep; s++) run.iterate(args);
        double iterate_ms = ms_since(ti) / nstep;

        double Ld = L;
        std::printf("%8d %14.3f %14.3f %14.3e\n",
                    L, buildK_ms, iterate_ms, iterate_ms / (Ld * Ld));
        std::fflush(stdout);
    }
    return 0;
}
