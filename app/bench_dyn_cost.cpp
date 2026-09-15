// Cost of one Fbr_dyn timestep against L, phase by phase, for the three layouts.
//
// The question: is a timestep O(L^2) (orbital bookkeeping on the L x L matrices
// rot, cc, K) with an L-independent MPS part, or is there an O(L^3) left?
//
// The quenches are those of the examples: SIAM (U=0.1, V=0.1) from the doubly
// occupied impurity with empty buffers (example/fbr_dyn_siam.cpp) for
// spin_symmetric and spin_block, IRLM (U=0.2, V=0.1) from |10> for leading.
// At fixed t the physics (window, bond dimension) does not depend on L, so the
// time of step n isolates the L dependence. The steps below replay
// Fbr_dyn::iterate() through its public methods, in the same order, with a
// timer around each phase:
//   buildK   Common::build_K                      O(n_imp L^2)
//   plan     plan_representative x2, plan_active_representative, plan_natural_orbitals
//   applyK   apply_plan_to_K: the Givens gates on K      O(#gates L)
//   applyfb  Fb_mps::apply: gates on rot and cc, MPS gates in the window, Slater swaps
//   tdvp     full_hamiltonian + evolve_one
// After the last step it times the measurement solver.correlator(0,0) against an
// independent O(L^2) evaluation of the same element. Before 2026-09-15 the library
// went through the dense effective_rot, O(L^3): 166 s at L=4000 (app/README.md).
//
//   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build/app/bench_dyn_cost block 1000 30
//   layout: sym | block | ns
//   writes app/output/bench_dyn_cost_<layout>_L<L>_U<U>.dat

#include "fbr/fbr.h"

#include <armadillo>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace fbr;
using clk = std::chrono::steady_clock;

static double sec_since(clk::time_point t0)
{
    return std::chrono::duration<double>(clk::now() - t0).count();
}

/// SIAM in the example's convention: {buf_up, imp_up, imp_dw, buf_dw} = {2, 0, 1, 3}
static ImpurityParam siam(int L, double U, Layout layout)
{
    double V = 0.1;
    arma::mat K(L, L, arma::fill::zeros);
    for (int i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
    arma::mat Umat(L, L, arma::fill::zeros);
    Umat(0, 1) = U;
    return ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {2, 0, 1, 3}, .layout = layout};
}

/// spinless IRLM, as in example/fbr_dyn_irlm.cpp
static ImpurityParam irlm(int L, double U)
{
    arma::mat K(L, L, arma::fill::zeros);
    for (int i = 1; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    K(0, 1) = K(1, 0) = 0.1;
    K(0, 0) = K(1, 1) = -U / 2;
    arma::mat Umat(L, L, arma::fill::zeros);
    Umat(0, 1) = U;
    return ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1}};
}

/// <c_i^dag c_j> in O(L^2): only rows i and j of effective_rot are needed,
///   Q.row(i) = ((rot_star.row(i) % phase^T) * rot_star^dag) * fb.rot
static cmpx correlator_L2(Fbr_dyn const& s, int i, int j)
{
    arma::cx_rowvec d = s.ip_phase(s.n_iter).st();
    auto row = [&](int k) {
        arma::cx_rowvec r = s.rot_star.row(k) % d;
        r = r * s.rot_star.t();
        return arma::cx_rowvec(r * s.fb.rot);
    };
    arma::cx_rowvec qi = row(i), qj = row(j);
    arma::cx_vec ccqj = s.fb.cc * qj.st();
    return arma::cdot(qi.st(), ccqj);
}

int main(int argc, char** argv)
{
    std::string lay = argc > 1 ? argv[1] : "block";
    int L = argc > 2 ? std::stoi(argv[2]) : 500;
    int nstep = argc > 3 ? std::stoi(argv[3]) : 30;
    std::string Ustr = argc > 4 ? argv[4] : (lay == "ns" ? "0.2" : "0.1");
    double U = std::stod(Ustr);
    double dt = 0.1;

    auto t0 = clk::now();
    ImpurityParam model = lay == "ns"    ? irlm(L, U)
                          : lay == "sym" ? siam(L, U, spin_symmetric)
                                         : siam(L, U, spin_block);
    model.to_star();
    double t_star = sec_since(t0);

    auto ek = arma::vec{model.Kmat.diag()};
    if (lay == "ns") { ek[0] = -10; ek[1] = 10; }
    else {
        ek[L / 2 - 1] = ek[L / 2] = -10;
        ek[L / 2 - 2] = ek[L / 2 + 1] = 10;
    }
    auto fb = slater<cmpx>(model, ek);
    fb.tol = 1e-10;

    t0 = clk::now();
    auto solver = Fbr_dyn(model, fb, dt);
    double t_ctor = sec_since(t0);
    TdvpParam args{.epsilon_M = 0};

    // the real-space site of n0 (impurity up for the SIAM, site 0 for the IRLM)
    int i0 = 0;

    std::string fname = "app/output/bench_dyn_cost_" + lay + "_L" + std::to_string(L) + "_U" + Ustr + ".dat";
    std::ofstream out(fname);
    out << "# Fbr_dyn timestep cost, layout=" << lay << " L=" << L << " U=" << Ustr
        << " dt=" << dt << " tol=1e-10 epsilon_M=0, one thread\n"
        << "# setup: to_star " << t_star << " s, Fbr_dyn constructor " << t_ctor << " s\n"
        << "# times in seconds; total = buildK+plan+applyK+applyfb+tdvp\n"
        << "# t n_active chi ngates buildK plan applyK applyfb tdvp total\n"
        << std::setprecision(6);

    for (int n = 0; n < nstep; n++) {
        double tb = 0, tp = 0, tk = 0, tf = 0, tt = 0;
        size_t ngates = 0;
        auto step = [&](auto&& make_plan) {
            auto c = clk::now();
            OrbitalUpdate<cmpx> up = make_plan();
            tp += sec_since(c);
            ngates += up.gates.size();
            c = clk::now();
            solver.apply_plan_to_K(solver.fb, up);
            tk += sec_since(c);
            c = clk::now();
            solver.fb.apply(up);
            tf += sec_since(c);
        };

        // --- Fbr_dyn::iterate(), phase by phase ---
        auto c = clk::now();
        solver.K = solver.build_K();
        solver.n_iter++;
        tb = sec_since(c);
        step([&] { return solver.fb.plan_representative(solver.K, 0); });
        step([&] { return solver.fb.plan_representative(solver.K, 1); });
        step([&] { return solver.fb.plan_active_representative(solver.K); });
        c = clk::now();
        solver.do_tdvp(args);
        tt = sec_since(c);
        step([&] { return solver.fb.plan_natural_orbitals(solver.fb.cc); });
        // ---

        double total = tb + tp + tk + tf + tt;
        out << (n + 1) * dt << " " << solver.fb.n_active() << " " << itensor::maxLinkDim(solver.fb.psi)
            << " " << ngates << " " << tb << " " << tp << " " << tk << " " << tf << " " << tt << " " << total
            << std::endl;
    }

    // measurement: library correlator (through effective_rot) vs an O(L^2) evaluation
    auto c = clk::now();
    cmpx n0_lib = solver.correlator(i0, i0);
    double t_lib = sec_since(c);
    c = clk::now();
    cmpx n0_L2 = correlator_L2(solver, i0, i0);
    double t_L2 = sec_since(c);
    out << "# measure: correlator(0,0) " << t_lib << " s, O(L^2) version " << t_L2 << " s, n0="
        << std::setprecision(12) << std::real(n0_lib) << " |diff|=" << std::abs(n0_lib - n0_L2) << "\n";
    std::cout << fname << " done\n";
    return 0;
}
