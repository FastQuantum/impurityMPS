// Parameter-tuning driver for the FBR (active-window) SIAM dynamics, mirroring
// test/test_ref_fbr.cpp's makeFbrRun. FBR runs with epsilonM=0 (no subspace
// expansion), so nKrylov/epsilonK are inert here — the only TDVP knob is err_goal
// (plus nIter_diag). Compares the FBR trajectory against the committed chain
// reference and prints per-snapshot max|dni|, max|dcc|.
//
// Usage: fbr_dyn_tune <U> <err_goal> <nIter_diag> <maxSteps>
#include "fbr/fbr_dyn.h"
#include "../../test/test_ref_common.h"

#include <iostream>
#include <iomanip>
#include <string>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

// star-geometry spin layout -> chain index map (copied from test_ref_fbr.cpp)
static uvec fbrIndexToChainIndex(int L)
{
    uvec p(L);
    p[0] = L / 2 - 1; p[1] = L / 2; p[2] = L / 2 - 2; p[3] = L / 2 + 1;
    for (int j = 0; j < L / 2 - 2; j++) p[2 * (j + 2)] = j;
    for (int j = 0; j < L / 2 - 2; j++) p[2 * (j + 2) + 1] = L / 2 + 2 + j;
    return p;
}

static auto makeFbrRun(int L, double dt, double U)
{
    Impurity model;
    {
        double V = 0.1;
        mat K(L, L, fill::zeros);
        for (auto i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
        K(0, 0) = -U / 2; K(1, 1) = -U / 2;
        K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
        mat Umat(L, L, fill::zeros);
        Umat(0, 1) = U;
        model = Impurity{{.Kmat = K, .Umat = Umat, .impPos = {2, 0, 1, 3}, .layout=spin_symmetric}};
    }
    auto ek = vec{model.param.Kmat.diag()};
    ek[L / 2 - 1] = ek[L / 2] = -10;
    ek[L / 2 - 2] = ek[L / 2 + 1] = 10;
    auto fb = model.slater<cmpx>(ek);
    auto solver = Fbr_dyn(model, fb, dt);
    solver.fb.tol = 1e-12;
    return solver;
}

int main(int argc, char** argv)
{
    int L = 100;
    double dt = 0.1;
    double U       = argc > 1 ? std::stod(argv[1]) : 0.2;
    double errGoal = argc > 2 ? std::stod(argv[2]) : 1e-8;
    int nIterDiag  = argc > 3 ? std::stoi(argv[3]) : 8;
    int maxSteps   = argc > 4 ? std::stoi(argv[4]) : 50;

    std::string us = argc > 1 ? argv[1] : "0.2";
    auto fbr = makeFbrRun(L, dt, U);
    auto p = fbrIndexToChainIndex(L);
    auto ref = loadReference("chain_dyn_siam_center_U" + us + "_ref.txt");

    std::cerr << "# FBR U=" << U << " errGoal=" << errGoal
              << " nIterDiag=" << nIterDiag << " (epsilonM=0, nKrylov inert)\n";
    std::cout << std::setprecision(3) << std::scientific;

    auto report = [&](int step, std::string const& label) {
        cx_mat cc = toChainOrder(fbr.correlator_all(), p);
        auto m = compare(cc, ref.at(label));
        std::cout << "  " << std::setw(8) << label
                  << "  dni=" << m.niMax << "  dcc=" << m.ccMax << "\n";
    };

    std::vector<std::pair<int,std::string>> wanted = {
        {0,"initial"},{1,"t=0.1"},{50,"t=5.0"},{100,"t=10.0"},{200,"t=20.0"}};

    if (ref.count("initial")) report(0, "initial");
    itensor::cpu_time t0;
    for (int step = 1; step <= maxSteps; step++) {
        fbr.iterate({.nIter_diag = nIterDiag, .err_goal = errGoal, .epsilonM = 0e-8});
        for (auto const& w : wanted)
            if (w.first == step && ref.count(w.second)) report(step, w.second);
    }
    std::cerr << "# wall " << t0.sincemark().wall << "s\n";
    return 0;
}
