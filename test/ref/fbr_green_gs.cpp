// The FBR ground states the Green function tests start from, saved once.
//
// Computing them is the slow part of test_ref_green.cpp -- 80 Fbr_gs sweeps at
// L=100, per model and U -- and the result never changes, so it is kept on disk.
// The test still checks what it loads: at t=0 the Green functions involve only
// the ground state, and they are compared there against the chain DMRG baseline
// to 1e-6. Fbr_gs itself stays covered by the [frame] test in test_ref_ns.cpp,
// which runs it at L=12 for pennies.
//
// Writes, for U = 0.1 and 0.2,
//     irlm: fbr_green_gs_L<L>_U<U>.dat        (makeIrlmModel)
//     siam: fbr_green_gs_siam_L<L>_U<U>.dat   (makeSiamModel, spin-symmetric)
// into the working directory. Fbr_gs is not bit-reproducible, so regenerate one
// model only, and only after changing that model or Fbr_gs.
//
// Usage: fbr_green_gs irlm|siam

#include "fbr/fbr_gs.h"
#include "../test_ref_common.h"

#include <iomanip>
#include <iostream>

using namespace std;
using namespace fbr;
using namespace fbrtest;

int main(int argc, char **argv)
{
    string which = argc > 1 ? argv[1] : "";
    if (which != "irlm" && which != "siam") {
        cerr << "usage: fbr_green_gs irlm|siam\n";
        return 1;
    }
    constexpr int L = 100;
    constexpr double V = 0.1;
    constexpr int n_iter = 80;

    for (auto us : {string("0.1"), string("0.2")}) {
        double U = std::stod(us);
        auto model = which == "irlm" ? makeIrlmModel(L, U, V) : makeSiamModel(L, U, V);

        itensor::cpu_time clk;
        auto gs = slater<double>(model);
        gs.tol = 1e-12;
        auto solver = Fbr_gs(model, gs);
        for (int i = 0; i < n_iter; i++) solver.iterate({.max_bond_dim = 512});

        string name = string("fbr_green_gs_") + (which == "siam" ? "siam_" : "")
                    + "L" + to_string(L) + "_U" + us + ".dat";
        saveFbMps(name, solver.fb);
        cout << setprecision(12)
             << "# wrote " << name << "  energy=" << solver.energy
             << " n_active=" << solver.fb.n_active()
             << " in " << clk.sincemark().wall << " s" << endl;
    }
    return 0;
}
