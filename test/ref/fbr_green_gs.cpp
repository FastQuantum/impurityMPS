// The FBR ground states the Green function tests start from, saved once.
//
// Computing them is the slow part of test_ref_green.cpp -- 80 Fbr_gs sweeps at
// L=100, twice -- and the result never changes, so it is kept on disk instead.
// The test still checks what it loads: at t=0 the Green functions involve only
// the ground state, and they are compared there against the chain DMRG baseline
// to 1e-6. Fbr_gs itself stays covered by the [frame] test in test_ref_ns.cpp,
// which runs it at L=12 for pennies.
//
// Writes output/fbr_green_gs_L<L>_U<U>.dat for U = 0.1 and 0.2.
//
// Usage: fbr_green_gs        (regenerate after changing the model or Fbr_gs)

#include "fbr/fbr_gs.h"
#include "../test_ref_common.h"

#include <iomanip>
#include <iostream>

using namespace std;
using namespace fbr;
using namespace fbrtest;

int main()
{
    constexpr int L = 100;
    constexpr double V = 0.1;
    constexpr int nIter = 80;

    for (auto us : {string("0.1"), string("0.2")}) {
        double U = std::stod(us);
        auto model = makeIrlmModel(L, U, V);

        itensor::cpu_time clk;
        auto gs = slater<double>(model);
        gs.tol = 1e-12;
        auto solver = Fbr_gs(model, gs);
        for (int i = 0; i < nIter; i++) solver.iterate({.max_bond_dim = 512});

        string name = "fbr_green_gs_L" + to_string(L) + "_U" + us + ".dat";
        saveFbMps(name, solver.fb);
        cout << setprecision(12)
             << "# wrote " << name << "  energy=" << solver.energy
             << " nActive=" << solver.fb.nActive()
             << " in " << clk.sincemark().wall << " s" << endl;
    }
    return 0;
}
