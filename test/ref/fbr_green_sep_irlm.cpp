// Green functions of the spinless IRLM computed the SEPARATE-frame way: psi0,
// c_0^dag psi0 and c_1^dag psi0 are each evolved by their own Fbr_dyn (own small
// active window), and brought into a common basis only at the measurement, by
// green_overlap.h. Same model and grid as test_ref_green.cpp / chain_green_irlm.cpp
// (L=100, V=0.1, dt=0.1, interacting ground state off disk), so the output is
// directly comparable to test/ref/output/chain_green_irlm_U<U>_ref.txt, against
// which the max deviation is reported as the run proceeds.
//
// Usage: fbr_green_sep_irlm [U]          (0.1 or 0.2; default 0.2)
// Env:   GREEN_NSTEP (default 200), GREEN_CUTOFF (measurement cutoff, default 1e-4)

#include "fbr/fbr_dyn.h"
#include "fbr/fbr_gs.h"
#include "fbr/green_overlap.h"
#include "../test_ref_common.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;
using namespace arma;
using namespace fbr;
using namespace fbrtest;

static int envI(char const *k, int fallback)
{ char const *v = std::getenv(k); return v ? std::atoi(v) : fallback; }
static double envD(char const *k, double fallback)
{ char const *v = std::getenv(k); return v ? std::atof(v) : fallback; }

int main(int argc, char **argv)
{
    string us = argc > 1 ? argv[1] : "0.2";
    double U = std::stod(us);
    int L = 100, n_part = L / 2;
    double V = 0.1, dt = 0.1;
    int nStep = envI("GREEN_NSTEP", 200);
    double cutoff = envD("GREEN_CUTOFF", 1e-4);

    auto model = makeIrlmModel(L, U, V);
    auto ref = loadGreenReference("chain_green_irlm_U" + us + "_ref.txt");

    // interacting ground state off disk (the slow part; identical between runs)
    auto psi0 = loadFbMps<double>(findRef("fbr_green_gs_L" + to_string(L) + "_U" + us + ".dat"))
                    .to_complex();
    psi0.tol = 1e-12;
    auto addParticle = [](Fb_mps<cmpx> const &p, int j) {
        auto s = p; s.apply_local_op("Cdag", j);
        double nrm = std::sqrt(std::real(itensor::innerC(s.psi, s.psi)));
        s.psi.normalize(); s.update_cc();
        return std::make_pair(s, nrm);
    };
    auto [B0, nrm0] = addParticle(psi0, 0);
    auto [B1, nrm1] = addParticle(psi0, 1);
    B0.tol = B1.tol = 1e-12;

    // three INDEPENDENT solvers, stepped in lockstep (shared star frame, n_iter)
    Fbr_dyn dPsi(model, psi0, dt), dB0(model, B0, dt), dB1(model, B1, dt);

    string name = "fbr_green_sep_irlm_U" + us + ".txt";
    ostringstream rows;
    rows << setprecision(12);
    itensor::cpu_time clk;
    double devMax = 0;

    for (int step = 0; step <= nStep && step < (int)ref.size(); step++) {
        double t = step * dt;
        cmpx G00 = -imag_1 * nrm0 * c_element(dPsi.fb, dB0.fb, 0, cutoff);
        cmpx G01 = -imag_1 * nrm1 * c_element(dPsi.fb, dB1.fb, 0, cutoff);
        double dev = std::max(std::abs(G00 - ref[step].G00), std::abs(G01 - ref[step].G01));
        devMax = std::max(devMax, dev);
        int m = std::max({itensor::maxLinkDim(dPsi.fb.psi),
                          itensor::maxLinkDim(dB0.fb.psi), itensor::maxLinkDim(dB1.fb.psi)});
        int na = std::max({dPsi.fb.n_active(), dB0.fb.n_active(), dB1.fb.n_active()});

        rows << t << " " << G00.real() << " " << G00.imag() << " "
             << G01.real() << " " << G01.imag() << " " << m << " " << na << "\n";

        ofstream out("test/ref/output/" + name);  // rewrite each step: interruptible
        if (!out) out.open(name);                  // fallback: current directory
        out << "fbr_green_sep_irlm_v1 L " << L << " U " << U << " V " << V
            << " dt " << dt << " cutoff " << cutoff << " steps " << (step + 1) << "\n"
            << "# t ReG00 ImG00 ReG01 ImG01 maxBondDim n_active   (vs chain: max|dG|="
            << devMax << ")\n" << rows.str();
        out.close();

        if (step % 10 == 0)
            cerr << "# t=" << t << " m=" << m << " n_active=" << na
                 << " max|dG|=" << devMax << "  " << clk.sincemark().wall << " s\n";
        if (step < nStep) { dPsi.iterate({.epsilon_M = 0}); dB0.iterate({.epsilon_M = 0});
                            dB1.iterate({.epsilon_M = 0}); }
    }
    cerr << "# wrote " << name << " ; max deviation from chain reference = " << devMax << "\n";
    return 0;
}
