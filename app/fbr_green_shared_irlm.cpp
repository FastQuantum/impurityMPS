// Green functions of the spinless IRLM the STAR way: the three states psi0,
// c_0^dag psi0, c_1^dag psi0 share one common orbital basis (Fbr_dyn_shared),
// whose active window has to hold the union of all three states' natural
// orbitals. Same model/grid/ground state as fbr_green_sep_irlm.cpp, so the two
// output files can be compared directly -- this is the "star" baseline the
// separate-frame method is measured against, on cost (n_active, bond dim).
//
// Usage: fbr_green_shared_irlm [U]       (0.1 or 0.2; default 0.2)
// Env:   GREEN_NSTEP (default 200)

#include "fbr/fbr_dyn.h"
#include "fbr/fbr_gs.h"
#include "../test/test_ref_common.h"

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

static cmpx cElement(Fb_mps<cmpx> const &A, Fb_mps<cmpx> const &B, int i)
{
    auto Ai = A; Ai.apply_local_op("Cdag", i);
    return itensor::innerC(Ai.psi, B.psi);
}

int main(int argc, char **argv)
{
    string us = argc > 1 ? argv[1] : "0.2";
    double U = std::stod(us);
    int L = envI("GREEN_L", 100);
    double V = 0.1, dt = 0.1;
    int nStep = envI("GREEN_NSTEP", 200);

    auto model = makeIrlmModel(L, U, V);

    // interacting ground state: off disk (L=100) or from the few-body Fbr_gs.
    std::vector<GreenSample> ref;
    if (L == 100)
        try { ref = loadGreenReference("chain_green_irlm_U" + us + "_ref.txt"); } catch (...) {}
    Fb_mps<cmpx> psi0;
    try {
        psi0 = loadFbMps<double>(findRef("fbr_green_gs_L" + to_string(L) + "_U" + us + ".dat"))
                   .to_complex();
    } catch (...) {
        auto gs = Fb_mps<double>::from_slater(model.rot, vec{model.Kmat.diag()},
                                              L / 2, model.n_imp(), leading);
        gs.tol = 1e-12;
        Fbr_gs gsSolver(model, gs);
        int nsweep = envI("GREEN_GSSWEEP", 80);
        for (int i = 0; i < nsweep; i++) gsSolver.iterate({.max_bond_dim = 256});
        cerr << "# Fbr_gs ground state at L=" << L << " U=" << us
             << ": energy=" << setprecision(10) << gsSolver.energy << "\n";
        psi0 = gsSolver.fb.to_complex();
    }
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

    auto solver = Fbr_dyn_shared(model, std::vector{psi0, B0, B1}, dt);

    string name = "fbr_green_shared_irlm_L" + to_string(L) + "_U" + us + ".dat";
    ostringstream rows;
    rows << setprecision(12);
    itensor::cpu_time clk;
    double devMax = 0;

    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;
        cmpx G00 = -imag_1 * nrm0 * cElement(solver.states[0], solver.states[1], 0);
        cmpx G01 = -imag_1 * nrm1 * cElement(solver.states[0], solver.states[2], 0);
        if (step < (int)ref.size())
            devMax = std::max(devMax, std::max(std::abs(G00 - ref[step].G00),
                                               std::abs(G01 - ref[step].G01)));
        int m = 0;
        for (auto const &s : solver.states) m = std::max(m, itensor::maxLinkDim(s.psi));
        int na = solver.states[0].n_active();

        rows << t << " " << G00.real() << " " << G00.imag() << " "
             << G01.real() << " " << G01.imag() << " " << m << " " << na << "\n";

        ofstream out("app/output/" + name);
        if (!out) out.open(name);
        out << "fbr_green_shared_irlm_v1 L " << L << " U " << U << " V " << V
            << " dt " << dt << " steps " << (step + 1) << "\n"
            << "# t ReG00 ImG00 ReG01 ImG01 maxBondDim n_active   (vs chain: max|dG|="
            << devMax << ")\n" << rows.str();
        out.close();

        if (step % 5 == 0)
            cerr << "# t=" << t << " m=" << m << " n_active=" << na
                 << " max|dG|=" << devMax << "  " << clk.sincemark().wall << " s\n";
        if (step < nStep) solver.iterate({.epsilon_M = 0});
    }
    cerr << "# wrote " << name << " ; max deviation from chain reference = " << devMax << "\n";
    return 0;
}
