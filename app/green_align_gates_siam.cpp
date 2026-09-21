// Cost of the frame ALIGNMENT in the separate-frame SIAM Green function.
//
// The impurity greater Green function is
//     G00(t) = -i <psi0| c_0(t) c_0^dag |psi0> = -i <c_0^dag psi0(t) | B(t)>,
// with A = |psi0> and B = c_0^dag|psi0>. Here A and B are evolved by their OWN
// Fbr_dyn, each keeping its own small active window; the two frames are brought
// into a common basis ONLY at the measurement, by green_overlap.h::c_element
// (align_to_frame + a plain MPS contraction).
//
// That alignment is a nearest-neighbour Givens circuit on the band where the two
// frames differ. The Green function it feeds is one scalar per step, wanted only
// to ~3 digits, so the ROTATED MPS the circuit builds is a throwaway: it can be
// truncated much harder than the states themselves. This program sweeps that
// truncation (mps_cutoff) and records, per step, the cost and accuracy at each.
//
// Measurement-cutoff-independent columns (written once):
//   t             time
//   gates         two-site Givens gates actually applied (depends on GREEN_CUTOFF)
//   gates_cand    candidate gates before the near-identity skip (givens_align_left)
//   band          width of the aligned band  b-a  (mismatch_band)
//   phases        single-site phase clean-ups applied
//   full          1 if the whole-chain fallback triggered that step, else 0
//   evolveA_s     wall seconds of one TDVP sweep of A=psi0   (time per sweep)
//   evolveB_s     wall seconds of one TDVP sweep of B=c_0^dag psi0
//   bondA bondB   max MPS bond dimension of A and B (the states -- these SATURATE)
//   naA naB       active-window size of A and B
// Then, for each mps_cutoff value j in the swept list (header: mps_cutoffs):
//   chi<j>        max bond dim of the rotated throwaway MPS at that truncation
//   alignS<j>     wall seconds of that alignment+contraction
//   ReG<j> ImG<j> the Green function at that truncation
//
// The point: the state bonds (bondA/bondB) saturate, so evolution is cheap and
// bounded; the alignment's chi<j> is what can run away, and a loose mps_cutoff
// keeps it -- and the cost -- bounded while G still agrees to ~3 digits across the
// swept cutoffs.
//
// Usage: green_align_gates_siam [L] [tmax] [U] [dt]
//   defaults: L=200, tmax=L/2, U=0.1, dt=0.1
// Env: GREEN_CUTOFF       Givens-skip / band threshold             (default 1e-4)
//      GREEN_MPS_CUTOFFS  comma list of throwaway-MPS truncations  (default 1e-6;
//                         pass e.g. 1e-4,1e-5,1e-6 to sweep cost vs accuracy)
//      GREEN_TOL          state MPS/circuit cutoff fb.tol          (default 1e-8)
//      GREEN_GSSWEEP      ground-state DMRG sweeps                 (default 80)
//
// GREEN_TOL governs how far the excitation's OWN bond dimension grows. At the
// tuned production default (1e-8) it stays off the numerical-noise floor; a
// much tighter value (1e-12) keeps noise-level singular values and inflates it.

#include "fbr/fbr_dyn.h"
#include "fbr/fbr_gs.h"
#include "fbr/green_overlap.h"
#include "../test/test_ref_common.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {
int    envI(char const *k, int d)    { return getenv(k) ? std::stoi(getenv(k)) : d; }
double envD(char const *k, double d) { return getenv(k) ? std::stod(getenv(k)) : d; }
double seconds(std::function<void()> f)
{
    auto t0 = std::chrono::steady_clock::now(); f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
// comma-separated list of doubles from env, or `def` (also comma-separated) if unset.
std::vector<double> envList(char const *k, std::string def)
{
    std::string s = getenv(k) ? getenv(k) : def;
    std::vector<double> out;
    std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stod(tok));
    return out;
}
} // namespace

int main(int argc, char **argv)
{
    int    L    = argc > 1 ? std::stoi(argv[1]) : 200;
    double tmax = argc > 2 ? std::stod(argv[2]) : L / 2.0;
    string us   = argc > 3 ? argv[3] : "0.1";
    double U    = std::stod(us);
    double dt   = argc > 4 ? std::stod(argv[4]) : 0.1;
    double V    = 0.1;
    double cutoff = envD("GREEN_CUTOFF", 1e-4);     // Givens-skip / band threshold
    double tol    = envD("GREEN_TOL", 1e-8);        // state MPS/circuit cutoff (fb.tol)
    // The rotated throwaway-MPS truncation. Default 1e-6: the tuned production
    // value, loose enough that chi_align (and the alignment time) saturate while G
    // keeps ~3 digits. Pass a comma list to sweep several at once (each step G is
    // measured at every value), recording cost and accuracy together on one run.
    std::vector<double> mpsCuts = envList("GREEN_MPS_CUTOFFS", "1e-6");
    int    nStep  = (int)std::llround(tmax / dt);

    auto model = fbrtest::makeSiamModel(L, U, V);   // spin_sym geometry, star frame

    // ---- ground state (spin_sym, the cheap symmetric geometry) --------------
    auto gs = slater<double>(model);
    gs.tol = tol;
    Fbr_gs gsSolver(model, gs);
    itensor::cpu_time clk;
    int nsweep = envI("GREEN_GSSWEEP", 80);
    for (int i = 0; i < nsweep; i++) gsSolver.iterate({.max_bond_dim = 512});
    cerr << setprecision(12)
         << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt << " tmax=" << tmax
         << " cutoff=" << cutoff << "\n"
         << "# ground state: energy=" << gsSolver.energy
         << " n_active=" << gsSolver.fb.n_active()
         << " in " << clk.sincemark().wall << " s\n";

    // ---- two states in SEPARATE frames --------------------------------------
    // The excitation B = c_0^dag|psi0> breaks spin-flip symmetry, so the dynamics
    // runs under spin_block (see app/fbr_green_siam.cpp). Both states share the
    // same star frame (same model), so their bath phases cancel in the overlap.
    model.geometry = spin_block;
    auto psi0 = gsSolver.fb.to_complex();
    psi0.geometry = spin_block;
    psi0.tol = tol;
    auto B = psi0;
    B.apply_local_op("Cdag", 0);                    // c_0^dag on impurity site 0 (up)
    double nrm = std::sqrt(std::real(itensor::innerC(B.psi, B.psi)));
    B.psi.normalize();
    B.update_cc();
    B.tol = tol;

    // two INDEPENDENT solvers, stepped in lockstep (shared star frame, dt)
    Fbr_dyn dA(model, psi0, dt), dB(model, B, dt);

    // impurity site 0 must be a single MPS orbital for c_element to be local
    {
        arma::cx_mat Q = dA.effective_rot();
        arma::rowvec row = arma::abs(Q.row(0));
        double mx = row.max();
        if (std::abs(mx - 1) > 1e-10
            || std::sqrt(arma::accu(arma::square(row)) - mx * mx) > 1e-10)
            throw std::runtime_error("impurity site 0 is not a single MPS orbital");
    }

    // ---- output -------------------------------------------------------------
    // Per-cutoff columns: chi<j> alignS<j> ReG<j> ImG<j> for j over mpsCuts (the
    // header names them and lists the values in mps_cutoffs). The measurement-cutoff
    // -independent quantities (gates, band, evolution, state bonds) are written once.
    string tag  = getenv("GREEN_TAG") ? getenv("GREEN_TAG") : "";
    string name = "green_align_gates_siam_L" + to_string(L) + "_U" + us + tag + ".dat";
    int nc = (int)mpsCuts.size();
    ostringstream cutlist, colhdr;
    cutlist << setprecision(3);
    for (int j = 0; j < nc; ++j) cutlist << (j ? "," : "") << mpsCuts[j];
    for (int j = 0; j < nc; ++j)
        colhdr << " chi" << j << " alignS" << j << " ReG" << j << " ImG" << j;

    ostringstream rows;
    rows << setprecision(8);
    itensor::cpu_time wallclk;

    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;

        // measurement at every mps_cutoff (each on its own copy of B, so evolution
        // is untouched); gates/band are the same for all, recorded from the last.
        AlignStats st;
        std::vector<int> chi(nc);
        std::vector<double> as(nc);
        std::vector<cmpx> G(nc);
        for (int j = 0; j < nc; ++j) {
            AlignStats s;
            as[j] = seconds([&] {
                G[j] = -imag_1 * nrm * c_element(dA.fb, dB.fb, 0, cutoff, false, mpsCuts[j], &s);
            });
            chi[j] = s.bond;
            st = s;                                  // gates/band/phases identical across j
        }

        int bondA = itensor::maxLinkDim(dA.fb.psi), bondB = itensor::maxLinkDim(dB.fb.psi);
        int naA = dA.fb.n_active(), naB = dB.fb.n_active();

        // one TDVP sweep of each state (time per sweep)
        double evolveA_s = 0, evolveB_s = 0;
        if (step < nStep) {
            evolveA_s = seconds([&] { dA.iterate({.epsilon_M = 0}); });
            evolveB_s = seconds([&] { dB.iterate({.epsilon_M = 0}); });
        }

        rows << t
             << " " << st.gates << " " << st.gates_candidate << " " << st.band
             << " " << st.phases << " " << (st.full ? 1 : 0)
             << " " << evolveA_s << " " << evolveB_s
             << " " << bondA << " " << bondB << " " << naA << " " << naB;
        for (int j = 0; j < nc; ++j)
            rows << " " << chi[j] << " " << as[j] << " " << G[j].real() << " " << G[j].imag();
        rows << "\n";

        // rewrite the whole file each step: the run is long, keep it interruptible
        ofstream out("app/output/" + name);
        if (!out) out.open(name);
        out << "green_align_gates_siam_v2 L " << L << " U " << U << " V " << V
            << " dt " << dt << " cutoff " << cutoff << " tol " << tol
            << " mps_cutoffs " << cutlist.str() << " steps " << (step + 1) << "\n"
            << "# t gates gates_cand band phases full evolveA_s evolveB_s "
               "bondA bondB naA naB" << colhdr.str() << "\n"
            << rows.str();
        out.close();

        if (step % 10 == 0) {
            cerr << "# t=" << t << " gates=" << st.gates << " band=" << st.band
                 << " evolveB=" << evolveB_s << "s  chi_align/align_s:";
            for (int j = 0; j < nc; ++j)
                cerr << " [" << mpsCuts[j] << ": " << chi[j] << "/" << as[j] << "s]";
            cerr << "  [" << wallclk.sincemark().wall << "s]\n";
        }
    }
    cerr << "# wrote app/output/" << name << "\n";
    return 0;
}
