// Cost benchmark for the SIAM (spin), analogous to bench_green_cost.cpp:
// few-body separate-frame Green function (small active window) vs the full STAR
// method (whole chain of L star orbitals active). Same interacting ground state
// for both (few-body: Fbr_gs; star: DMRG on the star Hamiltonian), all three
// states psi0, c_up^dag psi0, c_dw^dag psi0 evolved, evolved to t = L/2.
//
// Model: spin_symmetric SIAM of example/fbr_gs_siam.cpp -- two interleaved spin
// chains (hopping 0.5 at distance 2), impurity sites 0 (up) and 1 (dw),
// hybridization 0.5, Hubbard U between them.
//
// Usage: bench_green_cost_siam [U]         (default 0.025)
// Env:   GREEN_L (default 40); total time is L/2 (nStep = 5*L).

#include "fbr/fbr_dyn.h"      // also pulls in tdvp.h and basisextension.h
#include "fbr/fbr_gs.h"
#include "fbr/green_overlap.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {
int envI(char const *k, int d) { return getenv(k) ? std::stoi(getenv(k)) : d; }
double seconds(std::function<void()> f)
{
    auto t0 = std::chrono::steady_clock::now(); f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
// full-L star Hamiltonian from the (dense) star K and interaction Umat.
itensor::MPO starHamiltonian(itensor::Fermion const &sites, mat const &K, mat const &Umat)
{
    int L = K.n_rows;
    itensor::AutoMPO h(sites);
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++) {
            if (std::abs(Umat(i, j)) > 1e-15) h += Umat(i, j), "N", i + 1, "N", j + 1;
            if (std::abs(K(i, j)) > 1e-12)    h += K(i, j), "Cdag", i + 1, "C", j + 1;
        }
    return itensor::toMPO(h);
}
void applyCdag(itensor::Fermion const &sites, itensor::MPS &psi, int site)  // 0-based
{
    for (int k = 1; k <= site; k++) { auto A = sites.op("F", k) * psi(k); A.noPrime(); psi.set(k, A); }
    psi.position(site + 1);
    auto A = sites.op("Cdag", site + 1) * psi(site + 1); A.noPrime(); psi.set(site + 1, A);
}
// star Green-function measurement: <A| c_site |B> = <c_site^dag A | B>, a plain
// innerC in the common star basis (the cheap baseline the few-body overlap must beat).
cmpx starCElement(itensor::Fermion const &sites, itensor::MPS const &A,
                  itensor::MPS const &B, int site)
{
    auto Ai = A; applyCdag(sites, Ai, site);
    return itensor::innerC(Ai, B);
}
void starTdvp(itensor::MPS &psi, itensor::MPO const &mpo, double dt)
{
    auto sw = itensor::Sweeps(1);
    sw.maxdim() = 2048; sw.cutoff() = 1e-9; sw.niter() = 16; sw.noise() = 0;
    std::vector<double> eK(2, 1e-4);
    itensor::addBasis(psi, mpo, eK, {"Cutoff", 1e-8, "Method", "DensityMatrix",
                                     "KrylovOrd", 2, "DoNormalize", true, "Quiet", true, "Silent", true});
    itensor::tdvp(psi, mpo, -imag_1 * dt, sw, {"Truncate", true, "DoNormalize", true, "Quiet", true,
                  "Silent", true, "NumCenter", 2, "ErrGoal", 1e-8});
}
} // namespace

int main(int argc, char **argv)
{
    string us = argc > 1 ? argv[1] : "0.025";
    double U = std::stod(us);
    int L = envI("GREEN_L", 40), n_part = L / 2;
    double dt = 0.1;
    int nStep = envI("GREEN_NSTEP", 5 * L);           // total time = L/2

    mat K(L, L, fill::zeros);
    for (int i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = 0.5;      // impurity-bath hybridization
    mat Umat(L, L, fill::zeros); Umat(0, 1) = U;
    // spin_block (generic spin): the Green-function excitation c^dag_{imp,up} breaks
    // spin symmetry, so spin_symmetric (which only evolves one sector and mirrors it)
    // is invalid here.
    auto model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1}, .layout = spin_block};
    model.to_star();
    mat Kstar = model.Kmat, Ustar = model.Umat;
    int impUp = model.imp_pos[0], impDw = model.imp_pos[1];   // star positions of imp_up, imp_dw

    // ---- few-body ground state (Fbr_gs) ----
    auto gs = slater<double>(model);   gs.tol = 1e-9;
    Fbr_gs gsSolver(model, gs);
    for (int i = 0; i < 80; i++) gsSolver.iterate({.max_bond_dim = 256});
    cerr << "# few-body GS energy = " << setprecision(10) << gsSolver.energy
         << "  n_active=" << gsSolver.fb.n_active() << "\n";
    auto addFb = [](Fb_mps<cmpx> const &p, int j) {
        auto s = p; s.apply_local_op("Cdag", j); s.psi.normalize(); s.update_cc(); return s; };
    auto psi0 = gsSolver.fb.to_complex(); psi0.tol = 1e-9;
    auto fbB0 = addFb(psi0, 0), fbB1 = addFb(psi0, 1); fbB0.tol = fbB1.tol = 1e-9;
    Fbr_dyn dPsi(model, psi0, dt), dB0(model, fbB0, dt), dB1(model, fbB1, dt);

    // ---- full-L star ground state (DMRG) ----
    auto sites = itensor::Fermion(L, {"ConserveNf", true});
    auto mpo = starHamiltonian(sites, Kstar, Ustar);
    itensor::MPS sPsi;
    {
        auto st = itensor::InitState(sites, "0");
        uvec iek = sort_index(vec{Kstar.diag()});
        for (int j = 0; j < n_part; j++) st.set((int)iek[j] + 1, "1");
        sPsi = itensor::MPS(st);
        auto sw = itensor::Sweeps(1); sw.maxdim() = 2048; sw.cutoff() = 1e-9; sw.niter() = 4; sw.noise() = 1e-7;
        double e = 0;
        for (int i = 0; i < 60; i++) e = itensor::dmrg(sPsi, mpo, sw, {"Quiet", true, "Silent", true});
        cerr << "# star DMRG GS energy   = " << setprecision(10) << e << "\n";
    }
    auto sA = sPsi; sA *= cmpx(1, 0);
    auto sB0 = sPsi; applyCdag(sites, sB0, impUp); sB0.normalize(); sB0 *= cmpx(1, 0);
    auto sB1 = sPsi; applyCdag(sites, sB1, impDw); sB1.normalize(); sB1 *= cmpx(1, 0);

    double cutoff = getenv("GREEN_CUTOFF") ? std::stod(getenv("GREEN_CUTOFF")) : 1e-4;

    string name = "bench_green_cost_siam_L" + to_string(L) + "_U" + us + ".dat";
    ofstream out("app/output/" + name); if (!out) out.open(name);
    out << "bench_green_cost_siam_v2 L " << L << " U " << U << " dt " << dt
        << " nStep " << nStep << " cutoff " << cutoff << "\n"
        << "# t  fb_evolve_s fb_measure_s fb_nactive fb_bond   star_evolve_s star_measure_s star_bond   |dG|\n"
        << setprecision(6);
    cerr << fixed << setprecision(3);
    itensor::cpu_time clk;
    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;
        // --- Green-function computation (the point): few-body overlap vs star innerC ---
        cmpx g00fb, g01fb, g00st, g01st;
        double fb_meas = seconds([&] { g00fb = c_element(dPsi.fb, dB0.fb, 0, cutoff);
                                       g01fb = c_element(dPsi.fb, dB1.fb, 0, cutoff); });
        double st_meas = seconds([&] { g00st = starCElement(sites, sA, sB0, impUp);
                                       g01st = starCElement(sites, sA, sB1, impUp); });
        double dG = std::max(std::abs(g00fb - g00st), std::abs(g01fb - g01st));
        // --- one evolution step of all three states ---
        double fb_ev = seconds([&] { dPsi.iterate({.epsilon_M = 0}); dB0.iterate({.epsilon_M = 0});
                                     dB1.iterate({.epsilon_M = 0}); });
        double st_ev = seconds([&] { starTdvp(sA, mpo, dt); starTdvp(sB0, mpo, dt); starTdvp(sB1, mpo, dt); });
        int fb_na = std::max({dPsi.fb.n_active(), dB0.fb.n_active(), dB1.fb.n_active()});
        int fb_m  = std::max({itensor::maxLinkDim(dPsi.fb.psi), itensor::maxLinkDim(dB0.fb.psi),
                              itensor::maxLinkDim(dB1.fb.psi)});
        int st_m  = std::max({itensor::maxLinkDim(sA), itensor::maxLinkDim(sB0), itensor::maxLinkDim(sB1)});
        out << t << "  " << fb_ev << " " << fb_meas << " " << fb_na << " " << fb_m
            << "   " << st_ev << " " << st_meas << " " << st_m << "   " << dG << "\n";
        out.flush();
        if (step % 5 == 0) {
            auto bnd = fbr::mismatch_band(dPsi.fb, dB0.fb, cutoff);
            cerr << "  [band width=" << (bnd.b - bnd.a) << " of L=" << L << "]\n";
        }
        if (step % 5 == 0)
            cerr << "# L=" << L << " t=" << t << "/" << (L / 2)
                 << "  few-body: evolve " << fb_ev << "s + overlap " << fb_meas << "s (na=" << fb_na
                 << ",m=" << fb_m << ")   star: evolve " << st_ev << "s + innerC " << st_meas
                 << "s (m=" << st_m << ")   |dG|=" << dG << "  [" << clk.sincemark().wall << "s]\n";
    }
    cerr << "# wrote " << name << "\n";
    return 0;
}
