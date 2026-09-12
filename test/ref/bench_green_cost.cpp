// Cost benchmark: the FEW-BODY separate-frame Green function (small active
// window) against the full STAR method (the whole chain of L star orbitals is
// active, n_active = L). Both evolve the same three states psi0, c_0^dag psi0,
// c_1^dag psi0 from the same interacting ground state (few-body: Fbr_gs; star:
// DMRG), with the same model, and we record the WALL TIME PER SWEEP and the bond
// dimension of each, plus the few-body active-window size.
//
// Usage: bench_green_cost [U]        (default 0.2)
// Env:   GREEN_L (default 60), GREEN_NSTEP (default 50)

#include "fbr/fbr_dyn.h"      // also pulls in tdvp.h and basisextension.h
#include "fbr/fbr_gs.h"
#include "fbr/green_overlap.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {
int    envI(char const *k, int d)    { return getenv(k) ? std::stoi(getenv(k)) : d; }

double seconds(std::function<void()> f)
{
    auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---- full-L star helpers (raw ITensor on the star Hamiltonian) ----
itensor::MPO starHamiltonian(itensor::Fermion const &sites, mat const &K, double U)
{
    int L = K.n_rows;
    itensor::AutoMPO h(sites);
    if (std::abs(U) > 1e-15) h += U, "N", 1, "N", 2;    // impurity at star sites 0,1
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (std::abs(K(i, j)) > 1e-12) h += K(i, j), "Cdag", i + 1, "C", j + 1;
    return itensor::toMPO(h);
}
void applyCdag(itensor::Fermion const &sites, itensor::MPS &psi, int site)
{
    for (int k = 1; k <= site; k++) { auto A = sites.op("F", k) * psi(k); A.noPrime(); psi.set(k, A); }
    psi.position(site + 1);
    auto A = sites.op("Cdag", site + 1) * psi(site + 1); A.noPrime(); psi.set(site + 1, A);
}
void starTdvp(itensor::MPS &psi, itensor::MPO const &mpo, double dt)
{
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024; sweeps.cutoff() = 1e-10; sweeps.niter() = 16; sweeps.noise() = 0;
    std::vector<double> eK(2, 1e-4);
    itensor::addBasis(psi, mpo, eK, {"Cutoff", 1e-8, "Method", "DensityMatrix",
                                     "KrylovOrd", 2, "DoNormalize", true, "Quiet", true, "Silent", true});
    itensor::tdvp(psi, mpo, -imag_1 * dt, sweeps,
                  {"Truncate", true, "DoNormalize", true, "Quiet", true, "Silent", true,
                   "NumCenter", 2, "ErrGoal", 1e-8});
}
} // namespace

int main(int argc, char **argv)
{
    string us = argc > 1 ? argv[1] : "0.2";
    double U = std::stod(us);
    int L = envI("GREEN_L", 60), nStep = envI("GREEN_NSTEP", 50), n_part = L / 2;
    double V = 0.1, dt = 0.1;

    mat K(L, L, fill::zeros);
    for (int i = 1; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    K(0, 1) = K(1, 0) = V; K(0, 0) = K(1, 1) = -U / 2;
    mat Umat(L, L, fill::zeros); Umat(0, 1) = U;
    auto model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1}};
    model.to_star();
    mat Kstar = model.Kmat;

    // ---- few-body ground state (Fbr_gs) ----
    auto gs = Fb_mps<double>::from_slater(model.rot, vec{Kstar.diag()}, n_part, model.n_imp(), leading);
    gs.tol = 1e-9;
    Fbr_gs gsSolver(model, gs);
    for (int i = 0; i < 60; i++) gsSolver.iterate({.max_bond_dim = 256});
    cerr << "# few-body GS energy = " << setprecision(10) << gsSolver.energy << "\n";
    auto addFb = [](Fb_mps<cmpx> const &p, int j) {
        auto s = p; s.apply_local_op("Cdag", j); s.psi.normalize(); s.update_cc(); return s; };
    auto psi0 = gsSolver.fb.to_complex(); psi0.tol = 1e-9;
    auto fbB0 = addFb(psi0, 0), fbB1 = addFb(psi0, 1);
    fbB0.tol = fbB1.tol = 1e-9;
    Fbr_dyn dPsi(model, psi0, dt), dB0(model, fbB0, dt), dB1(model, fbB1, dt);

    // ---- full-L star ground state (DMRG on the star Hamiltonian) ----
    auto sites = itensor::Fermion(L, {"ConserveNf", true});
    auto mpo = starHamiltonian(sites, Kstar, U);
    itensor::MPS sPsi;
    {
        auto st = itensor::InitState(sites, "0");
        uvec iek = sort_index(vec{Kstar.diag()});
        for (int j = 0; j < n_part; j++) st.set((int)iek[j] + 1, "1");
        sPsi = itensor::MPS(st);
        auto sw = itensor::Sweeps(1); sw.maxdim() = 1024; sw.cutoff() = 1e-10; sw.niter() = 4; sw.noise() = 1e-8;
        double e = 0;
        for (int i = 0; i < 40; i++) e = itensor::dmrg(sPsi, mpo, sw, {"Quiet", true, "Silent", true});
        cerr << "# star DMRG GS energy   = " << setprecision(10) << e << "\n";
    }
    auto sA = sPsi; sA *= cmpx(1, 0);
    auto sB0 = sPsi; applyCdag(sites, sB0, 0); sB0.normalize(); sB0 *= cmpx(1, 0);
    auto sB1 = sPsi; applyCdag(sites, sB1, 1); sB1.normalize(); sB1 *= cmpx(1, 0);

    string name = "bench_green_cost_L" + to_string(L) + "_U" + us + ".txt";
    ofstream out("test/ref/output/" + name); if (!out) out.open(name);
    out << "bench_green_cost_v1 L " << L << " U " << U << " dt " << dt << "\n"
        << "# t  fewbody_sweep_s fb_nactive fb_bond   star_sweep_s star_bond\n" << setprecision(6);
    cerr << fixed << setprecision(3);
    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;
        double fb_s = seconds([&] { dPsi.iterate({.epsilon_M = 0}); dB0.iterate({.epsilon_M = 0});
                                    dB1.iterate({.epsilon_M = 0}); });
        double st_s = seconds([&] { starTdvp(sA, mpo, dt); starTdvp(sB0, mpo, dt); starTdvp(sB1, mpo, dt); });
        int fb_na = std::max({dPsi.fb.n_active(), dB0.fb.n_active(), dB1.fb.n_active()});
        int fb_m  = std::max({itensor::maxLinkDim(dPsi.fb.psi), itensor::maxLinkDim(dB0.fb.psi),
                              itensor::maxLinkDim(dB1.fb.psi)});
        int st_m  = std::max({itensor::maxLinkDim(sA), itensor::maxLinkDim(sB0), itensor::maxLinkDim(sB1)});
        out << t << "  " << fb_s << " " << fb_na << " " << fb_m << "   " << st_s << " " << st_m << "\n";
        out.flush();
        if (step % 5 == 0)
            cerr << "# t=" << t << "  few-body " << fb_s << "s (n_active=" << fb_na << ", bond=" << fb_m
                 << ")   star " << st_s << "s (bond=" << st_m << ")\n";
    }
    cerr << "# wrote " << name << "\n";
    return 0;
}
