// Brute-force CHAIN baseline for the SIAM impurity Green function, for comparison
// against the few-body (active-window) result of app/fbr_green_siam.cpp; the
// trusted SIAM counterpart of chain_green_irlm.cpp.
//
//     G00(t) = -i <psi0| c_0(t) c_0^dag(0) |psi0> = -i <c_0^dag A(t) | B(t)>
//
// with A=|psi0>, B=c_0^dag|psi0>, both evolved by the same full-Hamiltonian MPO.
// Everything is a plain real-space chain: DMRG for the ground state, two-site
// TDVP for the evolution, no orbital rotation and no interaction picture. The
// whole L-site MPS is kept (no active window), so this is the trusted, expensive
// reference the FBR is measured against.
//
// The model is the same SIAM as fbr_green_siam.cpp -- impurity coupled by V to a
// tight-binding bath of hopping 0.5, Hubbard U on the impurity -- laid out in the
// MPS-friendly "centre" geometry (chain_dyn_siam_center): the spin-up chain fills
// sites 0..L/2-1 with the up impurity at its right end, the spin-down chain fills
// L/2..L-1 with the down impurity at its left end, so the two impurities sit
// adjacent at the centre:
//     outer_up .. buf_up  imp_up | imp_dw  buf_dw .. outer_dw
// buf_* is the impurity's first bath site (the "site 2" neighbour), reached by V.
//
// Output (test/ref/output/chain_green_siam_L<L>_U<U>.dat, run from the repo root):
//   t  bond_dim  wall_s  ReG00  ImG00  n0  ReC0n  ImC0n
// same G00 / n0=<c_imp^dag c_imp> / C0n=<c_imp^dag c_neighbour> as the FBR run.
// Stops as soon as the bond dimension of any state reaches 1024.
//
// Usage: chain_green_siam [L] [tmax] [U] [dt]   (defaults: 100, L/2, 0.1, 0.1)

#include <itensor/all.h>
#include <fbr/itensor_utils.h>
#include <tdvp.h>
#include <basisextension.h>
#include <armadillo>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace std;
using namespace arma;
using cmpx = std::complex<double>;

namespace {

/// The true fermionic c_site^dag, Jordan-Wigner string included (0-based site).
void applyCdag(itensor::Fermion const& sites, itensor::MPS& psi, int site)
{
    for (int k = 1; k <= site; k++) {          // F is diagonal and unitary
        auto A = sites.op("F", k) * psi(k);
        A.noPrime();
        psi.set(k, A);
    }
    psi.position(site + 1);
    auto A = sites.op("Cdag", site + 1) * psi(site + 1);
    A.noPrime();
    psi.set(site + 1, A);
}

itensor::MPO getHamiltonian(itensor::Fermion const& sites, mat const& K,
                            double U, int iu, int id)
{
    int L = K.n_rows;
    itensor::AutoMPO h(sites);
    if (std::abs(U) > 1e-15) h += U, "N", iu + 1, "N", id + 1;   // Hubbard on the impurity
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (std::abs(K(i, j)) > 1e-12)
                h += K(i, j), "Cdag", i + 1, "C", j + 1;
    return itensor::toMPO(h);
}

void findGs(itensor::MPS& psi, itensor::MPO const& mpo)
{
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024;
    sweeps.cutoff() = 1e-12;
    sweeps.niter()  = 4;
    sweeps.noise()  = 1e-8;
    cout << "# dmrg sweep m energy" << endl << setprecision(12);
    double e = 0;
    for (int i = 0; i < 24; i++) {
        e = itensor::dmrg(psi, mpo, sweeps, {"Quiet", true, "Silent", true});
        if (i % 4 == 3) cout << "#   " << i + 1 << " " << itensor::maxLinkDim(psi) << " " << e << endl;
    }
    cout << "# ground state energy = " << e << endl;
}

void do_tdvp(itensor::MPS& psi, itensor::MPO const& mpo, double dt)
{
    fbr::TdvpParam args{.err_goal = 1e-8, .epsilon_M = 1e-4, .n_krylov = 2, .epsilon_K = 1e-4};
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = args.max_bond_dim;
    sweeps.cutoff() = 1e-12;
    sweeps.niter()  = args.n_iter_diag;
    sweeps.noise()  = args.noise;

    std::vector<double> epsilon_K(args.n_krylov, args.epsilon_K);
    itensor::addBasis(psi, mpo, epsilon_K,
                      {"Cutoff", args.epsilon_M, "Method", "DensityMatrix",
                       "KrylovOrd", args.n_krylov, "DoNormalize", true,
                       "Quiet", true, "Silent", true});

    itensor::tdvp(psi, mpo, -cmpx(0, 1) * dt, sweeps,
                  {"Truncate", true, "DoNormalize", true, "Quiet", true,
                   "Silent", true, "NumCenter", 2, "ErrGoal", args.err_goal});
}

} // namespace

int main(int argc, char** argv)
{
    int L       = argc > 1 ? std::stoi(argv[1]) : 100;
    double tmax = argc > 2 ? std::stod(argv[2]) : L / 2.0;
    string us   = argc > 3 ? argv[3] : "0.1";   // U as typed, names the output
    double U    = std::stod(us);
    double dt   = argc > 4 ? std::stod(argv[4]) : 0.1;
    double V    = 0.1;
    int n_imp   = 4;
    int nBath   = L / 2 - n_imp / 2;
    int maxBondDim = 1024;

    int iu = nBath + n_imp / 2 - 1;   // up impurity (real-space "site 0")
    int id = L / 2;                   // down impurity
    int nb = nBath;                   // up impurity's first bath neighbour ("site 2")

    mat K(L, L, fill::zeros);
    for (int i = 0; i < L / 2 - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;   // spin-up chain
    for (int i = L / 2; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;   // spin-down chain
    K(iu, iu) = -U / 2;
    K(id, id) = -U / 2;
    K(nBath, iu) = K(iu, nBath) = V;                                        // up hybridization
    K(L / 2, L / 2 + n_imp / 2 - 1) = K(L / 2 + n_imp / 2 - 1, L / 2) = V;  // down hybridization

    auto sites = itensor::Fermion(L, {"ConserveNf", true});
    auto mpo   = getHamiltonian(sites, K, U, iu, id);

    // half filling, one particle per two sites; DMRG relaxes to the true ground state
    itensor::MPS psi;
    {
        auto state = itensor::InitState(sites, "0");
        for (int j = 0; j < L / 2; j++) state.set(2 * j + 1, "1");
        psi = itensor::MPS(state);
    }
    itensor::cpu_time clk;
    findGs(psi, mpo);
    cout << "# ground state in " << clk.sincemark().wall << " s" << endl;

    // A=|psi0>, B=c_imp^dag|psi0> (normalized, norm folded back into G)
    auto A = psi;
    auto B = psi; applyCdag(sites, B, iu);
    double nrm = std::sqrt(std::real(itensor::innerC(B, B)));
    B.normalize();
    A *= cmpx(1, 0);
    B *= cmpx(1, 0);

    string name = "test/ref/output/chain_green_siam_L" + to_string(L)
                + "_U" + us + ".dat";
    ofstream out(name);
    out << setprecision(12);
    out << "# SIAM impurity Green function, brute-force CHAIN baseline\n"
        << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt << " tmax=" << tmax << "\n"
        << "# t  bond_dim  wall_s  ReG00  ImG00  n0  ReC0n  ImC0n\n";

    int nStep = (int)std::llround(tmax / dt);
    itensor::cpu_time t0;
    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;

        auto A0 = A; applyCdag(sites, A0, iu);          // <A| c_imp
        cmpx G00 = -cmpx(0, 1) * nrm * itensor::innerC(A0, B);

        cx_mat cc = fbr::get_cc(sites, A);              // |psi0> is an eigenstate: <n0>,<c0c2> are
                                                        // stationary up to truncation, same as the FBR
        double n0 = std::real(cc(iu, iu));
        cmpx c0n  = cc(iu, nb);
        int m = std::max(itensor::maxLinkDim(A), itensor::maxLinkDim(B));
        double wall = t0.sincemark().wall;

        out << t << " " << m << " " << wall
            << " " << G00.real() << " " << G00.imag()
            << " " << n0 << " " << c0n.real() << " " << c0n.imag() << endl;
        t0.mark();

        if (m >= maxBondDim) {
            out << "# bond dim " << m << " reached " << maxBondDim << " at t=" << t
                << "; stopping\n";
            cout << "# bond dim " << m << " reached " << maxBondDim << " at t=" << t << "; stopping\n";
            break;
        }
        if (step < nStep)
            for (auto* p : {&A, &B}) do_tdvp(*p, mpo, dt);
    }
    cout << "# wrote " << name << endl;
    return 0;
}
