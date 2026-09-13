// Brute-force STAR baseline for the SIAM impurity Green function, for comparison
// against the few-body (active-window) result of app/fbr_green_siam.cpp and the
// chain baseline chain_green_siam.cpp.
//
//     G00(t) = -i <psi0| c_0(t) c_0^dag(0) |psi0> = -i <c_0^dag A(t) | B(t)>
//
// Same physics as chain_green_siam, but each spin's bath is diagonalised into
// energy eigenmodes (star geometry, chain_dyn/ star_dyn_siam_center's computeKstar),
// the same basis the FBR evolves in -- only here the WHOLE L-site MPS is kept, no
// active window. The impurity couples to every bath eigenmode (long-range), so the
// MPS is far less friendly than the chain and the bond dimension climbs fast: this
// is exactly the cost the active-window FBR removes.
//
// The impurity orbitals are not rotated by computeKstar, so c_imp is still a local
// MPS operator and G00 is a plain overlap. The occupation and the impurity-to-
// neighbour correlator are real-space quantities, so the correlation matrix is
// rotated back, cc_real = rot * cc_star * rot^T, before reading n0 and C0n.
//
// Output (test/ref/output/star_green_siam_L<L>_U<U>.dat, run from the repo root):
//   t  bond_dim  wall_s  ReG00  ImG00  n0  ReC0n  ImC0n
// Stops as soon as the bond dimension of any state reaches 1024.
//
// Usage: star_green_siam [L] [tmax] [U] [dt]   (defaults: 100, L/2, 0.1, 0.1)

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

/// Star geometry and the rotation that produced it (star -> original basis):
/// c_i = sum_a rot(i,a) d_a, so <c_i^dag c_j> = (rot * cc_star * rot^T)(i,j).
/// Layout matches star_dyn_siam_center: [up bath | up imp | dw imp | dw bath],
/// the up impurity at the right of its half, the down impurity at the left.
auto computeKstar(mat K, int n_imp)
{
    int L = K.n_rows;
    int nBath = L / 2 - n_imp / 2;
    mat Kstar(L, L, fill::zeros);
    mat rot(L, L, fill::eye);

    auto pos_up = regspace<uvec>(0, L / 2 - 1);
    auto pos_dw = regspace<uvec>(L / 2, L - 1);
    for (int s : {0, 1}) {
        uvec pos      = s == 0 ? pos_up : pos_dw;
        uvec pos_bath = s == 0 ? pos.head(nBath)  : pos.tail(nBath);
        uvec pos_impu = s == 0 ? pos.tail(n_imp / 2) : pos.head(n_imp / 2);

        mat Kbath = K.submat(pos_bath, pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1, evec1, Kbath);
        uvec iek = s == 0 ? sort_index(abs(ek1), "descend") : sort_index(abs(ek1));
        mat evec = evec1.cols(iek);
        vec ek   = ek1.rows(iek);

        mat vk = K.submat(pos_impu, pos_bath).eval() * evec;
        Kstar.submat(pos_impu, pos_impu) = K.submat(pos_impu, pos_impu);
        for (auto j = 0u; j < ek.size(); j++) {
            int jj = pos_bath[j];
            Kstar(jj, jj) = ek[j];
            for (auto i = 0u; i < pos_impu.size(); i++) {
                int ii = pos_impu[i];
                Kstar(ii, jj) = Kstar(jj, ii) = vk(i, j);
            }
        }
        rot.cols(pos_bath) = rot.cols(pos_bath).eval() * evec;
    }
    return make_pair(Kstar, rot);
}

void applyCdag(itensor::Fermion const& sites, itensor::MPS& psi, int site)
{
    for (int k = 1; k <= site; k++) {
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
    if (std::abs(U) > 1e-15) h += U, "N", iu + 1, "N", id + 1;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (std::abs(K(i, j)) > 1e-12)
                h += K(i, j), "Cdag", i + 1, "C", j + 1;
    return itensor::toMPO(h);
}

void findGs(itensor::MPS& psi, itensor::MPO const& mpo)
{
    // The star couples the impurity to every bath mode (long range), so plain
    // DMRG traps easily in a wrong-occupation local minimum. A strong, slowly
    // decaying noise and a bond dimension grown over many sweeps is what shakes
    // it loose onto the true ground state.
    std::vector<double> noise = {1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-8, 0};
    cout << "# dmrg sweep m energy" << endl << setprecision(12);
    double e = 0;
    for (int i = 0; i < 60; i++) {
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = std::min(64 + 32 * i, 1024);
        sweeps.cutoff() = 1e-12;
        sweeps.niter()  = 4;
        sweeps.noise()  = i < (int)noise.size() ? noise[i] : 0.0;
        e = itensor::dmrg(psi, mpo, sweeps, {"Quiet", true, "Silent", true});
        if (i % 5 == 4) cout << "#   " << i + 1 << " " << itensor::maxLinkDim(psi) << " " << e << endl;
    }
    cout << "# ground state energy = " << e << endl;
}

void do_tdvp(itensor::MPS& psi, itensor::MPO const& mpo, double dt)
{
    // The star couples the impurity to all modes; resolve the expansion well
    // (same tuned settings as star_dyn_siam_center).
    fbr::TdvpParam args{.err_goal = 1e-7, .epsilon_M = 3e-7, .n_krylov = 2, .epsilon_K = 3e-8};
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

    int iu = nBath + n_imp / 2 - 1;   // up impurity ("site 0"), not rotated
    int id = L / 2;                   // down impurity
    int nb = nBath;                   // up impurity's first bath neighbour ("site 2")

    // real-space centre-chain SIAM, then diagonalise each bath into star modes
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L / 2 - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    for (int i = L / 2; i < L - 1; i++) K(i, i + 1) = K(i + 1, i) = 0.5;
    K(iu, iu) = -U / 2;
    K(id, id) = -U / 2;
    K(nBath, iu) = K(iu, nBath) = V;
    K(L / 2, L / 2 + n_imp / 2 - 1) = K(L / 2 + n_imp / 2 - 1, L / 2) = V;

    mat Kstar, rot;
    std::tie(Kstar, rot) = computeKstar(K, n_imp);
    cx_mat cxrot = conv_to<cx_mat>::from(rot);

    auto sites = itensor::Fermion(L, {"ConserveNf", true});
    auto mpo   = getHamiltonian(sites, Kstar, U, iu, id);

    // half filling; DMRG relaxes to the true ground state in the star basis.
    // Seed with the two physical impurities OCCUPIED and the rest of the
    // particles in the lowest bath modes: starting from an empty impurity, plain
    // DMRG traps in a wrong-occupation local minimum (the impurity is at the far
    // end of the star and two-site updates cannot fill it).
    itensor::MPS psi;
    {
        vec ek = vec{Kstar.diag()};
        ek[iu] = ek[id] = -1e9;                 // force the impurities filled first
        auto state = itensor::InitState(sites, "0");
        uvec iek = sort_index(ek);
        for (int j = 0; j < L / 2; j++) state.set((int)iek[j] + 1, "1");
        psi = itensor::MPS(state);
    }
    itensor::cpu_time clk;
    findGs(psi, mpo);
    cout << "# ground state in " << clk.sincemark().wall << " s" << endl;

    auto A = psi;
    auto B = psi; applyCdag(sites, B, iu);
    double nrm = std::sqrt(std::real(itensor::innerC(B, B)));
    B.normalize();
    A *= cmpx(1, 0);
    B *= cmpx(1, 0);

    string name = "test/ref/output/star_green_siam_L" + to_string(L)
                + "_U" + us + ".dat";
    ofstream out(name);
    out << setprecision(12);
    out << "# SIAM impurity Green function, brute-force STAR baseline\n"
        << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt << " tmax=" << tmax << "\n"
        << "# t  bond_dim  wall_s  ReG00  ImG00  n0  ReC0n  ImC0n\n";

    int nStep = (int)std::llround(tmax / dt);
    itensor::cpu_time t0;
    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;

        auto A0 = A; applyCdag(sites, A0, iu);
        cmpx G00 = -cmpx(0, 1) * nrm * itensor::innerC(A0, B);

        cx_mat cc = cxrot * fbr::get_cc(sites, A) * cxrot.t();   // back to real space
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
