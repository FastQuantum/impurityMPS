// Cost/accuracy of the SIAM Green-function excitation B = c_0up^dag|gs> under
// real-time evolution, as a function of the two truncation cutoffs:
//
//   mps_cutoff    the MPS/circuit truncation (Givens gates + TDVP sweeps: fb.tol)
//   activity_tol  the orbital-activity cutoff that promotes bath orbitals into
//                 the active window (coupling rank) and demotes empty/full ones
//                 back to the Slater part (fb.activity_tol)
//
// Only B is evolved (|gs> is an eigenstate, autocorrelation route; see
// excitation_bond_siam.cpp and app/README.md).
//
// Diagnostics per step (columns):
//   t  n_active  bond_dim  renyi_half  renyi_sum  slater_activity  n0up  n0dw  norm  wall_s  ReA  ImA
//   * renyi_half       Renyi-1/2 entropy at the bottleneck (max-dim) bond. Jumps
//                      when the bottleneck bond moves -- use renyi_sum for a curve.
//   * renyi_sum        Renyi-1/2 summed over all bonds (one sweep): extensive,
//                      smooth in time. If it keeps climbing while bond_dim is
//                      pinned at max_bond_dim, the truncation is discarding real
//                      weight; if it flattens, the cutoffs are loose enough.
//   * slater_activity  max_i min(n_i, 1-n_i) over the Slater (frozen) orbitals.
//                      This is the physics thrown away by the activity cutoff:
//                      it should sit near activity_tol. If it is much larger, the
//                      window is too aggressively pruned (signal, not noise).
//   * n0up n0dw        impurity occupations <c_i^dag c_i> of B(t) (Schrodinger,
//                      real-space, frame-independent). This is the physical
//                      observable to compare across cutoffs: a loose run is
//                      accurate as long as n0up(t) tracks the tight reference.
//   * norm             should stay 1; a drift flags an unstable truncation.
//   * ReA ImA          A(t)=<B0|B(t)> (star only; 0 for fbr, whose frames differ).
//
// Output: app/output/excitation_cutoff_siam_<method>_L<L>_U<U>_mc<mps>_ac<act>.dat
//
// Usage: excitation_cutoff_siam method [L] [tmax] [U] [dt] [maxdim] [mps_cutoff] [activity_tol]
//        defaults: fbr 100 L/2 0.05 0.1 1024 1e-9 (activity_tol defaults to mps_cutoff)
//        L a multiple of 4; method in {fbr, star, star_gs}

#include "fbr/fbr.h"
#include "full_mps.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {

ImpurityParam siam_model(int L, double U, double V)
{
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    auto model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1}, .layout = spin_block};
    model.to_star();
    return model;
}

/// max_i min(n_i, 1-n_i) over the Slater (frozen) orbitals of both spin sectors.
double slater_activity(Fb_mps<cmpx> const& fb)
{
    double m = 0;
    for (Spin s : {up, dw}) {
        auto [a, b] = fb.range(Part::slater, s);
        for (int i = a; i < b; i++) {
            double n = std::real(fb.cc(i, i));
            m = std::max(m, std::min(n, 1 - n));
        }
    }
    return m;
}

struct Writer {
    ofstream out;
    itensor::cpu_time clk;
    Writer(string const& name, string const& head) : out(name)
    {
        out << setprecision(12) << head
            << "# t  n_active  bond_dim  renyi_half  renyi_sum  slater_activity  n0up  n0dw  norm  wall_s  ReA  ImA\n";
        cout << "# writing " << name << endl;
    }
    void operator()(double t, int n_active, int bond, double renyi_max, double renyi_sum,
                    double sla, double n0up, double n0dw, double norm, cmpx A)
    {
        out << t << " " << n_active << " " << bond << " " << renyi_max << " " << renyi_sum
            << " " << sla << " " << n0up << " " << n0dw << " " << norm << " "
            << clk.sincemark().wall << " " << A.real() << " " << A.imag() << endl;
        clk.mark();
    }
};

} // namespace

int main(int argc, char** argv)
{
    string method = argc > 1 ? argv[1] : "fbr";
    int L         = argc > 2 ? std::stoi(argv[2]) : 100;
    double tmax   = argc > 3 ? std::stod(argv[3]) : L / 2.0;
    string us     = argc > 4 ? argv[4] : "0.05";
    double U      = std::stod(us);
    double dt     = argc > 5 ? std::stod(argv[5]) : 0.1;
    int maxdim    = argc > 6 ? std::stoi(argv[6]) : 1024;
    string mcs    = argc > 7 ? argv[7] : "1e-9";
    double mps_cut = std::stod(mcs);
    string acs    = argc > 8 ? argv[8] : mcs;   // activity_tol defaults to mps_cutoff
    double act_cut = std::stod(acs);
    double V      = 0.1;
    int nStep     = (int)std::llround(tmax / dt);

    bool star = method == "star" || method == "star_gs";
    if (!star && method != "fbr")
        throw invalid_argument("method must be fbr, star or star_gs");
    if (L % 4) throw invalid_argument("L must be a multiple of 4");

    auto model = siam_model(L, U, V);
    int i_up = (int)arma::abs(model.rot.row(0)).index_max();
    int i_dw = (int)arma::abs(model.rot.row(1)).index_max();

    string name = "app/output/excitation_cutoff_siam_" + method + "_L" + to_string(L)
                + "_U" + us + "_mc" + mcs + "_ac" + acs + ".dat";
    ostringstream head;
    head << setprecision(12)
         << "# SIAM excitation B=c_0up^dag|gs> (normalized), method=" << method << "\n"
         << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt << " tmax=" << tmax
         << " max_bond_dim=" << maxdim << " mps_cutoff=" << mps_cut
         << " activity_tol=" << act_cut << "\n";
    itensor::cpu_time clk;

    if (star) {
        auto sites = itensor::Fermion(L, {"ConserveNf", true});
        auto mpo = full_mps::hamiltonian(sites, model.Kmat, model.Umat);
        vec seed = vec{model.Kmat.diag()};
        seed[i_up] = seed[i_dw] = -1e9;
        auto psi = full_mps::product_state(sites, seed, model.n_part());
        double e0 = full_mps::ground_state(psi, mpo);
        head << "# E_gs=" << e0 << " gs_bond_dim=" << itensor::maxLinkDim(psi)
             << " gs_wall_s=" << clk.sincemark().wall << "\n";

        auto B = psi;
        if (method == "star") full_mps::apply_cdag(sites, B, i_up);
        double nrm = std::sqrt(std::real(itensor::innerC(B, B)));
        B.normalize();
        B *= cmpx(1, 0);
        auto B0 = B;
        head << "# nrm=" << nrm << "  G00(t) = -i nrm^2 exp(i E_gs t) A(t)\n";

        Writer write(name, head.str());
        for (int step = 0; step <= nStep; step++) {
            int m = itensor::maxLinkDim(B);
            auto [rsum, rmax] = fbr::renyi_half_profile(B);
            double norm = std::sqrt(std::real(itensor::innerC(B, B)));
            write(step * dt, L, m, rmax, rsum, 0.0, full_mps::density(sites, B, i_up),
                  full_mps::density(sites, B, i_dw), norm, itensor::innerC(B0, B));
            if (m >= maxdim) {
                write.out << "# bond dim " << m << " reached " << maxdim << "; stopping\n";
                break;
            }
            if (step < nStep) full_mps::tdvp_step(B, mpo, dt, mps_cut, maxdim);
        }
        return 0;
    }

    // ---- few-body: ground state (cached), then the excitation alone ----
    // gs_tol=1e-9 / 50 iters converges E to ~1e-6 at ~9x less cost than the old
    // 1e-12 / 80 iters (see app/gs_tune_siam.cpp); far tighter than the dynamics
    // cutoff needs. Solve once per (L,U) and cache it for later experiments.
    std::filesystem::create_directories("app/output/gs_cache");
    string gs_prefix = "app/output/gs_cache/siam_L" + to_string(L) + "_U" + us;
    Fb_mps<double> gsfb;
    if (std::filesystem::exists(gs_prefix + ".meta")) {
        gsfb = Fb_mps<double>::load(gs_prefix);
        head << "# gs loaded from cache " << gs_prefix
             << " gs_n_active=" << gsfb.n_active()
             << " gs_bond_dim=" << itensor::maxLinkDim(gsfb.psi) << "\n";
    } else {
        auto gs = slater<double>(model);
        gs.tol = 1e-9;
        auto gs_solver = Fbr_gs(model, gs);
        for (int i = 0; i < 50; i++) gs_solver.iterate({.max_bond_dim = 512});
        gsfb = gs_solver.fb;
        gsfb.save(gs_prefix);
        head << "# E_gs=" << gs_solver.energy << " gs_n_active=" << gsfb.n_active()
             << " gs_bond_dim=" << itensor::maxLinkDim(gsfb.psi)
             << " gs_wall_s=" << clk.sincemark().wall << " (cached to " << gs_prefix << ")\n";
    }

    auto fb = gsfb.to_complex();
    fb.tol = mps_cut;
    fb.activity_tol = act_cut;
    fb.apply_local_op("Cdag", 0);
    head << "# nrm=" << std::sqrt(std::real(itensor::innerC(fb.psi, fb.psi))) << "\n";
    fb.psi.normalize();
    fb.update_cc();

    auto solver = Fbr_dyn(model, fb, dt);
    Writer write(name, head.str());
    for (int step = 0; step <= nStep; step++) {
        double norm = std::sqrt(std::real(itensor::innerC(solver.fb.psi, solver.fb.psi)));
        int m = itensor::maxLinkDim(solver.fb.psi);
        auto [rsum, rmax] = fbr::renyi_half_profile(solver.fb.psi);
        write(step * dt, solver.fb.n_active(), m, rmax, rsum, slater_activity(solver.fb),
              std::real(solver.correlator(0, 0)), std::real(solver.correlator(1, 1)),
              norm, 0);
        if (m >= maxdim) {
            write.out << "# bond dim " << m << " reached " << maxdim << "; stopping\n";
            break;
        }
        if (step < nStep) solver.iterate({.max_bond_dim = maxdim, .epsilon_M = 0});
    }
    return 0;
}
