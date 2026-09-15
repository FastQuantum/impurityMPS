// Bond dimension of the SIAM Green-function excitation B = c_0up^dag|gs>
// evolving in time: the full star MPS against the few-body FBR, for a scan of U.
//
// Only B is evolved: |gs> is an eigenstate, so G00(t) = -i nrm^2 e^{iE0 t} <B|B(t)>
// (the autocorrelation route, see app/README.md), and the cost of the Green
// function is the cost of this one evolution.
//
// Model (test/test_ref_common.h makeSiamModel): interleaved spin chains of
// hopping 0.5, impurity sites 0 (up) and 1 (dw), V=0.1, e_imp=-U/2, U n0 n1,
// half filling. One ImpurityParam, after to_star(), is used by every method, so
// the star below is exactly the basis the FBR evolves in.
//
//   star     two-site TDVP of the whole L-site MPS in the star geometry
//   fbr      Fbr_dyn, spin_block layout (each spin sector gets its own orbitals;
//            the excitation breaks the spin-flip symmetry)
//   fbr_sym  Fbr_dyn, spin_symmetric layout (the up rotations mirror the down ones)
//
//   star_gs  the same full star TDVP applied to |gs> itself (a stationary state:
//            any bond growth there is the cost of the TDVP, not of the physics)
//
// All truncate the MPS at the same cutoff (default 1e-10; a different value is
// appended to the file name). The star runs stop once their bond dimension
// reaches max_bond_dim.
//
// Output: app/output/excitation_bond_siam_<method>_L<L>_U<U>.dat, columns
//   t  n_active  bond_dim  wall_s  n0up  n0dw  ReA  ImA
// with n0up, n0dw the impurity occupations of B(t) and A = <B|B(t)> (star only;
// 0 for the FBR, whose frames differ between t=0 and t).
//
// Usage: excitation_bond_siam method [L] [tmax] [U] [dt] [max_bond_dim] [cutoff]
//        (defaults star 100 L/2 0.05 0.1 1024 1e-10); L a multiple of 4

#include "fbr/fbr.h"
#include "full_mps.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {

ImpurityParam siam_model(int L, double U, double V, Layout layout)
{
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
    K(0, 0) = K(1, 1) = -U / 2;
    K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
    mat Umat(L, L, fill::zeros);
    Umat(0, 1) = U;
    auto model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1}, .layout = layout};
    model.to_star();
    return model;
}

struct Writer {
    ofstream out;
    itensor::cpu_time clk;
    Writer(string const& name, string const& head) : out(name)
    {
        out << setprecision(12) << head
            << "# t  n_active  bond_dim  wall_s  n0up  n0dw  ReA  ImA\n";
        cout << "# writing " << name << endl;
    }
    void operator()(double t, int n_active, int bond, double n0up, double n0dw, cmpx A)
    {
        out << t << " " << n_active << " " << bond << " " << clk.sincemark().wall
            << " " << n0up << " " << n0dw << " " << A.real() << " " << A.imag() << endl;
        clk.mark();
    }
};

} // namespace

int main(int argc, char** argv)
{
    string method = argc > 1 ? argv[1] : "star";
    int L         = argc > 2 ? std::stoi(argv[2]) : 100;
    double tmax   = argc > 3 ? std::stod(argv[3]) : L / 2.0;
    string us     = argc > 4 ? argv[4] : "0.05";   // U as typed, names the output
    double U      = std::stod(us);
    double dt     = argc > 5 ? std::stod(argv[5]) : 0.1;
    int maxdim    = argc > 6 ? std::stoi(argv[6]) : 1024;
    string cs     = argc > 7 ? argv[7] : "1e-10";
    double cutoff = std::stod(cs);
    double V      = 0.1;
    int nStep     = (int)std::llround(tmax / dt);

    bool star = method == "star" || method == "star_gs";
    if (!star && method != "fbr" && method != "fbr_sym")
        throw invalid_argument("method must be star, star_gs, fbr or fbr_sym");
    if (L % 4) throw invalid_argument("L must be a multiple of 4");

    auto model = siam_model(L, U, V, method == "fbr_sym" ? spin_symmetric : spin_block);
    // star positions of the real-space impurity sites 0 (up) and 1 (dw)
    int i_up = (int)arma::abs(model.rot.row(0)).index_max();
    int i_dw = (int)arma::abs(model.rot.row(1)).index_max();

    string name = "app/output/excitation_bond_siam_" + method + "_L" + to_string(L)
                + "_U" + us + (cs == "1e-10" ? "" : "_cut" + cs) + ".dat";
    ostringstream head;
    head << setprecision(12)
         << "# SIAM excitation B=c_0up^dag|gs> (normalized), method=" << method << "\n"
         << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt << " tmax=" << tmax
         << " cutoff=" << cutoff << " max_bond_dim=" << maxdim << "\n";
    itensor::cpu_time clk;

    if (star) {
        auto sites = itensor::Fermion(L, {"ConserveNf", true});
        auto mpo = full_mps::hamiltonian(sites, model.Kmat, model.Umat);
        vec seed = vec{model.Kmat.diag()};
        seed[i_up] = seed[i_dw] = -1e9;    // impurity occupied, or DMRG traps
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
            write(step * dt, L, m, full_mps::density(sites, B, i_up),
                  full_mps::density(sites, B, i_dw), itensor::innerC(B0, B));
            if (m >= maxdim) {
                write.out << "# bond dim " << m << " reached " << maxdim << "; stopping\n";
                break;
            }
            if (step < nStep) full_mps::tdvp_step(B, mpo, dt, cutoff, maxdim);
        }
        return 0;
    }

    // ---- few-body: ground state, then the excitation alone ----
    auto gs = slater<double>(model);
    gs.tol = 1e-12;
    auto gs_solver = Fbr_gs(model, gs);
    for (int i = 0; i < 80; i++) gs_solver.iterate({.max_bond_dim = 512});
    head << "# E_gs=" << gs_solver.energy << " gs_n_active=" << gs_solver.fb.n_active()
         << " gs_bond_dim=" << itensor::maxLinkDim(gs_solver.fb.psi)
         << " gs_wall_s=" << clk.sincemark().wall << "\n";

    auto fb = gs_solver.fb.to_complex();
    fb.tol = cutoff;
    fb.apply_local_op("Cdag", 0);
    head << "# nrm=" << std::sqrt(std::real(itensor::innerC(fb.psi, fb.psi))) << "\n";
    fb.psi.normalize();
    fb.update_cc();

    auto solver = Fbr_dyn(model, fb, dt);
    Writer write(name, head.str());
    for (int step = 0; step <= nStep; step++) {
        write(step * dt, solver.fb.n_active(), itensor::maxLinkDim(solver.fb.psi),
              std::real(solver.correlator(0, 0)), std::real(solver.correlator(1, 1)), 0);
        if (step < nStep) solver.iterate({.max_bond_dim = maxdim, .epsilon_M = 0});
    }
    return 0;
}
