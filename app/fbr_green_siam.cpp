// Impurity Green function of the SIAM, spin-symmetric layout, via Fbr_dyn_shared.
//
// The greater Green function on the impurity (real-space site 0, spin up) is
//
//     G00(t) = -i <psi0| c_0(t) c_0^dag(0) |psi0>,   c_0(t) = e^{iHt} c_0 e^{-iHt}.
//
// Writing A = |psi0> and B = c_0^dag|psi0> and evolving both with the same H,
//     G00(t) = -i <psi0| e^{iHt} c_0 e^{-iHt} c_0^dag |psi0> = -i <c_0^dag A(t) | B(t)>,
// a matrix element between two states that must share the SAME orbital basis at
// every time. Fbr_dyn_shared gives exactly that: the orbital rotations are found
// once (from psi0 and the averaged correlation matrix) and applied to both MPS.
//
// The impurity orbitals are never rotated, so the real-space impurity site is a
// single MPS orbital and c_0 is local there: <A| c_0 |B> = <c_0^dag A | B> is a
// plain MPS overlap (an MPO for c_0 would carry one unit of Nf flux, which
// ITensor's particle-number-conserving AutoMPO cannot build).
//
// The state B = c_0^dag|psi0> has N_up = N_dw + 1, so it does not by itself have
// the up/down mirror symmetry the spin_symmetric layout assumes. That symmetry
// is only used as a heuristic for which orbitals to rotate into the active
// window (the plan is taken from psi0 and mirrored); the wavefunction itself is
// a full MPS over both spin sectors, evolved by the full-Hamiltonian TDVP, so
// the physics is exact up to truncation. Fbr_dyn_shared::widen_to_all_states
// grows the (centered) window to hold every orbital where B differs from psi0,
// which is what keeps the extra up-electron's support inside the window.
//
// This is the spin analogue of example/fbr_green_irlm.cpp (spinless IRLM).
//
// Output columns (one file per L, in app/output/):
//   t  n_active  bond_dim  wall_s  ReG00  ImG00  n0  ReC0n  ImC0n  E_psi0  E_B
// where
//   n0  = <c_0^dag c_0>       impurity (spin-up) occupation
//   C0n = <c_0^dag c_2>       impurity-to-neighbour correlator: the impurity
//                             (real-space site 0) and its nearest bath site of
//                             the same spin (site 2 in the interleaved chain)
//   wall_s = wall-clock seconds spent on that single time step (both states)
//   E_psi0 = energy of the (stationary) ground state, a running sanity check
//   E_B    = energy of the excited state c_0^dag|psi0>
//
// Usage: fbr_green_siam [L] [tmax] [U] [dt]
//   defaults: L=100, tmax=L/2, U=0.1, dt=0.1

#include "fbr/fbr.h"
#include "../test/test_ref_common.h"   // makeSiamModel, shared with the tests

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;
using namespace arma;
using namespace fbr;

namespace {

/// c_j^dag|psi0>, normalized, together with the norm it had before normalizing.
/// The two states of one Fbr_dyn_shared share their Slater determinant, so the
/// state has to be normalized; its norm goes back into G afterwards.
pair<Fb_mps<cmpx>, double> add_particle(Fb_mps<cmpx> const& psi0, int j)
{
    auto state = psi0;
    state.apply_local_op("Cdag", j);
    double nrm = std::sqrt(std::real(itensor::innerC(state.psi, state.psi)));
    state.psi.normalize();
    state.update_cc();
    return {state, nrm};
}

/// <A| c_i |B>, for i a non-rotating impurity site: <c_i^dag A | B>.
cmpx c_element(Fb_mps<cmpx> const& A, Fb_mps<cmpx> const& B, int i)
{
    auto Ai = A;
    Ai.apply_local_op("Cdag", i);
    return itensor::innerC(Ai.psi, B.psi);
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

    auto model = fbrtest::makeSiamModel(L, U, V);

    // ---- ground state (spin-symmetric) -------------------------------------
    auto gs = slater<double>(model);
    gs.tol = 1e-12;
    auto gs_solver = Fbr_gs(model, gs);
    itensor::cpu_time clk;
    for (int i = 0; i < 80; i++) gs_solver.iterate({.max_bond_dim = 512});
    cout << setprecision(12)
         << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt
         << " tmax=" << tmax << "\n"
         << "# ground state: energy=" << gs_solver.energy
         << " n_active=" << gs_solver.fb.n_active()
         << " in " << clk.sincemark().wall << " s" << endl;

    // ---- one shared run: {c_0^dag psi0, psi0} ------------------------------
    // Master-slave convention: the first state is the master and drives the
    // shared orbital basis. The excitation B=c_0^dag|psi0> is the hard evolution,
    // so it is the master; the stationary ground state psi0 is the slave.
    auto psi0 = gs_solver.fb.to_complex();
    psi0.tol = 1e-12;
    auto [B, nrm] = add_particle(psi0, 0);
    B.tol = psi0.tol;
    auto solver = Fbr_dyn_shared(model, std::vector{B, psi0}, dt);
    constexpr int iB = 0, iPsi0 = 1;   // master, slave

    // the impurity site must be a single MPS orbital for c_element to be local:
    // row 0 of the effective rotation is then a pure phase on one column. (The
    // impurity sits at the centre of the spin-symmetric chain, not at index 0.)
    {
        arma::cx_mat Q = solver.effective_rot();
        arma::rowvec row = arma::abs(Q.row(0));
        double mx = row.max();
        if (std::abs(mx - 1) > 1e-10
            || std::sqrt(arma::accu(arma::square(row)) - mx * mx) > 1e-10)
            throw std::runtime_error("impurity site 0 is not a single MPS orbital");
    }

    // ---- output ------------------------------------------------------------
    string name = "app/output/fbr_green_siam_L" + to_string(L)
                + "_U" + us + ".dat";
    ofstream out(name);
    out << setprecision(12);
    out << "# SIAM impurity greater Green function, spin-symmetric FBR\n"
        << "# L=" << L << " U=" << U << " V=" << V << " dt=" << dt
        << " tmax=" << tmax << " E_gs=" << gs_solver.energy << "\n"
        << "# t  n_active  bond_dim  wall_s  ReG00  ImG00  n0  ReC0n  ImC0n  E_psi0  E_B\n";

    int nStep = (int)std::llround(tmax / dt);
    itensor::cpu_time t0;
    for (int step = 0; step <= nStep; step++) {
        double t = step * dt;

        // G00(t) = -i <c_0^dag psi0(t) | B(t)>, with psi0 the slave and B the master
        cmpx G00 = -imag_1 * nrm * c_element(solver.states[iPsi0], solver.states[iB], 0);
        double n0 = std::real(solver.correlator(0, 0, iPsi0));            // on psi0
        cmpx c0n  = solver.correlator(0, 2, iPsi0);   // impurity -> nearest bath (same spin)
        double wall = t0.sincemark().wall;

        out << t
            << " " << solver.states.front().n_active()
            << " " << itensor::maxLinkDim(solver.states[iB].psi)
            << " " << wall
            << " " << G00.real() << " " << G00.imag()
            << " " << n0
            << " " << c0n.real() << " " << c0n.imag()
            << " " << solver.energies[iPsi0] << " " << solver.energies[iB]
            << endl;
        t0.mark();

        if (step < nStep)
            solver.iterate({.epsilon_M = 0});
    }
    cout << "# wrote " << name << endl;
    return 0;
}
