// Diagnostic: track real-space correlator through each step of one iterate().
// Builds the same SIAM star state as compare_chain_vs_fb_siam.cpp at t=0,
// then manually runs the iterate() steps one-by-one and prints C[10,3] at each.
//
// Expectation: extract_representative*, swaps, and rotateToNaturalOrbitals
// are unitary basis transformations; they must preserve <c_i^dagger c_j> in
// the IP frame. Only TDVP should change it (by an O(dt) amount on the active
// sector). Any large jump localizes the bug.

#include "impurityMPS/impurity_dyn_spin.h"
#include <armadillo>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace arma;
using cmpx = std::complex<double>;

auto computeKstar(mat K, int nImp)
{
    int L = K.n_rows;
    int nBath = L/2 - nImp/2;
    mat Kstar(L, L, fill::zeros);
    mat rot(L, L, fill::eye);
    auto pos_up = regspace<uvec>(0, L/2-1);
    auto pos_dw = regspace<uvec>(L/2, L-1);
    for (int s : {0, 1}) {
        uvec pos      = (s==0) ? pos_up : pos_dw;
        uvec pos_bath = (s==0) ? pos.head(nBath)  : pos.tail(nBath);
        uvec pos_impu = (s==0) ? pos.tail(nImp/2) : pos.head(nImp/2);
        mat Kbath = K.submat(pos_bath, pos_bath);
        mat evec1; vec ek1; eig_sym(ek1, evec1, Kbath);
        uvec iek_asc = stable_sort_index(abs(ek1));
        uvec iek = (s==0) ? uvec(reverse(iek_asc)) : iek_asc;
        mat evec = evec1.cols(iek);
        vec ek = ek1.rows(iek);
        mat vk = K.submat(pos_impu, pos_bath).eval() * evec;
        Kstar.submat(pos_impu, pos_impu) = K.submat(pos_impu, pos_impu);
        for (auto j=0u; j<ek.size(); j++) {
            int jj = pos_bath[j];
            Kstar(jj, jj) = ek[j];
            for (auto i=0u; i<pos_impu.size(); i++) {
                int ii = pos_impu[i];
                Kstar(ii, jj) = Kstar(jj, ii) = vk(i, j);
            }
        }
        rot.cols(pos_bath) = rot.cols(pos_bath).eval() * evec;
    }
    return make_pair(Kstar, rot);
}

int main()
{
    const int L=32, nImp=4, nBath = L/2 - nImp/2;
    const double dt=0.1, U=0.0, V=0.1;
    const int maxdim = 256;
    const double tol = 1e-10;

    mat Kreal(L, L, fill::zeros);
    for (int i=0; i<L/2-1; i++) Kreal(i,i+1) = Kreal(i+1,i) = 0.5;
    for (int i=L/2; i<L-1; i++) Kreal(i,i+1) = Kreal(i+1,i) = 0.5;
    Kreal(nBath+nImp/2-1, nBath+nImp/2-1) = -U/2;
    Kreal(L/2, L/2)                       = -U/2;
    Kreal(nBath, nBath+nImp/2-1) = Kreal(nBath+nImp/2-1, nBath) = V;
    Kreal(L/2, L/2+nImp/2-1)     = Kreal(L/2+nImp/2-1, L/2)     = V;

    mat UmatSite(L, L, fill::zeros);
    UmatSite(nBath+nImp/2-1, L/2) = U;

    auto [Kstar, rot] = computeKstar(Kreal, nImp);
    Fb_mps_spin<cmpx> fb;
    {
        vec ek = Kstar.diag();
        ek[nBath + nImp/2 - 1] = -10;
        ek[L/2]                = -10;
        ek[nBath]              =  10;
        ek[L/2 + nImp/2 - 1]   =  10;
        fb = Fb_mps_spin<cmpx>::from_slater(rot*cmpx(1,0), ek, L/2, nImp);
    }

    ImpuritySpin model;
    model.param.Kmat = Kstar;
    model.param.Umat = UmatSite;
    model.param.rot  = mat(L, L, fill::eye);
    model.param.impPos = {L/2-2, L/2-1, L/2, L/2+1};

    Impurity_dyn_spin solver(model, fb, dt);
    solver.fb.tol = 1e-10;

    // --- chain side ---
    mat UmatChain(nImp, nImp, fill::zeros);
    UmatChain(nImp/2-1, nImp/2) = U;
    itensor::Fermion sites_chain(L, {"ConserveNf", true});
    itensor::MPS psi_chain;
    {
        int nPart = L/2;
        auto state = itensor::InitState(sites_chain, "0");
        for (int j = 0; j < nPart; j++) state.set(2*j+1, "1");
        state.set(L/2-1, "0");
        state.set(L/2,   "1");
        state.set(L/2+1, "1");
        state.set(L/2+2, "0");
        psi_chain = itensor::MPS(state);
    }
    {
        mat Kfake = Kreal;
        Kfake(nBath, nBath-1)        = Kfake(nBath-1, nBath)        = 0;
        int b2 = L/2 + nImp/2;
        Kfake(b2-1, b2)              = Kfake(b2, b2-1)              = 0;
        Kfake(nBath, nBath+nImp/2-1) = Kfake(nBath+nImp/2-1, nBath) = 0;
        Kfake(L/2, L/2+nImp/2-1)     = Kfake(L/2+nImp/2-1, L/2)     = 0;
        Kfake(L/2-2, L/2-2) =  10;
        Kfake(L/2-1, L/2-1) = -10;
        Kfake(L/2,   L/2)   = -10;
        Kfake(L/2+1, L/2+1) =  10;
        itensor::AutoMPO h(sites_chain);
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (std::abs(Kfake(i,j)) > 1e-12)
                    h += Kfake(i,j), "Cdag", i+1, "C", j+1;
        auto mpoFake = itensor::toMPO(h);
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = 512; sweeps.cutoff() = 1e-10;
        sweeps.niter() = 4; sweeps.noise() = 1e-8;
        for (int i = 0; i < 20; i++)
            itensor::dmrg(psi_chain, mpoFake, sweeps, {"Quiet", true, "Silent", true});
    }
    itensor::MPO mpo_chain;
    {
        itensor::AutoMPO h(sites_chain);
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (std::abs(Kreal(i,j)) > 1e-12)
                    h += Kreal(i,j), "Cdag", i+1, "C", j+1;
        mpo_chain = itensor::toMPO(h);
    }
    auto chainCorr14_0 = [&]() {
        auto C = itensor::correlationMatrixC(psi_chain, sites_chain, "Cdag", "C");
        return cmpx(C[14][0]);
    };

    auto print_corr = [&](string_view label) {
        cx_mat C_raw = solver.fb.correlator_all();    // fb in IP frame (no Schroedinger correction)
        cx_mat C_sch = solver.correlator_all();        // Schroedinger picture (IP corrected)
        cout << "  " << setw(40) << left << label
             << "C_raw[14,0]=" << setw(22) << C_raw(14,0)
             << "C_sch[14,0]=" << setw(22) << C_sch(14,0)
             << " p1=" << solver.fb.p1 << " p2=" << solver.fb.p2
             << "\n";
    };

    cout << setprecision(8);
    cout << "Initial state:\n";
    print_corr("t=0");

    // Full initial-state diff between chain and fb
    {
        auto Cc = itensor::correlationMatrixC(psi_chain, sites_chain, "Cdag", "C");
        cx_mat C_chain(L, L);
        for (int i=0; i<L; i++) for (int j=0; j<L; j++) C_chain(i,j) = Cc[i][j];
        cx_mat C_fb = solver.correlator_all();
        cx_mat D = C_chain - C_fb;
        mat Dabs = arma::abs(D);
        arma::uword wi = Dabs.index_max() % L;
        arma::uword wj = Dabs.index_max() / L;
        cout << "  |chain - fb|_inf at t=0 = " << Dabs.max()
             << " at (" << wi << "," << wj << ")"
             << "  Cc=" << C_chain(wi,wj) << "  Cf=" << C_fb(wi,wj) << "\n";
        cout << "  C_chain[14,0]=" << C_chain(14,0) << "  C_fb[14,0]=" << C_fb(14,0) << "\n";
        cout << "  C_chain[15,0]=" << C_chain(15,0) << "  C_fb[15,0]=" << C_fb(15,0) << "\n";
        cout << "  C_chain[14,13]=" << C_chain(14,13) << "  C_fb[14,13]=" << C_fb(14,13) << "\n";
        cout << "  C_chain[15,15]=" << C_chain(15,15) << "  C_fb[15,15]=" << C_fb(15,15) << "\n";
        cout << "  C_chain[14,14]=" << C_chain(14,14) << "  C_fb[14,14]=" << C_fb(14,14) << "\n";
    }

    auto run_iterate_substeps = [&](int step) {
        cout << "\n=== iterate #" << step << " ===\n";
        arma::cx_mat exp_ih_(L, L, arma::fill::eye);
        exp_ih_.submat(solver.bath_pos, solver.bath_pos) =
            expIH<cmpx>(solver.Kbath * solver.nIter * dt);
        arma::cx_mat rotL_ = exp_ih_ * solver.rotS.t() * solver.fb.rot;
        solver.K = rotL_.t() * solver.Kip0 * rotL_;
        solver.nIter++;
        print_corr("after K rotation");

        if (step == 1) {
            // Expected K in fb basis: fb.rot.t() * Kreal * fb.rot (no IP, no Trotter)
            arma::cx_mat K_expected = solver.fb.rot.t() * Kreal * cmpx(1,0) * solver.fb.rot;
            arma::mat diff = arma::abs(solver.K - K_expected);
            cout << "  [K diag] max|K_computed - K_no_IP|=" << diff.max()
                 << " K_comp[14,0]=" << solver.K(14,0)
                 << " K_exp[14,0]=" << K_expected(14,0)
                 << "\n";
            // Also compare to Kip0 rotated to fb basis
            arma::cx_mat K_kip0_fb = solver.fb.rot.t() * solver.Kip0 * solver.fb.rot;
            cout << "  [K diag] max|K_computed - Kip0_in_fb|=" << arma::abs(solver.K - K_kip0_fb).max()
                 << " Kip0_fb[14,0]=" << K_kip0_fb(14,0)
                 << "\n";
            // Check if Kip0 == Kreal (in fb basis) to see if commutator term is meaningful
            arma::cx_mat Kreal_star = solver.fb.rot.t() * Kreal * cmpx(1,0) * solver.fb.rot;
            cout << "  [K diag] max|Kip0_fb - Kreal_fb|=" << arma::abs(K_kip0_fb - Kreal_star).max() << "\n";
            // Print K on the impurity-bath cross block (14..17, 0..13)
            cout << "  [K diag] K row 14 (bath cols 0..5): ";
            for (int c=0; c<6; c++) cout << solver.K(14,c) << " ";
            cout << "\n";
            cout << "  [K diag] K_exp row 14 (bath cols 0..5): ";
            for (int c=0; c<6; c++) cout << K_expected(14,c) << " ";
            cout << "\n";
        }
        solver.extract_representative(0);
        print_corr("after extract_representative(0)");
        solver.extract_representative(1);
        print_corr("after extract_representative(1)");
        solver.extract_representative_final();
        print_corr("after extract_representative_final");
        // Print rotating-block occupations BEFORE doTdvp, to see what came in
        {
            arma::vec ni = solver.fb.occupations_ni();
            cout << "    ni[" << solver.fb.p1 << "..." << solver.fb.p2-1 << "] = ";
            for (int k = solver.fb.p1; k < solver.fb.p2; k++) cout << ni[k] << " ";
            cout << "\n";
        }
        arma::cx_mat cc_before = solver.fb.cc;
        arma::cx_mat K_snapshot = solver.K;
        solver.doTdvp({.max_bond_dim = 256, .nIter_diag = 16, .epsilonM = 1e-4});
        print_corr("after doTdvp");

        if (step == 1) {
            // Manually evolve cc_before under K_snapshot for dt
            arma::cx_mat expKdt = expIH<cmpx>(-K_snapshot * dt);  // exp(+iK dt) — left factor for cc evolution
            arma::cx_mat cc_manual = expKdt * cc_before * expKdt.t();
            arma::mat diff = arma::abs(solver.fb.cc - cc_manual);
            cout << "  [cc diag] max|cc_tdvp - cc_manual|=" << diff.max() << "\n";
            cout << "  [cc diag] cc_tdvp[14,0]=" << solver.fb.cc(14,0)
                 << " cc_manual[14,0]=" << cc_manual(14,0) << "\n";
            // Convert cc_manual to real-space: C_raw_manual = conj(fb.rot)*cc_manual*fb.rot^T
            arma::cx_mat Q = solver.fb.rot;
            arma::cx_mat C_raw_manual = arma::conj(Q) * cc_manual * Q.st();
            cout << "  [cc diag] C_raw_manual[14,0]=" << C_raw_manual(14,0)
                 << " actual C_raw[14,0]=" << solver.fb.correlator_all()(14,0) << "\n";
        }
        {
            arma::vec ni = solver.fb.occupations_ni();
            cout << "    ni[" << solver.fb.p1 << "..." << solver.fb.p2-1 << "] = ";
            for (int k = solver.fb.p1; k < solver.fb.p2; k++) cout << ni[k] << " ";
            cout << "\n";
        }
        solver.rotateToNaturalOrbitals();
        print_corr("after rotateToNaturalOrbitals");
        {
            arma::vec ni = solver.fb.occupations_ni();
            cout << "    ni[" << solver.fb.p1 << "..." << solver.fb.p2-1 << "] = ";
            for (int k = solver.fb.p1; k < solver.fb.p2; k++) cout << ni[k] << " ";
            cout << "\n";
        }
    };

    for (int s = 1; s <= 5; s++) {
        // evolve chain by one dt step
        {
            auto sweeps = itensor::Sweeps(1);
            sweeps.maxdim() = maxdim; sweeps.cutoff() = tol;
            sweeps.niter() = 16; sweeps.noise() = 0.0;
            std::vector<double> epsilonK(15, 1e-4);
            itensor::addBasis(psi_chain, mpo_chain, epsilonK,
                              {"Cutoff", 1e-4, "Method", "DensityMatrix",
                               "KrylovOrd", 15, "DoNormalize", true,
                               "Quiet", true, "Silent", true});
            itensor::tdvp(psi_chain, mpo_chain, -cmpx(0,1)*dt, sweeps,
                          {"Truncate", true, "DoNormalize", true,
                           "Quiet", true, "Silent", true,
                           "NumCenter", 2, "ErrGoal", 1e-8});
        }
        run_iterate_substeps(s);
        cmpx Cc14_0 = chainCorr14_0();
        cout << "  >>> chain C[14,0] at t=" << s*dt << " = " << Cc14_0 << "\n";
    }

    return 0;
}
