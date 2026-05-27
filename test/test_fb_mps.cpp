#include <catch2/catch.hpp>
#include "fbr/fb_mps_spin.h"
#include "fbr/fb_mps_spin_block.h"
#include "fbr/fb_mps.h"

using namespace arma;
using namespace fbr;

TEST_CASE("ensure_reflection_mat", "[fb_mps_spin]") {
    const int L = 6;
    mat K(L, L, fill::zeros);
    mat br(L/2, L/2, fill::randu);
    K.submat(L/2, L/2, L-1, L-1) = br;

    Fb_mps_spin<double>::ensure_reflection_mat(K);

    // K(L/2-1-i, L/2-1-j) == br(i,j)
    for (int i = 0; i < L/2; i++)
        for (int j = 0; j < L/2; j++)
            REQUIRE(K(L/2-1-i, L/2-1-j) == Approx(br(i, j)));
}

TEST_CASE("ensure_reflection_col", "[fb_mps_spin]") {
    const int L = 6;
    mat R(L, L, fill::zeros);
    mat right(L, L/2, fill::randu);
    R.cols(L/2, L-1) = right;

    Fb_mps_spin<double>::ensure_reflection_col(R);

    // R.col(L/2-1-k) == right.col(k)
    for (int k = 0; k < L/2; k++)
        REQUIRE(norm(R.col(L/2-1-k) - right.col(k)) == Approx(0).margin(1e-14));
}

// ---- interval method tests ----
// Layout for L=8, imp_size=2:
//   [ bath_up:0,1,2 | imp_up:3 | imp_dw:4 | bath_dw:5,6,7 ]

namespace {
Fb_mps_spin<double> make_fb(int L=8, int imp_size=2, int nPart=4) {
    return Fb_mps_spin<double>::from_slater(mat(L, L, fill::eye),
                                            linspace(-1.0, 1.0, L), nPart, imp_size);
}
} // namespace

TEST_CASE("interval_impurity and interval_bath", "[fb_mps_spin]") {
    auto fb = make_fb();
    REQUIRE(fb.interval_impurity_full() == std::make_pair(3, 5));
    REQUIRE(fb.interval_impurity(up)    == std::make_pair(3, 4));
    REQUIRE(fb.interval_impurity(dw)    == std::make_pair(4, 5));
    REQUIRE(fb.interval_bath(up)        == std::make_pair(0, 3));
    REQUIRE(fb.interval_bath(dw)        == std::make_pair(5, 8));
}

TEST_CASE("interval_active and interval_slater: initial window equals impurity", "[fb_mps_spin]") {
    auto fb = make_fb();
    REQUIRE(fb.interval_active_full() == std::make_pair(3, 5));
    REQUIRE(fb.interval_active(up)    == std::make_pair(3, 4));
    REQUIRE(fb.interval_active(dw)    == std::make_pair(4, 5));
    REQUIRE(fb.interval_slater(up)    == std::make_pair(0, 3));
    REQUIRE(fb.interval_slater(dw)    == std::make_pair(5, 8));
    // rotating region is empty when active window == impurity
    auto [au, bu] = fb.interval_rotating(up);
    auto [ad, bd] = fb.interval_rotating(dw);
    REQUIRE(au == bu);
    REQUIRE(ad == bd);
}

TEST_CASE("interval_active, interval_slater and interval_rotating: extended window", "[fb_mps_spin]") {
    auto fb = make_fb();
    fb.p1 = 1; fb.p2 = 7;
    REQUIRE(fb.interval_active_full() == std::make_pair(1, 7));
    REQUIRE(fb.interval_active(up)    == std::make_pair(1, 4));
    REQUIRE(fb.interval_active(dw)    == std::make_pair(4, 7));
    REQUIRE(fb.interval_slater(up)    == std::make_pair(0, 1));
    REQUIRE(fb.interval_slater(dw)    == std::make_pair(7, 8));
    REQUIRE(fb.interval_rotating(up)  == std::make_pair(1, 3));
    REQUIRE(fb.interval_rotating(dw)  == std::make_pair(5, 7));
}

// ---- extract_representative ----
// Compare extract_representative between Fb_mps_spin and Fb_mps.
//
// Setup: SIAM, L=8, imp_size=2, both describe the same physical system
//   spinless layout:  [imp_up:0 | imp_dw:1 | bath_up:2,4,6 | bath_dw:3,5,7]  (even=up, odd=down)
//   spin layout:      [bath_up:0,1,2 | imp_up:3 | imp_dw:4 | bath_dw:5,6,7]  (impurity at center)
//
// Both start with the same occupations (2 particles in each sector): imp + 1 bath orbital.
// After extract_representative(nRef=0), both should rotate their K matrices consistently,
// reflecting the same underlying physics just with different site orderings.

TEST_CASE("extract_representative: spin vs spinless", "[fb_mps_spin]") {
    const int L = 8, imp_size = 2;

    // Base kinetic matrix: 4-site (1 impurity + 3 bath) SIAM structure
    //   row/col 0: impurity, 1-3: bath
    mat Kbase = {{0.0, 1.0, 2.0, 3.0},
                 {1.0, 0.1, 0.0, 0.0},
                 {2.0, 0.0, 0.2, 0.0},
                 {3.0, 0.0, 0.0, 0.3}};

    // Spinless: L=8 with interleaved layout (even=up, odd=down)
    //   [imp_up:0 | imp_dw:1 | bath_up:2 | bath_dw:3 | bath_up:4 | bath_dw:5 | bath_up:6 | bath_dw:7]
    // Build 8x8 K from interleaved copies of Kbase
    mat Ksl(L, L, fill::zeros);
    // Map: imp (site 0 in Kbase) → sites 0(up), 1(down)
    //      bath 0 (site 1) → sites 2(up), 3(down)
    //      bath 1 (site 2) → sites 4(up), 5(down)
    //      bath 2 (site 3) → sites 6(up), 7(down)
    uvec up_sites   = {0, 2, 4, 6};  // even indices
    uvec down_sites = {1, 3, 5, 7};  // odd indices

    // Place Kbase structure for up and down sectors
    Ksl.submat(up_sites, up_sites) = Kbase;
    Ksl.submat(down_sites, down_sites) = Kbase;

    // Spin: L=8 with impurity at center
    //   [bath_up:0,1,2 | imp_up:3 | imp_dw:4 | bath_dw:5,6,7]
    // Build block-diagonal K: down block = Kbase, up block = reflected Kbase
    mat Ksp(L, L, fill::zeros);
    Ksp.submat(L/2, L/2, L-1, L-1) = Kbase;  // down block
    Fb_mps_spin<double>::ensure_reflection_mat(Ksp);  // fill in up block by reflection

    // Create Fb_mps objects with same occupations in each sector: imp + 1 bath
    auto fb_sl = Fb_mps<double>::from_slater(mat(L, L, fill::eye),
                                             vec{-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0},
                                             4, imp_size, false);

    auto fb_sp = Fb_mps_spin<double>::from_slater(mat(L, L, fill::eye),
                                                  vec{2.0, 1.0, -1.0, -2.0, -2.0, -1.0, 1.0, 2.0},
                                                  4, imp_size);

    // extract_representative rotates to natural orbitals and selects unoccupied bath
    fb_sl.extract_representative(Ksl, 0);
    fb_sp.extract_representative(Ksp, 0, /*use_active=*/false);

    vec eigs_sl = eig_sym(Ksl);
    vec eigs_sp = eig_sym(Ksp);

    REQUIRE(norm(eigs_sl - eigs_sp) < 1e-10);
}

// ---- extract_representative_final ----
// Same setup and layout as above.  After promoting orbitals with nRef=0 and nRef=1
// into the active window, extract_representative_final rotates the remaining
// active bath so the coupling to the impurity is concentrated in the leading columns.
// Both systems start with identical eigenvalues and every step is unitary, so
// eig_sym(Ksl) == eig_sym(Ksp) must hold throughout.

TEST_CASE("extract_representative_final: spin vs spinless", "[fb_mps_spin]") {
    const int L = 12, imp_size = 2, nPart = 6;

    // 1 impurity + 5 bath sites per spin
    mat Kbase = {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                 {1.0, 0.1, 0.0, 0.0, 0.0, 0.0},
                 {2.0, 0.0, 0.2, 0.0, 0.0, 0.0},
                 {3.0, 0.0, 0.0, 0.3, 0.0, 0.0},
                 {4.0, 0.0, 0.0, 0.0, 0.4, 0.0},
                 {5.0, 0.0, 0.0, 0.0, 0.0, 0.5}};

    // Spinless: [imp_up:0 | imp_dw:1 | bath_up:2,4,6,8,10 | bath_dw:3,5,7,9,11]
    mat Ksl(L, L, fill::zeros);
    uvec up_sites   = {0, 2, 4, 6, 8, 10};
    uvec down_sites = {1, 3, 5, 7, 9, 11};
    Ksl.submat(up_sites,   up_sites)   = Kbase;
    Ksl.submat(down_sites, down_sites) = Kbase;

    // Spin: [bath_up:0-4 | imp_up:5 | imp_dw:6 | bath_dw:7-11]
    mat Ksp(L, L, fill::zeros);
    Ksp.submat(L/2, L/2, L-1, L-1) = Kbase;
    Fb_mps_spin<double>::ensure_reflection_mat(Ksp);

    auto fb_sl = Fb_mps<double>::from_slater(mat(L, L, fill::eye),
                                             vec{-3.0, -3.0, -2.0, -2.0, -1.0, -1.0,
                                                  1.0,  1.0,  2.0,  2.0,  3.0,  3.0},
                                             nPart, imp_size, /*spin=*/false);

    auto fb_sp = Fb_mps_spin<double>::from_slater(mat(L, L, fill::eye),
                                                  vec{3.0, 2.0, 1.0, -1.0, -2.0, -3.0,
                                                     -3.0,-2.0,-1.0,  1.0,  2.0,  3.0},
                                                  nPart, imp_size);

    fb_sl.extract_representative(Ksl, 0);
    fb_sl.extract_representative(Ksl, 1);
    fb_sp.extract_representative(Ksp, 0, /*use_active=*/false);
    fb_sp.extract_representative(Ksp, 1, /*use_active=*/false);

    fb_sp.extract_representative_final(Ksp);
    fb_sl.extract_representative_final(Ksl, imp_size, fb_sl.nActive);

    REQUIRE(norm(eig_sym(Ksl) - eig_sym(Ksp)) < 1e-10);
}

// ---- Fb_mps_spin_block: cross-check vs Fb_mps_spin on symmetric K ----
//
// On an Sz-symmetric K (reflection-symmetric, as produced by toStar() with
// spin-reflection), the block version must reproduce the spin version's K
// spectrum after extract_representative + extract_representative_final.

TEST_CASE("Fb_mps_spin_block: extract_representative matches spin on symmetric K", "[fb_mps_spin_block]") {
    const int L = 12, imp_size = 2, nPart = 6;

    mat Kbase = {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                 {1.0, 0.1, 0.0, 0.0, 0.0, 0.0},
                 {2.0, 0.0, 0.2, 0.0, 0.0, 0.0},
                 {3.0, 0.0, 0.0, 0.3, 0.0, 0.0},
                 {4.0, 0.0, 0.0, 0.0, 0.4, 0.0},
                 {5.0, 0.0, 0.0, 0.0, 0.0, 0.5}};

    mat Ksp(L, L, fill::zeros);
    Ksp.submat(L/2, L/2, L-1, L-1) = Kbase;
    Fb_mps_spin<double>::ensure_reflection_mat(Ksp);
    mat Kbl = Ksp;

    vec ek{3.0, 2.0, 1.0, -1.0, -2.0, -3.0,
          -3.0,-2.0,-1.0,  1.0,  2.0,  3.0};
    mat rot(L, L, fill::eye);

    auto fb_sp = Fb_mps_spin<double>::from_slater(rot, ek, nPart, imp_size);
    auto fb_bl = Fb_mps_spin_block<double>::from_slater(rot, ek, nPart, imp_size);

    fb_sp.extract_representative(Ksp, 0, /*use_active=*/false);
    fb_bl.extract_representative(Kbl, 0, /*use_active=*/false);
    INFO("After extract_representative(0): norm(Ksp-Kbl)=" << norm(Ksp-Kbl)
         << " norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
         << " norm(cc diff)=" << norm(fb_sp.cc-fb_bl.cc));
    REQUIRE(norm(Ksp - Kbl) < 1e-10);
    REQUIRE(norm(fb_sp.rot - fb_bl.rot) < 1e-10);
    REQUIRE(norm(fb_sp.cc  - fb_bl.cc)  < 1e-10);

    fb_sp.extract_representative(Ksp, 1, /*use_active=*/false);
    fb_bl.extract_representative(Kbl, 1, /*use_active=*/false);
    INFO("After extract_representative(1): norm(Ksp-Kbl)=" << norm(Ksp-Kbl)
         << " norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
         << " norm(cc diff)=" << norm(fb_sp.cc-fb_bl.cc));
    REQUIRE(fb_bl.p1 == fb_sp.p1);
    REQUIRE(fb_bl.p2 == fb_sp.p2);
    REQUIRE(norm(Ksp - Kbl) < 1e-10);
    REQUIRE(norm(fb_sp.rot - fb_bl.rot) < 1e-10);
    REQUIRE(norm(fb_sp.cc  - fb_bl.cc)  < 1e-10);

    fb_sp.extract_representative_final(Ksp);
    fb_bl.extract_representative_final(Kbl);
    INFO("After extract_representative_final: norm(Ksp-Kbl)=" << norm(Ksp-Kbl)
         << " norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
         << " norm(cc diff)=" << norm(fb_sp.cc-fb_bl.cc));
    REQUIRE(norm(Ksp - Kbl) < 1e-10);
    REQUIRE(norm(fb_sp.rot - fb_bl.rot) < 1e-10);
    REQUIRE(norm(fb_sp.cc  - fb_bl.cc)  < 1e-10);
}

TEST_CASE("Fb_mps_spin_block: rotateToNaturalOrbitals matches spin on symmetric K", "[fb_mps_spin_block]") {
    const int L = 12, imp_size = 2;

    // Build a symmetric cc with some active and inactive orbitals per spin
    // Layout: bath_up:0-4 | imp_up:5 | imp_dw:6 | bath_dw:7-11
    mat cc_template(L, L, fill::zeros);
    // Slater diag (will be promoted later)
    cc_template.diag().fill(0.0);
    // Active occupations (mix near 0, 0.5, 1)
    cc_template(2,2) = cc_template(L-1-2, L-1-2) = 0.99;   // near 1 inactive
    cc_template(3,3) = cc_template(L-1-3, L-1-3) = 0.6;    // active
    cc_template(4,4) = cc_template(L-1-4, L-1-4) = 0.5;    // most active
    cc_template(5,5) = cc_template(L-1-5, L-1-5) = 0.7;    // active (impurity)
    cc_template(6,6) = cc_template(L-1-6, L-1-6) = 0.3;    // active (impurity)
    // small off-diagonal to make eigvecs non-trivial
    cc_template(3,4) = cc_template(4,3) = 0.05;
    cc_template(L-1-3, L-1-4) = cc_template(L-1-4, L-1-3) = 0.05;

    auto fb_sp = Fb_mps_spin<double>::from_slater(mat(L, L, fill::eye), linspace(-1.0,1.0,L), 6, imp_size);
    auto fb_bl = Fb_mps_spin_block<double>::from_slater(mat(L, L, fill::eye), linspace(-1.0,1.0,L), 6, imp_size);
    fb_sp.cc = cc_template;
    fb_bl.cc = cc_template;
    fb_sp.p1 = 2;  fb_sp.p2 = L-2;
    fb_bl.p1 = 2;  fb_bl.p2 = L-2;

    fb_sp.rotateToNaturalOrbitals();
    fb_bl.rotateToNaturalOrbitals();
    INFO("rotateToNaturalOrbitals: norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
         << " norm(cc diff)=" << norm(fb_sp.cc-fb_bl.cc));

    REQUIRE(fb_bl.p1 == fb_sp.p1);
    REQUIRE(fb_bl.p2 == fb_sp.p2);
    REQUIRE(norm(fb_sp.rot - fb_bl.rot) < 1e-10);
    REQUIRE(norm(fb_sp.cc  - fb_bl.cc)  < 1e-10);
}

// ---- correlator: real-space <c_i^dagger c_j> via rot ----
//
// Setup: SIAM in star geometry, [bath_up | imp_up | imp_dw | bath_dw], built the same way
// as computeKstar() in example/fbr_dyn_siam_center.cpp.
//
// Convention (fbr_param.h): rot satisfies  K_real = rot * Kstar * rot.t(),
// i.e. c_i = sum_a rot[i,a] * d_a, where d_a is the orbital living on MPS site a.
// Therefore real-space correlations:
//   <c_i^dagger c_j> = (conj(rot) * cc * rot.st())[i,j]   (= rot * cc * rot.t() for real rot)
//
// On a Slater state cc = diag(occ), this gives the exact real-space correlator analytically,
// so it serves as the ground-truth that every correlator_* function must reproduce.

namespace {
// Build a real-space SIAM kinetic matrix and its star-geometry counterpart.
// Layout: [bath_up:0..nBath-1 | imp_up:nBath | imp_dw:L/2 | bath_dw:L/2+1..L-1]  (nImp=2)
// Returns (K_real, Kstar, rot) with rot * Kstar * rot.t() == K_real.
std::tuple<mat,mat,mat> build_siam_real_and_star(int L, double V=0.1, double U=0.2) {
    const int nImp = 2;
    const int nBath = L/2 - nImp/2;

    mat K(L, L, fill::zeros);
    for (int i = 0; i < L/2-1; i++) K(i,i+1) = K(i+1,i) = 0.5;
    for (int i = L/2; i < L-1; i++) K(i,i+1) = K(i+1,i) = 0.5;
    K(nBath+nImp/2-1, nBath+nImp/2-1) = -U/2;
    K(L/2, L/2)                       = -U/2;
    K(nBath, nBath+nImp/2-1) = K(nBath+nImp/2-1, nBath) = V;
    K(L/2, L/2+nImp/2-1)     = K(L/2+nImp/2-1, L/2)     = V;

    mat Kstar(L, L, fill::zeros);
    mat rot(L, L, fill::eye);
    auto pos_up = regspace<uvec>(0, L/2-1);
    auto pos_dw = regspace<uvec>(L/2, L-1);
    for (int s : {0, 1}) {
        uvec pos      = (s==0) ? pos_up : pos_dw;
        uvec pos_bath = (s==0) ? pos.head(nBath)  : pos.tail(nBath);
        uvec pos_impu = (s==0) ? pos.tail(nImp/2) : pos.head(nImp/2);
        mat Kbath = K.submat(pos_bath, pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1, evec1, Kbath);
        uvec iek = (s==0) ? sort_index(abs(ek1), "descend") : sort_index(abs(ek1));
        mat evec = evec1.cols(iek);
        vec ek   = ek1.rows(iek);
        mat vk   = K.submat(pos_impu, pos_bath).eval() * evec;
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
    return {K, Kstar, rot};
}
} // namespace

TEST_CASE("Fb_mps_spin: real-space correlator on SIAM star matches rot*cc*rot.t()", "[fb_mps_spin][correlator]") {
    const int L = 12, nImp = 2;
    const int nBath = L/2 - nImp/2;
    auto [Kreal, Kstar, rot] = build_siam_real_and_star(L);
    REQUIRE(norm(rot * Kstar * rot.t() - Kreal, "fro") < 1e-10);

    // Half-filled Slater state in the star basis, with physical impurities forced occupied.
    vec ek = Kstar.diag();
    ek[nBath + nImp/2 - 1] = -10;  // imp up
    ek[L/2]                = -10;  // imp dw
    auto fb = Fb_mps_spin<double>::from_slater(rot, ek, L/2, nImp);

    // Ground truth: c_i = sum_a rot[i,a] d_a  =>  Corr = rot * cc * rot.t() for real rot.
    mat Corr_true = rot * fb.cc * rot.t();

    SECTION("correlator_all()") {
        mat Corr_code = fb.correlator_all();
        INFO("|Corr_code - Corr_true|_F = " << norm(Corr_code - Corr_true, "fro"));
        REQUIRE(norm(Corr_code - Corr_true, "fro") < 1e-10);
    }
    SECTION("correlator(i,j)") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            for (int j : {0, 3, 5, 6, 8, L-1})
                REQUIRE(std::abs(fb.correlator(i, j) - Corr_true(i, j)) < 1e-10);
    }
    SECTION("correlator_all_i(j) is column j of Corr_true") {
        for (int j : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_all_i(j) - Corr_true.col(j), 2) < 1e-10);
    }
    SECTION("correlator_all_j(i) is row i of Corr_true") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_all_j(i) - Corr_true.row(i).t(), 2) < 1e-10);
    }
}

TEST_CASE("Fb_mps_spin_block: real-space correlator on SIAM star matches rot*cc*rot.t()", "[fb_mps_spin_block][correlator]") {
    const int L = 12, nImp = 2;
    const int nBath = L/2 - nImp/2;
    auto [Kreal, Kstar, rot] = build_siam_real_and_star(L);
    REQUIRE(norm(rot * Kstar * rot.t() - Kreal, "fro") < 1e-10);

    vec ek = Kstar.diag();
    ek[nBath + nImp/2 - 1] = -10;
    ek[L/2]                = -10;
    auto fb = Fb_mps_spin_block<double>::from_slater(rot, ek, L/2, nImp);

    mat Corr_true = rot * fb.cc * rot.t();

    SECTION("correlator_all()") {
        REQUIRE(norm(fb.correlator_all() - Corr_true, "fro") < 1e-10);
    }
    SECTION("correlator(i,j)") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            for (int j : {0, 3, 5, 6, 8, L-1})
                REQUIRE(std::abs(fb.correlator(i, j) - Corr_true(i, j)) < 1e-10);
    }
    SECTION("correlator_all_i(j)") {
        for (int j : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_all_i(j) - Corr_true.col(j), 2) < 1e-10);
    }
    SECTION("correlator_all_j(i)") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_all_j(i) - Corr_true.row(i).t(), 2) < 1e-10);
    }
}
