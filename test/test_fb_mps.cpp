#include <catch2/catch.hpp>
#include "impurityMPS/fb_mps_spin.h"
#include "impurityMPS/fb_mps.h"

using namespace arma;

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
