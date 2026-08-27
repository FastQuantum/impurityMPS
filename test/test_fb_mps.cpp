#include <catch2/catch.hpp>
#include "fbr/fb_mps.h"

using namespace arma;
using namespace fbr;

TEST_CASE("ensure_reflection", "[fb_mps_spin]") {
    const int L = 6;
    mat K(L, L, fill::zeros);
    mat br(L/2, L/2, fill::randu);
    K.submat(L/2, L/2, L-1, L-1) = br;

    Fb_mps<double>::ensure_reflection(K);

    // K(L/2-1-i, L/2-1-j) == br(i,j)
    for (int i = 0; i < L/2; i++)
        for (int j = 0; j < L/2; j++)
            REQUIRE(K(L/2-1-i, L/2-1-j) == Approx(br(i, j)));
}

// ---- interval method tests ----
// Layout for L=8, imp_size=2:
//   [ bath_up:0,1,2 | imp_up:3 | imp_dw:4 | bath_dw:5,6,7 ]

namespace {
Fb_mps<double> make_fb(int L=8, int imp_size=2, int n_part=4) {
    return Fb_mps<double>::from_slater(mat(L, L, fill::eye),
                                            linspace(-1.0, 1.0, L), n_part, imp_size, spin_symmetric);
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

// ---- representative planning ----
// Compare representative updates between Fb_mps_spin and Fb_mps.
//
// Setup: SIAM, L=8, imp_size=2, both describe the same physical system
//   spinless layout:  [imp_up:0 | imp_dw:1 | bath_up:2,4,6 | bath_dw:3,5,7]  (even=up, odd=down)
//   spin layout:      [bath_up:0,1,2 | imp_up:3 | imp_dw:4 | bath_dw:5,6,7]  (impurity at center)
//
// Both start with the same occupations (2 particles in each sector): imp + 1 bath orbital.
// After planning for nRef=0, both should rotate their K matrices consistently,
// reflecting the same underlying physics just with different site orderings.

TEST_CASE("plan_representative: spin vs spinless", "[fb_mps_spin]") {
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
    Fb_mps<double>::ensure_reflection(Ksp);  // fill in up block by reflection

    // Create Fb_mps objects with same occupations in each sector: imp + 1 bath
    auto fb_sl = Fb_mps<double>::from_slater(mat(L, L, fill::eye),
                                             vec{-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 2.0, 2.0},
                                             4, imp_size, leading);

    auto fb_sp = Fb_mps<double>::from_slater(mat(L, L, fill::eye),
                                                  vec{2.0, 1.0, -1.0, -2.0, -2.0, -1.0, 1.0, 2.0},
                                                  4, imp_size, spin_symmetric);

    auto update_sl=fb_sl.plan_representative(Ksl,0);
    update_sl.apply_as_basis(Ksl);
    fb_sl.apply(update_sl);

    auto update_sp=fb_sp.plan_representative(Ksp,0);
    update_sp.apply_as_basis(Ksp);
    Fb_mps<double>::ensure_reflection(Ksp);
    fb_sp.apply(update_sp);

    vec eigs_sl = eig_sym(Ksl);
    vec eigs_sp = eig_sym(Ksp);

    REQUIRE(norm(eigs_sl - eigs_sp) < 1e-10);
}

// ---- active representative planning ----
// Same setup and layout as above.  After promoting orbitals with nRef=0 and nRef=1
// into the active window, plan_active_representative rotates the remaining
// active bath so the coupling to the impurity is concentrated in the leading columns.
// Both systems start with identical eigenvalues and every step is unitary, so
// eig_sym(Ksl) == eig_sym(Ksp) must hold throughout.

TEST_CASE("plan_active_representative: spin vs spinless", "[fb_mps_spin]") {
    const int L = 12, imp_size = 2, n_part = 6;

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
    Fb_mps<double>::ensure_reflection(Ksp);

    auto fb_sl = Fb_mps<double>::from_slater(mat(L, L, fill::eye),
                                             vec{-3.0, -3.0, -2.0, -2.0, -1.0, -1.0,
                                                  1.0,  1.0,  2.0,  2.0,  3.0,  3.0},
                                             n_part, imp_size, leading);

    auto fb_sp = Fb_mps<double>::from_slater(mat(L, L, fill::eye),
                                                  vec{3.0, 2.0, 1.0, -1.0, -2.0, -3.0,
                                                     -3.0,-2.0,-1.0,  1.0,  2.0,  3.0},
                                                  n_part, imp_size, spin_symmetric);

    auto apply_sl=[&](auto const& update) {
        update.apply_as_basis(Ksl);
        fb_sl.apply(update);
    };
    auto apply_sp=[&](auto const& update) {
        update.apply_as_basis(Ksp);
        Fb_mps<double>::ensure_reflection(Ksp);
        fb_sp.apply(update);
    };

    apply_sl(fb_sl.plan_representative(Ksl,0));
    apply_sl(fb_sl.plan_representative(Ksl,1));
    apply_sp(fb_sp.plan_representative(Ksp,0));
    apply_sp(fb_sp.plan_representative(Ksp,1));

    apply_sp(fb_sp.plan_active_representative(Ksp));
    apply_sl(fb_sl.plan_active_representative(Ksl));

    REQUIRE(norm(eig_sym(Ksl) - eig_sym(Ksp)) < 1e-10);
}

TEST_CASE("Fb_mps Slater swap includes the fermionic string", "[fb_mps][orbital_update]")
{
    constexpr int L=6;
    constexpr int i=2;
    constexpr int j=5;
    auto check=[&](vec const& ek) {
        auto fb=Fb_mps<double>::from_slater(mat(L,L,fill::eye),ek,3,1, leading);
        double ni=std::real(fb.cc(i,i));
        double nj=std::real(fb.cc(j,j));

        itensor::AutoMPO ampo(fb.sites);
        ampo+=1.0,"Cdag",i+1,"C",j+1;
        ampo+=-1.0,"Cdag",j+1,"C",i+1;
        auto expected=itensor::applyMPO(itensor::toMPO(ampo),fb.psi,
                                        {"Cutoff",fb.tol,"Normalize",false});
        expected.noPrime();

        OrbitalUpdate<double> update(fb.p1,fb.p2);
        update.gates.emplace_back(i,j);
        fb.apply(update);

        auto expected_norm=itensor::innerC(expected,expected);
        REQUIRE(std::abs(itensor::innerC(expected,fb.psi)-expected_norm)<1e-12);
        REQUIRE(fb.occupations_ni2()(i)==Approx(nj).margin(1e-12));
        REQUIRE(fb.occupations_ni2()(j)==Approx(ni).margin(1e-12));
    };

    SECTION("occupied i to empty j") {
        check(vec{-3.0,-2.0,-1.0,2.0,3.0,1.0});
    }
    SECTION("occupied j to empty i") {
        check(vec{-3.0,3.0,2.0,-2.0,1.0,-1.0});
    }
}

// ---- Fb_mps_spin_block: cross-check vs Fb_mps_spin on symmetric K ----
//
// On an Sz-symmetric K (reflection-symmetric, as produced by to_star() with
// spin-reflection), the block version must reproduce the spin version's K
// spectrum after representative and active-representative planning.

TEST_CASE("Fb_mps_spin_block: representative plans match spin on symmetric K", "[fb_mps_spin_block]") {
    const int L = 12, imp_size = 2, n_part = 6;

    mat Kbase = {{0.0, 1.0, 2.0, 3.0, 4.0, 5.0},
                 {1.0, 0.1, 0.0, 0.0, 0.0, 0.0},
                 {2.0, 0.0, 0.2, 0.0, 0.0, 0.0},
                 {3.0, 0.0, 0.0, 0.3, 0.0, 0.0},
                 {4.0, 0.0, 0.0, 0.0, 0.4, 0.0},
                 {5.0, 0.0, 0.0, 0.0, 0.0, 0.5}};

    mat Ksp(L, L, fill::zeros);
    Ksp.submat(L/2, L/2, L-1, L-1) = Kbase;
    Fb_mps<double>::ensure_reflection(Ksp);
    mat Kbl = Ksp;

    vec ek{3.0, 2.0, 1.0, -1.0, -2.0, -3.0,
          -3.0,-2.0,-1.0,  1.0,  2.0,  3.0};
    mat rot(L, L, fill::eye);

    auto fb_sp = Fb_mps<double>::from_slater(rot, ek, n_part, imp_size, spin_symmetric);
    auto fb_bl = Fb_mps<double>::from_slater(rot, ek, n_part, imp_size, spin_block);

    auto apply_sp=[&](auto const& update) {
        update.apply_as_basis(Ksp);
        Fb_mps<double>::ensure_reflection(Ksp);
        fb_sp.apply(update);
    };
    auto apply_bl=[&](auto const& update) {
        update.apply_as_basis(Kbl);
        fb_bl.apply(update);
    };

    apply_sp(fb_sp.plan_representative(Ksp,0));
    apply_bl(fb_bl.plan_representative(Kbl,0));
    INFO("After plan_representative(0): norm(Ksp-Kbl)=" << norm(Ksp-Kbl)
         << " norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
         << " norm(cc diff)=" << norm(fb_sp.cc-fb_bl.cc));
    REQUIRE(norm(Ksp - Kbl) < 1e-10);
    REQUIRE(norm(fb_sp.rot - fb_bl.rot) < 1e-10);
    REQUIRE(norm(fb_sp.cc  - fb_bl.cc)  < 1e-10);

    apply_sp(fb_sp.plan_representative(Ksp,1));
    apply_bl(fb_bl.plan_representative(Kbl,1));
    INFO("After plan_representative(1): norm(Ksp-Kbl)=" << norm(Ksp-Kbl)
         << " norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
         << " norm(cc diff)=" << norm(fb_sp.cc-fb_bl.cc));
    REQUIRE(fb_bl.p1 == fb_sp.p1);
    REQUIRE(fb_bl.p2 == fb_sp.p2);
    REQUIRE(norm(Ksp - Kbl) < 1e-10);
    REQUIRE(norm(fb_sp.rot - fb_bl.rot) < 1e-10);
    REQUIRE(norm(fb_sp.cc  - fb_bl.cc)  < 1e-10);

    apply_sp(fb_sp.plan_active_representative(Ksp));
    apply_bl(fb_bl.plan_active_representative(Kbl));
    INFO("After plan_active_representative: norm(Ksp-Kbl)=" << norm(Ksp-Kbl)
         << " norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
         << " norm(cc diff)=" << norm(fb_sp.cc-fb_bl.cc));
    REQUIRE(norm(Ksp - Kbl) < 1e-10);
    REQUIRE(norm(fb_sp.rot - fb_bl.rot) < 1e-10);
    REQUIRE(norm(fb_sp.cc  - fb_bl.cc)  < 1e-10);
}

TEST_CASE("Fb_mps_spin_block: natural-orbital plans match spin on symmetric K", "[fb_mps_spin_block]") {
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

    auto fb_sp = Fb_mps<double>::from_slater(mat(L, L, fill::eye), linspace(-1.0,1.0,L), 6, imp_size, spin_symmetric);
    auto fb_bl = Fb_mps<double>::from_slater(mat(L, L, fill::eye), linspace(-1.0,1.0,L), 6, imp_size, spin_block);
    fb_sp.cc = cc_template;
    fb_bl.cc = cc_template;
    fb_sp.p1 = 2;  fb_sp.p2 = L-2;
    fb_bl.p1 = 2;  fb_bl.p2 = L-2;

    fb_sp.apply(fb_sp.plan_natural_orbitals(fb_sp.cc));
    fb_bl.apply(fb_bl.plan_natural_orbitals(fb_bl.cc));
    INFO("plan_natural_orbitals: norm(rot diff)=" << norm(fb_sp.rot-fb_bl.rot)
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
// Convention (impurity_param.h): rot satisfies  K_real = rot * Kstar * rot.t(),
// i.e. c_i = sum_a rot[i,a] * d_a, where d_a is the orbital living on MPS site a.
// Therefore real-space correlations:
//   <c_i^dagger c_j> = (conj(rot) * cc * rot.st())[i,j]   (= rot * cc * rot.t() for real rot)
//
// On a Slater state cc = diag(occ), this gives the exact real-space correlator analytically,
// so it serves as the ground-truth that every correlator_* function must reproduce.

namespace {
// Build a real-space SIAM kinetic matrix and its star-geometry counterpart.
// Layout: [bath_up:0..nBath-1 | imp_up:nBath | imp_dw:L/2 | bath_dw:L/2+1..L-1]  (n_imp=2)
// Returns (K_real, Kstar, rot) with rot * Kstar * rot.t() == K_real.
std::tuple<mat,mat,mat> build_siam_real_and_star(int L, double V=0.1, double U=0.2) {
    const int n_imp = 2;
    const int nBath = L/2 - n_imp/2;

    mat K(L, L, fill::zeros);
    for (int i = 0; i < L/2-1; i++) K(i,i+1) = K(i+1,i) = 0.5;
    for (int i = L/2; i < L-1; i++) K(i,i+1) = K(i+1,i) = 0.5;
    K(nBath+n_imp/2-1, nBath+n_imp/2-1) = -U/2;
    K(L/2, L/2)                       = -U/2;
    K(nBath, nBath+n_imp/2-1) = K(nBath+n_imp/2-1, nBath) = V;
    K(L/2, L/2+n_imp/2-1)     = K(L/2+n_imp/2-1, L/2)     = V;

    mat Kstar(L, L, fill::zeros);
    mat rot(L, L, fill::eye);
    auto pos_up = regspace<uvec>(0, L/2-1);
    auto pos_dw = regspace<uvec>(L/2, L-1);
    for (int s : {0, 1}) {
        uvec pos      = (s==0) ? pos_up : pos_dw;
        uvec pos_bath = (s==0) ? pos.head(nBath)  : pos.tail(nBath);
        uvec pos_impu = (s==0) ? pos.tail(n_imp/2) : pos.head(n_imp/2);
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
    const int L = 12, n_imp = 2;
    const int nBath = L/2 - n_imp/2;
    auto [Kreal, Kstar, rot] = build_siam_real_and_star(L);
    REQUIRE(norm(rot * Kstar * rot.t() - Kreal, "fro") < 1e-10);

    // Half-filled Slater state in the star basis, with physical impurities forced occupied.
    vec ek = Kstar.diag();
    ek[nBath + n_imp/2 - 1] = -10;  // imp up
    ek[L/2]                = -10;  // imp dw
    auto fb = Fb_mps<double>::from_slater(rot, ek, L/2, n_imp, spin_symmetric);

    // Ground truth: c_i = sum_a rot[i,a] d_a  =>  Corr = rot * cc * rot.t() for real rot.
    mat Corr_true = rot * fb.cc * rot.t();

    SECTION("correlator()") {
        mat Corr_code = fb.correlator();
        INFO("|Corr_code - Corr_true|_F = " << norm(Corr_code - Corr_true, "fro"));
        REQUIRE(norm(Corr_code - Corr_true, "fro") < 1e-10);
    }
    SECTION("correlator(i,j)") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            for (int j : {0, 3, 5, 6, 8, L-1})
                REQUIRE(std::abs(fb.correlator(i, j) - Corr_true(i, j)) < 1e-10);
    }
    SECTION("correlator_col(j) is column j of Corr_true") {
        for (int j : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_col(j) - Corr_true.col(j), 2) < 1e-10);
    }
    SECTION("correlator_row(i) is row i of Corr_true") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_row(i) - Corr_true.row(i).t(), 2) < 1e-10);
    }
}

TEST_CASE("Fb_mps_spin_block: real-space correlator on SIAM star matches rot*cc*rot.t()", "[fb_mps_spin_block][correlator]") {
    const int L = 12, n_imp = 2;
    const int nBath = L/2 - n_imp/2;
    auto [Kreal, Kstar, rot] = build_siam_real_and_star(L);
    REQUIRE(norm(rot * Kstar * rot.t() - Kreal, "fro") < 1e-10);

    vec ek = Kstar.diag();
    ek[nBath + n_imp/2 - 1] = -10;
    ek[L/2]                = -10;
    auto fb = Fb_mps<double>::from_slater(rot, ek, L/2, n_imp, spin_block);

    mat Corr_true = rot * fb.cc * rot.t();

    SECTION("correlator()") {
        REQUIRE(norm(fb.correlator() - Corr_true, "fro") < 1e-10);
    }
    SECTION("correlator(i,j)") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            for (int j : {0, 3, 5, 6, 8, L-1})
                REQUIRE(std::abs(fb.correlator(i, j) - Corr_true(i, j)) < 1e-10);
    }
    SECTION("correlator_col(j)") {
        for (int j : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_col(j) - Corr_true.col(j), 2) < 1e-10);
    }
    SECTION("correlator_row(i)") {
        for (int i : {0, 3, 5, 6, 8, L-1})
            REQUIRE(norm(fb.correlator_row(i) - Corr_true.row(i).t(), 2) < 1e-10);
    }
}

namespace {
template<class Fb>
void check_complex_correlators(Fb const& fb, cx_mat const& expected)
{
    REQUIRE(norm(fb.correlator()-expected,"fro")<1e-12);
    for (int i : {0,2,5}) {
        for (int j : {1,3,5})
            REQUIRE(std::abs(fb.correlator(i,j)-expected(i,j))<1e-12);
        REQUIRE(norm(fb.correlator_row(i)-expected.row(i).st(),2)<1e-12);
    }
    for (int j : {1,3,5})
        REQUIRE(norm(fb.correlator_col(j)-expected.col(j),2)<1e-12);
}
} // namespace

TEST_CASE("complex frames use simple transpose in real-space correlators",
          "[fb_mps][fb_mps_spin][fb_mps_spin_block][correlator][complex]")
{
    arma::arma_rng::set_seed(8142);
    constexpr int L=6;
    cx_mat raw=cx_mat(L,L,fill::randn)+imag_1*cx_mat(L,L,fill::randn);
    cx_mat rot,R;
    qr(rot,R,raw);
    vec ek={-3.0,-2.0,-1.0,1.0,2.0,3.0};

    auto fb=Fb_mps<cmpx>::from_slater(rot,ek,L/2,2, leading);
    auto fb_spin=Fb_mps<cmpx>::from_slater(rot,ek,L/2,2, spin_symmetric);
    auto fb_block=Fb_mps<cmpx>::from_slater(rot,ek,L/2,2, spin_block);

    // c_i=sum_a rot(i,a)d_a. For complex rot the two transposes are not
    // interchangeable: Qinv=rot.st(), then C=Qinv.t()*cc*Qinv.
    cx_mat expected=arma::conj(rot)*fb.cc*rot.st();
    cx_mat real_only_formula=rot*fb.cc*rot.t();
    REQUIRE(norm(expected-real_only_formula,"fro")>1e-3);

    check_complex_correlators(fb,expected);
    check_complex_correlators(fb_spin,expected);
    check_complex_correlators(fb_block,expected);
}

// ---- apply_local_op ----
// With an identity rotation the real-space site maps directly onto the MPS orbital.
// We check that local operators act on impurity sites and reject Slater sites.

TEST_CASE("Fb_mps apply_local_op: N on impurity and active-window guard", "[fb_mps][apply_local_op]") {
    const int L = 8, imp_size = 2, n_part = 4;
    // ascending ek occupies the 4 lowest -> sites 0,1,2,3; impurity sites 0,1 occupied.
    vec ek = {-2, -2, 1, 1, 2, 2, 3, 3};
    auto fb = Fb_mps<double>::from_slater(mat(L, L, fill::eye), ek, n_part, imp_size, leading);

    REQUIRE(std::real(fb.cc(0,0)) == Approx(1.0));
    fb.apply_local_op("N", 0);                                   // occupied impurity, in active window
    REQUIRE(std::real(fb.cc(0,0)) == Approx(1.0));
    REQUIRE_THROWS_AS(fb.apply_local_op("N", 5), std::invalid_argument);  // Slater site rejected
}

TEST_CASE("Fb_mps_spin apply_local_op: N/Cdag and active-window guard", "[fb_mps_spin][apply_local_op]") {
    const int L = 8, imp_size = 2, n_part = 4;
    // ascending ek occupies {2,3,5,6}; impurity sites are 3 (up) and 4 (dw):
    // site 3 occupied, site 4 empty.
    vec ek = {2, 1, -3, -2, 5, -1, 0.5, 3};
    auto fb = Fb_mps<double>::from_slater(mat(L, L, fill::eye), ek, n_part, imp_size, spin_symmetric);

    REQUIRE(fb.interval_active_full() == std::make_pair(3, 5));
    REQUIRE(fb.occupations_ni()(3) == Approx(1.0));
    REQUIRE(fb.occupations_ni()(4) == Approx(0.0).margin(1e-12));

    fb.apply_local_op("N", 3);                                   // occupied impurity, unchanged
    REQUIRE(fb.occupations_ni()(3) == Approx(1.0));

    fb.apply_local_op("Cdag", 4);                                // create on empty impurity (dw)
    REQUIRE(fb.occupations_ni()(4) == Approx(1.0));

    REQUIRE_THROWS_AS(fb.apply_local_op("N", 0), std::invalid_argument);  // Slater site rejected
}

TEST_CASE("Fb_mps_spin_block apply_local_op: N and active-window guard", "[fb_mps_spin_block][apply_local_op]") {
    const int L = 8, imp_size = 2, n_part = 4;
    vec ek = {2, 1, -3, -2, 5, -1, 0.5, 3};
    auto fb = Fb_mps<double>::from_slater(mat(L, L, fill::eye), ek, n_part, imp_size, spin_block);

    REQUIRE(fb.occupations_ni()(3) == Approx(1.0));
    fb.apply_local_op("N", 3);                                   // occupied impurity, unchanged
    REQUIRE(fb.occupations_ni()(3) == Approx(1.0));
    REQUIRE_THROWS_AS(fb.apply_local_op("N", 0), std::invalid_argument);  // Slater site rejected
}

TEST_CASE("Fb_mps complex apply_local_op updates the active correlator",
          "[fb_mps][apply_local_op][complex]")
{
    constexpr int L=6;
    cx_mat rot(L,L,fill::eye);
    rot(2,2)=cmpx(0,1);
    rot(3,3)=std::exp(cmpx(0,0.37));
    vec ek={1.0,-2.0,-1.0,2.0,3.0,4.0};
    auto fb=Fb_mps<cmpx>::from_slater(rot,ek,2,2, leading);

    REQUIRE(std::real(fb.cc(0,0))==Approx(0.0).margin(1e-12));
    fb.apply_local_op("Cdag",0);
    REQUIRE(std::real(fb.cc(0,0))==Approx(1.0));
    REQUIRE_THROWS_AS(fb.apply_local_op("N",2),std::invalid_argument);
}
