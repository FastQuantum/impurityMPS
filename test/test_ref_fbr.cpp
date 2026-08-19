#include <catch2/catch.hpp>

#include "fbr/fbr_dyn_spin.h"
#include "test_ref_common.h"

#include <map>
#include <string>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

namespace {

// Star-geometry spin layout: bath orbitals are energy-sorted eigenmodes, so the
// FBR site -> chain index map is the discontinuous permutation below.
uvec fbrIndexToChainIndex(int L)
{
    uvec p(L);
    p[0] = L / 2 - 1;
    p[1] = L / 2;
    p[2] = L / 2 - 2;
    p[3] = L / 2 + 1;
    for (int j = 0; j < L / 2 - 2; j++) p[2 * (j + 2)] = j;
    for (int j = 0; j < L / 2 - 2; j++) p[2 * (j + 2) + 1] = L / 2 + 2 + j;
    return p;
}

// Build the SIAM FBR dynamics solver at interaction strength U (star geometry,
// active-window). Same model as example/ref/{chain,star}_dyn_siam_center.cpp.
Fbr_dyn_spin makeFbrRun(int L, double dt, double U)
{
    ImpuritySpin model;
    {
        double V = 0.1;
        mat K(L, L, fill::zeros);
        for (auto i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
        K(0, 0) = -U / 2;
        K(1, 1) = -U / 2;
        K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
        mat Umat(L, L, fill::zeros);
        Umat(0, 1) = U;
        model = ImpuritySpin{{.Kmat = K, .Umat = Umat, .impPos = {2, 0, 1, 3}}};
    }

    auto ek = vec{model.param.Kmat.diag()};
    ek[L / 2 - 1] = ek[L / 2] = -10;
    ek[L / 2 - 2] = ek[L / 2 + 1] = 10;
    auto fb = Fb_mps_spin<cmpx>::from_slater(model.param.rot * cmpx(1, 0), ek,
                                             model.param.nPart(), model.param.nImp());
    auto solver = Fbr_dyn_spin(model, fb, dt);
    solver.fb.tol = 1e-12;
    return solver;
}

// Run one FBR trajectory at U and compare against the chain reference for that U.
// Cached so the (slow) run happens once per U across all TEST_CASEs.
TrajResult const &resultFor(double U, std::string const &us)
{
    static std::map<std::string, TrajResult> cache;
    auto it = cache.find(us);
    if (it != cache.end()) return it->second;

    constexpr int L = 100;
    constexpr double dt = 0.1;
    auto fbr = makeFbrRun(L, dt, U);
    auto p = fbrIndexToChainIndex(L);
    auto iter = [](Fbr_dyn_spin &f) {
        f.iterate({.nIter_diag = 8, .err_goal = 1e-8, .epsilonM = 0e-8, .nKrylov = 15});
    };
    auto corr = [](Fbr_dyn_spin &f) { return f.correlator_all(); };

    auto ref = loadReference("chain_dyn_siam_center_U" + us + "_ref.txt");
    auto res = compareTrajectory(fbr, iter, corr, ref, p);
    return cache.emplace(us, std::move(res)).first->second;
}

// Assert every snapshot against the tolerance table.
void checkChain(TrajResult const &res, std::map<std::string, Tol> const &tol)
{
    for (auto const &[label, m] : res) {
        INFO("chain " << label << ": niMax=" << m.niMax << " ccMax=" << m.ccMax);
        REQUIRE(tol.count(label) == 1);
        REQUIRE(m.niMax < tol.at(label).first);
        REQUIRE(m.ccMax < tol.at(label).second);
    }
}

} // namespace

TEST_CASE("buildK O(L^2) matches O(L^3) reference", "[fb_ref_fbr][buildK]") {
    constexpr int L = 40;
    constexpr double dt = 0.1;
    auto fbr = makeFbrRun(L, dt, 0.2);

    arma::arma_rng::set_seed(12345);
    // The K decomposition is algebraic, valid for any rot; use a random unitary fb.rot.
    cx_mat G = cx_mat(L, L, fill::randn) + imag_1 * cx_mat(L, L, fill::randn);
    cx_mat Q, R;
    qr(Q, R, G);
    fbr.fb.rot = Q;

    for (int n : {0, 1, 5, 37, 200}) {
        fbr.nIter = n;
        cx_mat Knew = fbr.buildK();
        cx_mat Kref = fbr.buildK_reference();
        double err = abs(Knew - Kref).max();
        INFO("nIter = " << n << ", max abs error = " << err);
        REQUIRE(err < 1e-9);
    }
}

TEST_CASE("fbr vs chain center reference U=0.2", "[fb_ref_fbr]") {
    checkChain(resultFor(0.2, "0.2"), chainTol());
}
TEST_CASE("fbr vs chain center reference U=0.1", "[fb_ref_fbr]") {
    checkChain(resultFor(0.1, "0.1"), chainTol());
}
