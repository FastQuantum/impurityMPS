#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "test_ref_common.h"

#include <map>
#include <string>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

namespace {

// Spinless (interleaved up/down) layout: the FBR site -> chain index map.
uvec fbrIndexToChainIndex(int L)
{
    uvec p(L);
    for (int k = 0; k < L / 2; k++) {
        p[2 * k]     = L / 2 - 1 - k;
        p[2 * k + 1] = L / 2 + k;
    }
    return p;
}

Fbr_dyn makeFbrRun(int L, double dt, double U)
{
    Impurity model;
    {
        double V = 0.1;
        mat K(L, L, fill::zeros);
        for (auto i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
        K(0, 0) = -U / 2;
        K(1, 1) = -U / 2;
        K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
        mat Umat(L, L, fill::zeros);
        Umat(0, 1) = U;
        model = Impurity{{.Kmat = K, .Umat = Umat, .impPos = {0, 1, 2, 3}}};
    }

    auto ek = vec{model.param.Kmat.diag()};
    ek[0] = ek[1] = -10;
    ek[2] = ek[3] = 10;
    auto fb = Fb_mps<cmpx>::from_slater(model.param.rot * cmpx(1, 0), ek,
                                        model.param.nPart(), model.param.nImp(), false);
    auto solver = Fbr_dyn(model, fb, dt);
    solver.fb.tol = 1e-12;
    return solver;
}

TrajResult const &resultFor(double U, std::string const &us)
{
    static std::map<std::string, TrajResult> cache;
    auto it = cache.find(us);
    if (it != cache.end()) return it->second;

    constexpr int L = 100;
    constexpr double dt = 0.1;
    auto fbr = makeFbrRun(L, dt, U);
    auto p = fbrIndexToChainIndex(L);
    auto iter = [](Fbr_dyn &f) { f.iterate({.max_bond_dim = 2048, .epsilonM = 1e-4}); };
    auto corr = [](Fbr_dyn &f) { return f.fb.correlator_all(); };

    auto ref = loadReference("chain_dyn_siam_center_U" + us + "_ref.txt");
    auto res = compareTrajectory(fbr, iter, corr, ref, p);
    return cache.emplace(us, std::move(res)).first->second;
}

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

TEST_CASE("fbr_ns vs chain center reference U=0.2", "[fb_ref_ns]") {
    checkChain(resultFor(0.2, "0.2"), chainTol());
}
TEST_CASE("fbr_ns vs chain center reference U=0.1", "[fb_ref_ns]") {
    checkChain(resultFor(0.1, "0.1"), chainTol());
}
