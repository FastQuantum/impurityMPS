#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "test_ref_common.h"

#include <map>
#include <string>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

namespace {

using Solver = Fbr_dyn;

// Star-geometry spin layout: bath orbitals are energy-sorted eigenmodes, so the
// FBR site -> chain index map is the discontinuous permutation below.
uvec fbrIndexToChainIndex(int L)
{
    uvec p(L);
    p[0] = L / 2 - 1;
    p[1] = L / 2;
    p[2] = L / 2 - 2;
    p[3] = L / 2 + 1;
    // both baths are listed from the center outwards, so the up one -- which
    // runs towards index 0 -- is mirrored with respect to the dw one
    for (int j = 0; j < L / 2 - 2; j++) p[2 * (j + 2)] = L / 2 - 3 - j;
    for (int j = 0; j < L / 2 - 2; j++) p[2 * (j + 2) + 1] = L / 2 + 2 + j;
    return p;
}

// Build the SIAM FBR dynamics solver at interaction strength U (star geometry,
// active-window). Same model as test/ref/{chain,star}_dyn_siam_center.cpp.
Solver makeFbrRun(int L, double dt, double U)
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
        model = Impurity{{.Kmat = K, .Umat = Umat, .imp_pos = {2, 0, 1, 3}, .layout=spin_symmetric}};
    }

    auto ek = vec{model.param.Kmat.diag()};
    ek[L / 2 - 1] = ek[L / 2] = -10;
    ek[L / 2 - 2] = ek[L / 2 + 1] = 10;
    auto fb = slater<cmpx>(model, ek);
    auto solver = Fbr_dyn(model, fb, dt);
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
    auto iter = [](Solver &f) {
        // FBR: epsilon_M=0 skips the subspace expansion, so n_krylov/epsilon_K are
        // inert and err_goal (default 1e-7) is the only TDVP knob; FBR is insensitive
        // to it (identical to 1e-8, tolerant to 1e-6). See test/ref/fbr_dyn_tune.cpp.
        f.iterate({.n_iter_diag = 8, .epsilon_M = 0});
    };
    auto corr = [](Solver &f) { return f.correlator(); };

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

TEST_CASE("build_K O(L^2) matches O(L^3) reference", "[fb_ref_fbr][build_K]") {
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
        fbr.n_iter = n;
        cx_mat Knew = fbr.build_K();
        cx_mat Kref = fbr.build_K_reference();
        double err = abs(Knew - Kref).max();
        INFO("n_iter = " << n << ", max abs error = " << err);
        REQUIRE(err < 1e-9);
    }
}

TEST_CASE("fbr vs chain center reference U=0.2", "[fb_ref_fbr]") {
    checkChain(resultFor(0.2, "0.2"), chainTol());
}
TEST_CASE("fbr vs chain center reference U=0.1", "[fb_ref_fbr]") {
    checkChain(resultFor(0.1, "0.1"), chainTol());
}

TEST_CASE("multi-state solver with one state matches single-state solver", "[multi_state]") {
    constexpr int L=12;
    constexpr double dt=0.1;
    constexpr double U=0.2;

    mat K0(L,L,fill::zeros);
    for (int i=0; i<L-2; ++i)
        K0(i,i+2)=K0(i+2,i)=0.5;
    K0(0,0)=K0(1,1)=-U/2;
    K0(0,2)=K0(2,0)=K0(1,3)=K0(3,1)=0.1;
    mat Umat(L,L,fill::zeros);
    Umat(0,1)=U;
    auto model=Impurity{{.Kmat=K0,.Umat=Umat,.imp_pos={2,0,1,3}, .layout=spin_symmetric}};

    auto ek=vec{model.param.Kmat.diag()};
    ek[L/2-1]=ek[L/2]=-10;
    ek[L/2-2]=ek[L/2+1]=10;
    auto fb=slater<cmpx>(model, ek);
    fb.tol=1e-12;

    auto incompatible=fb;
    incompatible.cc(fb.p2,fb.p2)=1.0-incompatible.cc(fb.p2,fb.p2);
    REQUIRE_THROWS_AS(Fbr_dyn_shared(model,std::vector{fb,incompatible},dt),std::invalid_argument);

    auto old_solver=Fbr_dyn(model,fb,dt);
    auto new_solver=Fbr_dyn_shared(model,std::vector{fb},dt);
    TdvpParam args {.max_bond_dim=512,.n_iter_diag=8,.epsilon_M=0};

    for (int step=0; step<10; ++step) {
        old_solver.iterate(args);
        new_solver.iterate(args);

        auto const& old=old_solver.fb;
        auto const& current=new_solver.states.front();
        CAPTURE(step,old.p1,old.p2,current.p1,current.p2);
        INFO("correlator error = " << arma::abs(new_solver.correlator()-old_solver.correlator()).max());
        INFO("energy error = " << std::abs(new_solver.energies.front()-old_solver.energy));
        INFO("rot error = " << arma::abs(current.rot-old.rot).max());
        INFO("cc error = " << arma::abs(current.cc-old.cc).max());
        INFO("K error = " << arma::abs(new_solver.K-old_solver.K).max());

        REQUIRE(current.p1==old.p1);
        REQUIRE(current.p2==old.p2);
        REQUIRE(arma::abs(new_solver.correlator()-old_solver.correlator()).max()<1e-8);
        REQUIRE(std::abs(new_solver.energies.front()-old_solver.energy)<1e-8);
    }
}
