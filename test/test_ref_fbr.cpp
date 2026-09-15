#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "test_ref_common.h"

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

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

// The O(L^3) conjugation that build_K() replaces with a rank-2*n_imp update:
// rot^dag * Kip0 * rot, with the interaction-picture phase built as a dense
// matrix. Only this test needs it, so it lives here rather than in the solver.
cx_mat referenceK(Solver const& s)
{
    int L = s.param.length();
    cx_mat exp_ih(L, L, fill::eye);
    if (s.n_iter > 0)
        exp_ih.submat(s.bath_pos, s.bath_pos) = exp_iH<cmpx>(s.Kbath * (double(s.n_iter) * s.dt));
    cx_mat rot = exp_ih * s.rot_star.t() * s.fb.rot;
    return rot.t() * s.Kip0 * rot;
}

// Build the SIAM FBR dynamics solver at interaction strength U (star geometry,
// active-window). Same model as test/ref/{chain,star}_dyn_siam_center.cpp.
Solver makeFbrRun(int L, double dt, double U)
{
    ImpurityParam model;
    {
        double V = 0.1;
        mat K(L, L, fill::zeros);
        for (auto i = 0; i < L - 2; i++) K(i, i + 2) = K(i + 2, i) = 0.5;
        K(0, 0) = -U / 2;
        K(1, 1) = -U / 2;
        K(0, 2) = K(2, 0) = K(1, 3) = K(3, 1) = V;
        mat Umat(L, L, fill::zeros);
        Umat(0, 1) = U;
        model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {2, 0, 1, 3}, .layout=spin_symmetric};
        model.to_star();
    }

    auto ek = vec{model.Kmat.diag()};
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

// ---- L=1000 quench coverage (ported from the retired `unify` branch) ----
//
// The reference files test/ref/output/fbr_dyn_siam_L1000_U*.txt hold one compact
// row per time step of the SIAM impurity quench (impurity forced to |1100>, then
// evolved). Reading only the first few steps keeps this a smoke test that the
// solver still scales to L=1000; the full trajectories are kept for manual and
// longer comparisons. Regenerate with example/fbr_dyn_siam.cpp.

struct LargeLReferenceRow {
    double time;
    int maxBondDim;
    double n0;
    double n1;
    int nActive;
};

std::vector<LargeLReferenceRow> loadLargeLReference(std::string const &name, int nSteps)
{
    std::ifstream in(findRef(name));
    std::string header;
    std::getline(in, header);
    if (header != "time m <n0> <n1>  nActive time(s)")
        throw std::runtime_error("invalid L=1000 reference header in " + name);

    std::vector<LargeLReferenceRow> rows;
    rows.reserve(nSteps);
    for (int step = 0; step < nSteps; ++step) {
        LargeLReferenceRow row;
        double wallTime = 0;
        if (!(in >> row.time >> row.maxBondDim >> row.n0 >> row.n1 >> row.nActive
                 >> wallTime))
            throw std::runtime_error("not enough rows in L=1000 reference " + name);
        rows.push_back(row);
    }
    return rows;
}

void checkLargeL(double U, std::string const &us)
{
    // Five steps for each U make the two L=1000 cases together take roughly as
    // long as the existing default L=100 reference group.
    constexpr int L = 1000;
    constexpr int nSteps = 5;
    constexpr double dt = 0.1;
    auto ref = loadLargeLReference("fbr_dyn_siam_L1000_U" + us + ".txt", nSteps);
    auto fbr = makeFbrRun(L, dt, U);
    fbr.fb.tol = 1e-10; // Match the default tol used to produce the reference files.

    for (int step = 0; step < nSteps; ++step) {
        fbr.iterate({.epsilon_M = 0});
        auto const &expected = ref[step];
        auto ni = fbr.fb.occupations_ni();
        double n0 = ni[L / 2];
        double n1 = ni[L / 2 + 1];
        int m = itensor::maxLinkDim(fbr.fb.psi);

        CAPTURE(U, step, n0, n1, expected.n0, expected.n1, m, expected.maxBondDim,
                fbr.fb.n_active(), expected.nActive);
        REQUIRE(expected.time == Approx((step + 1) * dt).margin(1e-12));
        REQUIRE(n0 == Approx(expected.n0).margin(1e-9));
        REQUIRE(n1 == Approx(expected.n1).margin(1e-9));
        REQUIRE(m == expected.maxBondDim);
        REQUIRE(fbr.fb.n_active() == expected.nActive);
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
        cx_mat Kref = referenceK(fbr);
        double err = abs(Knew - Kref).max();
        INFO("n_iter = " << n << ", max abs error = " << err);
        REQUIRE(err < 1e-9);
    }
}

TEST_CASE("O(L^2) correlator elements, rows and columns match the full correlator",
          "[fb_ref_fbr][correlator]") {
    constexpr int L = 40;
    constexpr double dt = 0.1;
    auto fbr = makeFbrRun(L, dt, 0.2);

    // Algebraic identities, valid for any frame and any cc: a random unitary
    // fb.rot and a random (not even Hermitian) cc.
    arma::arma_rng::set_seed(4321);
    cx_mat G = cx_mat(L, L, fill::randn) + imag_1 * cx_mat(L, L, fill::randn);
    cx_mat Q, R;
    qr(Q, R, G);
    fbr.fb.rot = Q;
    fbr.fb.cc = cx_mat(L, L, fill::randn) + imag_1 * cx_mat(L, L, fill::randn);

    for (int n : {0, 1, 5, 37}) {
        fbr.n_iter = n;
        cx_mat full = fbr.correlator();   // through the dense effective_rot
        for (int i : {0, 1, 2, L / 2, L - 1}) {
            INFO("n_iter = " << n << ", i = " << i);
            REQUIRE(norm(fbr.correlator_col(i) - full.col(i), "inf") < 1e-10);
            REQUIRE(norm(fbr.correlator_row(i) - full.row(i).st(), "inf") < 1e-10);
            for (int j : {0, 3, L - 2})
                REQUIRE(std::abs(fbr.correlator(i, j) - full(i, j)) < 1e-10);
        }
    }
}

TEST_CASE("fbr vs chain center reference U=0.2", "[fb_ref_fbr]") {
    checkChain(resultFor(0.2, "0.2"), chainTol());
}
TEST_CASE("fbr vs chain center reference U=0.1", "[fb_ref_fbr]") {
    checkChain(resultFor(0.1, "0.1"), chainTol());
}

TEST_CASE("fbr L=1000 quench reference U=0.2", "[fb_ref_fbr][large_l]") {
    checkLargeL(0.2, "0.2");
}
TEST_CASE("fbr L=1000 quench reference U=0.1", "[fb_ref_fbr][large_l]") {
    checkLargeL(0.1, "0.1");
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
    auto model = ImpurityParam{.Kmat=K0,.Umat=Umat,.imp_pos={2,0,1,3}, .layout=spin_symmetric};
    model.to_star();

    auto ek=vec{model.Kmat.diag()};
    ek[L/2-1]=ek[L/2]=-10;
    ek[L/2-2]=ek[L/2+1]=10;
    auto fb=slater<cmpx>(model, ek);
    fb.tol=1e-12;

    auto incompatible=fb;
    incompatible.cc(fb.active.b,fb.active.b)=1.0-incompatible.cc(fb.active.b,fb.active.b);
    REQUIRE_THROWS_AS(Fbr_dyn_shared(model,std::vector{fb,incompatible},dt),std::invalid_argument);

    auto old_solver=Fbr_dyn(model,fb,dt);
    auto new_solver=Fbr_dyn_shared(model,std::vector{fb},dt);
    TdvpParam args {.max_bond_dim=512,.n_iter_diag=8,.epsilon_M=0};

    for (int step=0; step<10; ++step) {
        old_solver.iterate(args);
        new_solver.iterate(args);

        auto const& old=old_solver.fb;
        auto const& current=new_solver.states.front();
        CAPTURE(step,old.active.a,old.active.b,current.active.a,current.active.b);
        INFO("correlator error = " << arma::abs(new_solver.correlator()-old_solver.correlator()).max());
        INFO("energy error = " << std::abs(new_solver.energies.front()-old_solver.energy));
        INFO("rot error = " << arma::abs(current.rot-old.rot).max());
        INFO("cc error = " << arma::abs(current.cc-old.cc).max());
        INFO("K error = " << arma::abs(new_solver.K-old_solver.K).max());

        REQUIRE(current.active.a==old.active.a);
        REQUIRE(current.active.b==old.active.b);
        REQUIRE(arma::abs(new_solver.correlator()-old_solver.correlator()).max()<1e-8);
        REQUIRE(std::abs(new_solver.energies.front()-old_solver.energy)<1e-8);
    }
}

// spin_symmetric evolves only the dw sector and mirrors it onto up, so a state
// that is not its own mirror image would silently get the dw correlators. Both
// solvers must refuse it, and spin_block must take it.
TEST_CASE("spin_symmetric refuses a spin-polarized state", "[spin_guard]") {
    constexpr int L=8;
    constexpr double dt=0.1;
    constexpr double U=0.2;

    auto model=makeSiamModel(L,U,0.1);
    int m=L/2;                      // impurity orbitals at m-1 (up) and m (dw)
    auto ek=vec{model.Kmat.diag()};
    ek[m-1]=ek[m]=-10;
    auto symmetric=slater<cmpx>(model,ek);
    ek[m-1]=10;                     // up impurity empty, dw full
    auto polarized=slater<cmpx>(model,ek);

    REQUIRE_NOTHROW(Fbr_dyn(model,symmetric,dt));
    REQUIRE_THROWS_AS(Fbr_dyn(model,polarized,dt),std::invalid_argument);
    REQUIRE_THROWS_AS(Fbr_dyn_shared(model,std::vector{polarized},dt),std::invalid_argument);
    // a slave is mirrored as well as the master
    auto polarized_slave=symmetric;
    polarized_slave.cc(m-1,m-1)=0;
    REQUIRE_THROWS_AS(Fbr_dyn_shared(model,std::vector{symmetric,polarized_slave},dt),
                      std::invalid_argument);

    model.layout=polarized.layout=spin_block;
    REQUIRE_NOTHROW(Fbr_dyn(model,polarized,dt));
}
