#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "fbr/fbr_gs.h"
#include "test_ref_common.h"

#include <map>
#include <string>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

namespace {

using Solver = Fbr_dyn;

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
        model = ImpurityParam{.Kmat = K, .Umat = Umat, .imp_pos = {0, 1, 2, 3}};
        model.to_star();
    }

    auto ek = vec{model.Kmat.diag()};
    ek[0] = ek[1] = -10;
    ek[2] = ek[3] = 10;
    auto fb = slater<cmpx>(model, ek);
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
    auto iter = [](Solver &f) { f.iterate({.max_bond_dim = 2048, .epsilon_M = 1e-4}); };
    auto corr = [](Solver &f) { return f.correlator(); };

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
    auto model = ImpurityParam{.Kmat=K0,.Umat=Umat,.imp_pos={0,1,2,3}};
    model.to_star();

    auto ek=vec{model.Kmat.diag()};
    ek[0]=ek[1]=-10;
    ek[2]=ek[3]=10;
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
        CAPTURE(step,old.n_active(),current.n_active());
        INFO("correlator error = " << arma::abs(new_solver.correlator()-old_solver.correlator()).max());
        INFO("energy error = " << std::abs(new_solver.energies.front()-old_solver.energy));
        INFO("rot error = " << arma::abs(current.rot-old.rot).max());
        INFO("cc error = " << arma::abs(current.cc-old.cc).max());
        INFO("K error = " << arma::abs(new_solver.K-old_solver.K).max());

        REQUIRE(current.n_active()==old.n_active());
        REQUIRE(arma::abs(new_solver.correlator()-old_solver.correlator()).max()<1e-8);
        REQUIRE(std::abs(new_solver.energies.front()-old_solver.energy)<1e-8);
    }
}

// A ground state coming out of Fbr_gs is no longer in the model's star frame,
// which is the basis Kip0 and the interaction-picture phases are written in.
// The dynamics takes that frame from the model, so such a state still evolves
// correctly. Taking it from the initial state instead (as it used to) leaves the
// solver evolving with a Hamiltonian written in a basis the MPS is not in, from
// the very first step. At U=0 the ground state is an eigenstate, so its
// correlator has to stay put; the broken version drifts by O(0.1) by t=1.
TEST_CASE("dynamics starting from a rotated frame", "[fbr_dyn][frame]") {
    constexpr int L=12;
    constexpr double dt=0.05;

    mat K(L,L,fill::zeros);
    for (int i=1; i<L-1; ++i) K(i,i+1)=K(i+1,i)=0.5;
    K(0,1)=K(1,0)=0.5;
    mat Umat(L,L,fill::zeros);        // U=0: the ground state is an eigenstate
    auto model = ImpurityParam{.Kmat=K,.Umat=Umat,.imp_pos={0,1}};
    model.to_star();

    auto gs=slater<double>(model);
    gs.tol=1e-12;
    auto gs_solver=Fbr_gs(model,gs);
    for (int i=0; i<60; ++i) gs_solver.iterate({.max_bond_dim=256});

    auto fb=gs_solver.fb.to_complex();
    fb.tol=1e-12;
    // the ground state really did leave the star frame
    REQUIRE(arma::abs(fb.rot-model.rot*cmpx(1,0)).max()>0.1);

    auto solver=Fbr_dyn(model,fb,dt);
    cx_mat cc0=solver.correlator();
    for (int step=0; step<20; ++step) solver.iterate({.epsilon_M=0});

    double drift=arma::abs(solver.correlator()-cc0).max();
    INFO("correlator drift of a stationary state = "<<drift);
    REQUIRE(drift<1e-3);
}
