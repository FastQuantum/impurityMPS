#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "test_ref_common.h"

#include <map>
#include <string>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

namespace {

using Solver = Fbr_dyn<Fb_mps_spin_block<cmpx>>;

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

Solver makeFbrRun(int L, double dt, double U)
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
    auto fb = Fb_mps_spin_block<cmpx>::from_slater(model.param.rot * cmpx(1, 0), ek,
                                                   model.param.nPart(), model.param.nImp());
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
    auto iter = [](Solver &f) {
        // epsilonM=0 skips the subspace expansion (nKrylov/epsilonK inert); err_goal
        // comes from the default (1e-7), to which FBR dynamics is insensitive.
        f.iterate({.epsilonM = 0});
    };
    auto corr = [](Solver &f) { return f.correlator_all(); };

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

TEST_CASE("fbr_block vs chain center reference U=0.2", "[fb_ref_block]") {
    checkChain(resultFor(0.2, "0.2"), chainTol());
}
TEST_CASE("fbr_block vs chain center reference U=0.1", "[fb_ref_block]") {
    checkChain(resultFor(0.1, "0.1"), chainTol());
}

TEST_CASE("multi-state solver with one state matches single-state solver",
          "[multi_state]")
{
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
    auto model=ImpuritySpin{{.Kmat=K0,.Umat=Umat,.impPos={2,0,1,3}}};

    auto ek=vec{model.param.Kmat.diag()};
    ek[L/2-1]=ek[L/2]=-10;
    ek[L/2-2]=ek[L/2+1]=10;
    auto fb=Fb_mps_spin_block<cmpx>::from_slater(model.param.rot*cmpx(1,0),ek,
                                                 model.param.nPart(),model.param.nImp());
    fb.tol=1e-12;

    auto incompatible=fb;
    incompatible.cc(fb.p2,fb.p2)=1.0-incompatible.cc(fb.p2,fb.p2);
    REQUIRE_THROWS_AS(Fbr_ns_dyn(model,std::vector{fb,incompatible},dt),std::invalid_argument);

    auto old_solver=Fbr_dyn(model,fb,dt);
    auto new_solver=Fbr_ns_dyn(model,std::vector{fb},dt);
    TdvpParam args {.max_bond_dim=512,.nIter_diag=8,.epsilonM=0};

    for (int step=0; step<10; ++step) {
        old_solver.iterate(args);
        new_solver.iterate(args);

        auto const& old=old_solver.fb;
        auto const& current=new_solver.states.front();
        CAPTURE(step,old.p1,old.p2,current.p1,current.p2);
        INFO("correlator error = " << arma::abs(new_solver.correlator_all()-old_solver.correlator_all()).max());
        INFO("energy error = " << std::abs(new_solver.energies.front()-old_solver.energy));

        REQUIRE(current.p1==old.p1);
        REQUIRE(current.p2==old.p2);
        REQUIRE(arma::abs(new_solver.correlator_all()-old_solver.correlator_all()).max()<1e-8);
        REQUIRE(std::abs(new_solver.energies.front()-old_solver.energy)<1e-8);
    }
}
