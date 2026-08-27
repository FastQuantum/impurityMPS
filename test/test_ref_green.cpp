#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "test_ref_common.h"

#include <map>
#include <string>
#include <vector>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

// The impurity Green functions of the spinless IRLM, computed with the FBR
// (active-window, star geometry, interaction picture) and compared against the
// trusted chain baseline in test/ref/output/chain_green_irlm_U<U>_ref.txt,
// produced by test/ref/chain_green_irlm.cpp.
//
//     G(i,j,t) = -i <psi0| c_i(t) c_j^dag(0) |psi0> = -i <c_i^dag A(t) | B_j(t)>
//
// with A=|psi0> and B_j=c_j^dag|psi0> evolved together in one common orbital
// basis by Fbr_ns_dyn. The impurity orbitals are never rotated, so the
// real-space site i is the MPS orbital i and c_i is local there.

namespace {

// Same model as test/ref/chain_green_irlm.cpp and example/fbr_dyn_irlm.cpp.

// c_j^dag|psi0>, normalized, with the norm it had before normalizing: the states
// of one Fbr_ns_dyn share their Slater determinant, so they have to be
// normalized, and the norm goes back into G afterwards.
std::pair<Fb_mps<cmpx>, double> addParticle(Fb_mps<cmpx> const &psi0, int j)
{
    auto state = psi0;
    state.applyLocalOp("Cdag", j);
    double nrm = std::sqrt(std::real(itensor::innerC(state.psi, state.psi)));
    state.psi.normalize();
    state.update_cc();
    return {state, nrm};
}

cmpx cElement(Fb_mps<cmpx> const &A, Fb_mps<cmpx> const &B, int i)
{
    auto Ai = A;
    Ai.applyLocalOp("Cdag", i);
    return itensor::innerC(Ai.psi, B.psi);
}

// A bound per time window, as chainTol() does for the correlator references.
// Each entry covers the steps from the previous entry's time up to its own, so
// the windows are disjoint: {20.0,...} bounds 10 < t <= 20 only.
struct GreenTol { double t; double tol; };

// Measured against the committed reference, at ~2x above what is observed.
// Worse of U=0.1 and U=0.2, per window: 7.0e-5 for t<=5, 7.1e-5 for 5<t<=10,
// 2.7e-5 for 10<t<=20. The two methods do not drift apart with time -- the last
// window is the smallest because both Green functions have decayed by then
// (|G01| falls from 0.29 to ~0.12), so the absolute difference shrinks with the
// signal. At t=0 they agree to 5e-8, which the separate bound below covers.
std::vector<GreenTol> greenTol()
{
    return {{5.0, 1.5e-4}, {10.0, 1.5e-4}, {20.0, 6e-5}};
}

struct GreenError {
    std::map<double, double> byBucket;   ///< bucket time -> max deviation in it
    double atZero = 0;                   ///< t=0: G only involves the ground state
    double tMax = 0;
};

GreenError const &resultFor(std::string const &us)
{
    static std::map<std::string, GreenError> cache;
    auto it = cache.find(us);
    if (it != cache.end()) return it->second;

    constexpr int L = 100;
    constexpr double dt = 0.1;
    constexpr double V = 0.1;
    double U = std::stod(us);

    auto ref = loadGreenReference("chain_green_irlm_U" + us + "_ref.txt");
    // the reference rows are compared step by step, so they have to be on the
    // same time grid as the evolution below
    REQUIRE(ref.size() > 1);
    REQUIRE(std::abs((ref[1].t - ref[0].t) - dt) < 1e-12);

    auto model = makeIrlmModel(L, U, V);

    // The ground state comes off disk: 80 Fbr_gs sweeps at L=100 took longer
    // than everything else in this file put together, and the answer is always
    // the same. It is still checked -- at t=0 the Green functions involve only
    // the ground state, and the comparison against the chain DMRG baseline
    // below is tighter there than anywhere else.
    auto psi0 = loadFbMps<double>(findRef("fbr_green_gs_L" + std::to_string(L)
                                          + "_U" + us + ".dat")).to_complex();
    auto [B0, nrm0] = addParticle(psi0, 0);
    auto [B1, nrm1] = addParticle(psi0, 1);
    // the three states share one active window, which has to hold every orbital
    // where they differ; a tight tolerance keeps it wide enough
    psi0.tol = B0.tol = B1.tol = 1e-12;
    auto solver = Fbr_ns_dyn(model, std::vector{psi0, B0, B1}, dt);

    std::size_t nStep = ref.size();
#ifndef FBR_ENABLE_LONG_TEST
    // Default: enough of the evolution to see the two methods track each other,
    // and no more. The whole file (out to t=20) is a long test:
    // -DFBR_ENABLE_LONG_TEST=ON.
    nStep = std::min<std::size_t>(nStep, 21);
#endif

    GreenError err;
    for (std::size_t step = 0; step < nStep; step++) {
        cmpx G00 = -imag_1 * nrm0 * cElement(solver.states[0], solver.states[1], 0);
        cmpx G01 = -imag_1 * nrm1 * cElement(solver.states[0], solver.states[2], 0);
        double d = std::max(std::abs(G00 - ref[step].G00), std::abs(G01 - ref[step].G01));
        for (auto const &bucket : greenTol())
            if (ref[step].t <= bucket.t + 1e-9) {
                double &slot = err.byBucket[bucket.t];
                slot = std::max(slot, d);
                break;
            }
        if (step == 0) err.atZero = d;
        err.tMax = ref[step].t;
        if (step + 1 < nStep) solver.iterate({.epsilonM = 0});
    }
    return cache.emplace(us, err).first->second;
}

// Both sides are approximations -- the chain baseline is itself a TDVP run -- so
// these bound how far apart two methods drift, not an exact error.
void checkGreen(GreenError const &err)
{
    // At t=0 the Green functions only involve the ground state, so this compares
    // the FBR ground state with the chain DMRG one, and is much tighter than the
    // bound on the evolution that follows.
    INFO("|dG(t=0)| = " << err.atZero);
    REQUIRE(err.atZero < 1e-6);

    for (auto const &bucket : greenTol()) {
        auto it = err.byBucket.find(bucket.t);
        if (it == err.byBucket.end()) continue;      // the run did not get there
        INFO("up to t=" << bucket.t << " (run reached t=" << err.tMax
             << "): max|dG| = " << it->second);
        REQUIRE(it->second < bucket.tol);
    }
}

} // namespace

TEST_CASE("fbr green vs chain reference U=0.2", "[fb_ref_green]") {
    checkGreen(resultFor("0.2"));
}
TEST_CASE("fbr green vs chain reference U=0.1", "[fb_ref_green]") {
    checkGreen(resultFor("0.1"));
}

// ---- L=1000 regression ----
//
// A different kind of check from the ones above. There is no trusted baseline
// at L=1000 -- a real-space chain TDVP of three states on 1000 sites is out of
// reach, which is the whole point of the active-window method -- so this
// replays the FBR against its own recorded trajectory
// (test/ref/fbr_green_irlm_L1000.cpp). It cannot say the answer is right; what
// it catches is the solver behaving differently at a size the L=100 tests never
// reach, where the window machinery does the work that matters (it holds ~14
// active orbitals out of 1000 here).
//
// The reference state is a Slater determinant with both impurity orbitals
// empty, not a ground state, so these are the Green functions of that quench.
//
// What is checked: the Green functions themselves, and that the two integers
// describing the window machinery stay in the same ballpark. They are not
// required to match exactly. A different compiler or BLAS can send the
// truncation down a slightly different path and shift them by one or two
// without anything being wrong; what would matter is the window or the bond
// dimension growing out of proportion, which is what the margins below catch.
namespace {

// Must match test/ref/fbr_green_irlm_L1000.cpp, which produced the files.
constexpr int largeL = 1000;
constexpr double largeTol = 1e-10;

// Same ballpark: a quarter off, or three, whichever is looser. Enough room for
// the truncation to land somewhere slightly different, tight enough that the
// window or the bond dimension running away still fails.
bool nearEnough(int got, int want)
{
    return std::abs(got - want) <= std::max(3, want / 4);
}

void checkLargeL(std::string const &us)
{
#ifdef FBR_ENABLE_LONG_TEST
    constexpr int nSteps = 201;
#else
    constexpr int nSteps = 5;
#endif
    constexpr double dt = 0.1;
    constexpr double V = 0.1;
    double U = std::stod(us);

    auto ref = loadLargeLGreenReference("fbr_green_irlm_L" + std::to_string(largeL)
                                        + "_U" + us + ".txt", nSteps);
    auto model = makeIrlmModel(largeL, U, V);

    auto ek = vec{model.param.Kmat.diag()};
    ek[0] = ek[1] = 10;            // both impurity orbitals empty, so c^dag acts
    auto psi0 = slater<cmpx>(model, ek);
    psi0.tol = largeTol;
    auto [B0, nrm0] = addParticle(psi0, 0);
    auto [B1, nrm1] = addParticle(psi0, 1);
    B0.tol = B1.tol = largeTol;
    auto solver = Fbr_ns_dyn(model, std::vector{psi0, B0, B1}, dt);

    for (int step = 0; step < nSteps; step++) {
        cmpx G00 = -imag_1 * nrm0 * cElement(solver.states[0], solver.states[1], 0);
        cmpx G01 = -imag_1 * nrm1 * cElement(solver.states[0], solver.states[2], 0);
        int m = 0;
        for (auto const &s : solver.states) m = std::max(m, itensor::maxLinkDim(s.psi));
        int nActive = solver.states[0].nActive();
        auto const &want = ref[step];

        CAPTURE(U, step, m, nActive, want.maxBondDim, want.nActive);
        INFO("|dG00|=" << std::abs(G00 - want.G00) << " |dG01|=" << std::abs(G01 - want.G01));
        REQUIRE(want.t == Approx(step * dt).margin(1e-12));
        REQUIRE(std::abs(G00 - want.G00) < 1e-6);
        REQUIRE(std::abs(G01 - want.G01) < 1e-6);
        REQUIRE(nearEnough(m, want.maxBondDim));
        REQUIRE(nearEnough(nActive, want.nActive));

        if (step + 1 < nSteps) solver.iterate({.epsilonM = 0});
    }
}

} // namespace

TEST_CASE("fbr green L=1000 regression U=0.2", "[fb_ref_green][large_l]") {
    checkLargeL("0.2");
}
TEST_CASE("fbr green L=1000 regression U=0.1", "[fb_ref_green][large_l]") {
    checkLargeL("0.1");
}
