#include <catch2/catch.hpp>

#include "fbr/fbr_dyn.h"
#include "test_ref_common.h"

#include <map>
#include <string>
#include <vector>

using namespace arma;
using namespace fbr;
using namespace fbrtest;

// Impurity Green functions computed with the FBR (active-window, star geometry,
// interaction picture) and compared against the trusted chain baselines of
// test/ref/, at the usual L=100, V=0.1 and U=0.1, 0.2:
//
//   SIAM  G00 against output/chain_green_siam_L100_U<U>.dat (chain_green_siam.cpp)
//   IRLM  G00, G01 against output/chain_green_irlm_U<U>_ref.txt (chain_green_irlm.cpp)
//
//     G(i,j,t) = -i <psi0| c_i(t) c_j^dag(0) |psi0> = -i <c_i^dag psi0(t) | B_j(t)>
//
// with B_j=c_j^dag|psi0>. Each element is one Fbr_dyn_shared run of the two
// states {B_j, psi0}: the excitation is the master, whose natural orbitals the
// shared basis follows. The impurity orbitals are never rotated, so the
// real-space site i is the MPS orbital i and c_i is local there.

namespace {

// c_j^dag|psi0>, normalized, with the norm it had before normalizing: the states
// of one Fbr_dyn_shared share their Slater determinant, so they have to be
// normalized, and the norm goes back into G afterwards.
std::pair<Fb_mps<cmpx>, double> addParticle(Fb_mps<cmpx> const &psi0, int j)
{
    auto state = psi0;
    state.apply_local_op("Cdag", j);
    double nrm = std::sqrt(std::real(itensor::innerC(state.psi, state.psi)));
    state.psi.normalize();
    state.update_cc();
    return {state, nrm};
}

cmpx cElement(Fb_mps<cmpx> const &A, Fb_mps<cmpx> const &B, int i)
{
    auto Ai = A;
    Ai.apply_local_op("Cdag", i);
    return itensor::innerC(Ai.psi, B.psi);
}

// A bound per time window, as chainTol() does for the correlator references.
// Each entry covers the steps from the previous entry's time up to its own, so
// the windows are disjoint: {20.0,...} bounds 10 < t <= 20 only.
struct GreenTol { double t; double tol; };

// IRLM, measured against the committed reference, at ~2x above what is
// observed. Worse of U=0.1 and U=0.2, per window: 7.0e-5 for t<=5, 7.1e-5 for
// 5<t<=10, 2.8e-5 for 10<t<=20 (the same with the master-slave basis as with
// the earlier averaged one). The two methods do not drift apart with time --
// the last window is the smallest because both Green functions have decayed by
// then (|G01| falls from 0.29 to ~0.12), so the absolute difference shrinks with
// the signal. At t=0 they agree to 5e-8, which the separate bound below covers.
std::vector<GreenTol> irlmTol()
{
    return {{5.0, 1.5e-4}, {10.0, 1.5e-4}, {20.0, 6e-5}};
}

// SIAM, the same way: ~2x above the worse of U=0.1 and U=0.2 per window, which
// is 1.8e-5 for t<=5, 1.8e-5 for 5<t<=10 and 1.6e-5 for 10<t<=20. At t=0 the
// cached ground state agrees with the chain DMRG one to 5e-11 (U=0.1) and 2e-9
// (U=0.2).
std::vector<GreenTol> siamTol()
{
    return {{5.0, 4e-5}, {10.0, 4e-5}, {20.0, 3.5e-5}};
}

struct GreenError {
    std::map<double, double> byBucket;   ///< bucket time -> max deviation in it
    double atZero = 0;                   ///< t=0: G only involves the ground state
    double tMax = 0;
};

// How many reference rows to compare. Default: enough of the evolution to see
// the two methods track each other (t<=2). The long test goes to t=20:
// -DFBR_ENABLE_LONG_TEST=ON.
std::size_t stepsToCompare(std::size_t inFile)
{
#ifdef FBR_ENABLE_LONG_TEST
    return std::min<std::size_t>(inFile, 201);
#else
    return std::min<std::size_t>(inFile, 21);
#endif
}

void record(GreenError &err, std::vector<GreenTol> const &tols, std::size_t step,
            double t, double d)
{
    for (auto const &bucket : tols)
        if (t <= bucket.t + 1e-9) {
            double &slot = err.byBucket[bucket.t];
            slot = std::max(slot, d);
            break;
        }
    if (step == 0) err.atZero = d;
    err.tMax = t;
}

// The reference rows are compared step by step, so they have to be on the same
// time grid as the evolution.
void requireGrid(std::vector<GreenSample> const &ref, double dt)
{
    REQUIRE(ref.size() > 1);
    REQUIRE(std::abs((ref[1].t - ref[0].t) - dt) < 1e-12);
}

GreenError const &irlmResult(std::string const &us)
{
    static std::map<std::string, GreenError> cache;
    auto it = cache.find(us);
    if (it != cache.end()) return it->second;

    constexpr int L = 100;
    constexpr double dt = 0.1;
    constexpr double V = 0.1;
    double U = std::stod(us);

    auto ref = loadGreenReference("chain_green_irlm_U" + us + "_ref.txt");
    requireGrid(ref, dt);
    auto model = makeIrlmModel(L, U, V);

    // The ground state comes off disk (test/ref/fbr_green_gs.cpp): 80 Fbr_gs
    // sweeps at L=100 took longer than everything else in this file put
    // together, and the answer is always the same. It is still checked -- at t=0
    // the Green functions involve only the ground state, and the comparison
    // against the chain DMRG baseline below is tighter there than anywhere else.
    auto psi0 = loadFbMps<double>(findRef("fbr_green_gs_L" + std::to_string(L)
                                          + "_U" + us + ".dat")).to_complex();
    auto [B0, nrm0] = addParticle(psi0, 0);
    auto [B1, nrm1] = addParticle(psi0, 1);
    // a tight tolerance keeps the window wide enough to hold the orbitals where
    // psi0 and the excitation differ
    psi0.tol = B0.tol = B1.tol = 1e-12;
    auto solver0 = Fbr_dyn_shared(model, std::vector{B0, psi0}, dt);
    auto solver1 = Fbr_dyn_shared(model, std::vector{B1, psi0}, dt);

    GreenError err;
    std::size_t nStep = stepsToCompare(ref.size());
    for (std::size_t step = 0; step < nStep; step++) {
        cmpx G00 = -imag_1 * nrm0 * cElement(solver0.states[1], solver0.states[0], 0);
        cmpx G01 = -imag_1 * nrm1 * cElement(solver1.states[1], solver1.states[0], 0);
        double d = std::max(std::abs(G00 - ref[step].G00), std::abs(G01 - ref[step].G01));
        record(err, irlmTol(), step, ref[step].t, d);
        if (step + 1 < nStep) { solver0.iterate({.epsilon_M = 0}); solver1.iterate({.epsilon_M = 0}); }
    }
    return cache.emplace(us, err).first->second;
}

// The SIAM of app/fbr_green_siam.cpp: spin-symmetric layout, the up impurity
// orbital on site 0. Its ground state also comes off disk.
GreenError const &siamResult(std::string const &us)
{
    static std::map<std::string, GreenError> cache;
    auto it = cache.find(us);
    if (it != cache.end()) return it->second;

    constexpr int L = 100;
    constexpr double dt = 0.1;
    constexpr double V = 0.1;
    double U = std::stod(us);

    auto ref = loadSiamGreenReference("chain_green_siam_L" + std::to_string(L)
                                      + "_U" + us + ".dat");
    requireGrid(ref, dt);
    auto model = makeSiamModel(L, U, V);

    auto psi0 = loadFbMps<double>(findRef("fbr_green_gs_siam_L" + std::to_string(L)
                                          + "_U" + us + ".dat")).to_complex();
    auto [B, nrm] = addParticle(psi0, 0);
    psi0.tol = B.tol = 1e-12;
    auto solver = Fbr_dyn_shared(model, std::vector{B, psi0}, dt);

    GreenError err;
    std::size_t nStep = stepsToCompare(ref.size());
    for (std::size_t step = 0; step < nStep; step++) {
        cmpx G00 = -imag_1 * nrm * cElement(solver.states[1], solver.states[0], 0);
        record(err, siamTol(), step, ref[step].t, std::abs(G00 - ref[step].G00));
        if (step + 1 < nStep) solver.iterate({.epsilon_M = 0});
    }
    return cache.emplace(us, err).first->second;
}

// Both sides are approximations -- the chain baseline is itself a TDVP run -- so
// these bound how far apart two methods drift, not an exact error.
void checkGreen(GreenError const &err, std::vector<GreenTol> const &tols)
{
    // At t=0 the Green functions only involve the ground state, so this compares
    // the FBR ground state with the chain DMRG one, and is much tighter than the
    // bound on the evolution that follows.
    INFO("|dG(t=0)| = " << err.atZero);
    REQUIRE(err.atZero < 1e-6);

    for (auto const &bucket : tols) {
        auto it = err.byBucket.find(bucket.t);
        if (it == err.byBucket.end()) continue;      // the run did not get there
        INFO("up to t=" << bucket.t << " (run reached t=" << err.tMax
             << "): max|dG| = " << it->second);
        REQUIRE(it->second < bucket.tol);
    }
}

} // namespace

TEST_CASE("fbr green SIAM vs chain reference U=0.1", "[fb_ref_green][siam]") {
    checkGreen(siamResult("0.1"), siamTol());
}
TEST_CASE("fbr green SIAM vs chain reference U=0.2", "[fb_ref_green][siam]") {
    checkGreen(siamResult("0.2"), siamTol());
}
TEST_CASE("fbr green IRLM vs chain reference U=0.2", "[fb_ref_green][irlm]") {
    checkGreen(irlmResult("0.2"), irlmTol());
}
TEST_CASE("fbr green IRLM vs chain reference U=0.1", "[fb_ref_green][irlm]") {
    checkGreen(irlmResult("0.1"), irlmTol());
}
