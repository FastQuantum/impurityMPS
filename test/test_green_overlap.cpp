// Different-frame overlaps (green_overlap.h) checked against the exact
// Slater-determinant formula, a common-frame ground truth, and the invariance
// a relabeling swap must have.
#include <catch2/catch.hpp>
#include "fbr/green_overlap.h"

using namespace arma;
using namespace fbr;

namespace {

cx_mat random_unitary(int L, int seed)
{
    arma_rng::set_seed(seed);
    cx_mat X(L,L,fill::randn); X += cx_mat(L,L,fill::randn)*cmpx(0,1);
    cx_mat Q,R; qr(Q,R,X);
    return Q * diagmat(R.diag()/abs(R.diag()));
}

cx_mat imp_frame(int L, int seed)   // identity on the impurity site 0
{
    cx_mat F(L,L,fill::eye);
    F.submat(1,1,L-1,L-1) = random_unitary(L-1,seed);
    return F;
}

// A product (Slater) state on the SAME ITensor sites as `proto`, in frame `rot`,
// occupying the columns listed in `occ`.
Fb_mps<cmpx> slater_like(Fb_mps<cmpx> const& proto, cx_mat const& rot, uvec const& occ)
{
    auto fb = proto;
    fb.rot = rot;
    auto st = itensor::InitState(fb.sites,"0");
    fb.cc.zeros();
    for (auto k : occ) { st.set((int)k+1,"1"); fb.cc(k,k)=1; }
    fb.psi = itensor::MPS(st);
    fb.active = {0, fb.length()};
    fb.tol = 1e-12;
    return fb;
}

Fb_mps<cmpx> proto_state(int L)
{
    return Fb_mps<double>::from_slater(mat(L,L,fill::eye),
                                       regspace<vec>(0,L-1), 1, 1, leading).to_complex();
}

// <A|B> for two Slater states = det( (A.rot^dag B.rot)[occA, occB] ).
cmpx det_overlap(cx_mat const& rotA, uvec occA, cx_mat const& rotB, uvec occB)
{
    cx_mat M = cx_mat(rotA.t()*rotB).submat(sort(occA), sort(occB));
    return det(M);
}

} // namespace

TEST_CASE("overlap of Slater states in different frames == determinant", "[green_overlap]")
{
    const int L=8, n_part=4;
    auto proto = proto_state(L);

    for (int trial=0; trial<5; ++trial) {
        cx_mat rotA = random_unitary(L, 10+trial);
        cx_mat rotB = random_unitary(L, 100+trial);
        uvec occA = sort(uvec(shuffle(regspace<uvec>(0,L-1))).head(n_part));
        uvec occB = sort(uvec(shuffle(regspace<uvec>(0,L-1))).head(n_part));

        auto A = slater_like(proto, rotA, occA);
        auto B = slater_like(proto, rotB, occB);

        REQUIRE(std::abs(overlap(A,B) - det_overlap(rotA,occA,rotB,occB)) < 1e-10);
    }
}

TEST_CASE("overlap is invariant under an entangled representation", "[green_overlap]")
{
    // Rotate each state into a scrambled working frame: the physical Slater state
    // (hence the overlap) is unchanged, but the MPS now has bond dimension > 1.
    const int L=8, n_part=4;
    auto proto = proto_state(L);
    for (int trial=0; trial<4; ++trial) {
        cx_mat rotA = random_unitary(L, 3+trial), rotB = random_unitary(L, 40+trial);
        uvec occA = sort(uvec(shuffle(regspace<uvec>(0,L-1))).head(n_part));
        uvec occB = sort(uvec(shuffle(regspace<uvec>(0,L-1))).head(n_part));
        auto A = slater_like(proto, rotA, occA);
        auto B = slater_like(proto, rotB, occB);
        cmpx ref = det_overlap(rotA,occA,rotB,occB);
        align_to_frame(A, random_unitary(L, 500+trial));
        align_to_frame(B, random_unitary(L, 900+trial));
        REQUIRE(std::abs(overlap(A,B) - ref) < 1e-9);
    }
}

TEST_CASE("c_element in different frames == common-frame contraction", "[green_overlap]")
{
    // A (n_part) and B (n_part+1) in their own working frames, as in a Green
    // function; c_element aligns B to A and applies c_0 at the impurity. Ground
    // truth: bring BOTH to a third common frame and let ITensor contract.
    const int L=6;
    auto proto = proto_state(L);
    for (int trial=0; trial<5; ++trial) {
        auto A0 = slater_like(proto, imp_frame(L,11+trial), uvec{1,2,4}); // site 0 empty
        auto B0 = slater_like(proto, imp_frame(L,31+trial), uvec{0,2,3,5});

        cx_mat C = imp_frame(L,51+trial);
        auto Ac=A0; align_to_frame(Ac,C);
        auto Bc=B0; align_to_frame(Bc,C);
        auto Ai=Ac; Ai.apply_local_op("Cdag",0);
        cmpx ref = itensor::innerC(Ai.psi, Bc.psi);

        auto Aw=A0; align_to_frame(Aw, imp_frame(L,71+trial));
        auto Bw=B0; align_to_frame(Bw, imp_frame(L,91+trial));
        REQUIRE(std::abs(c_element(Aw,Bw,0) - ref) < 1e-9);
    }
}

TEST_CASE("a Slater relabeling swap preserves the global phase", "[green_overlap]")
{
    // swap_slater_orbitals is a frame relabeling: the physical state is unchanged,
    // so overlap(A, swapped A) must be +1, not -1. A -1 (anti-symmetric hopping)
    // is invisible to single-state observables but corrupts cross-state elements.
    const int L=8;
    auto proto = proto_state(L);
    auto A = slater_like(proto, cx_mat(L,L,fill::eye), uvec{1,2,4});
    auto B = A;
    B.active = {0,1};                       // orbitals 1 and 5 are Slater
    OrbitalUpdate<cmpx> up(0,1);
    up.gates.emplace_back(1,5);             // swap occupied 1 with empty 5
    up.active = {0,1};
    B.apply(up);
    REQUIRE(std::abs(overlap(A,B) - cmpx(1,0)) < 1e-10);
}
