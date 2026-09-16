#include <catch2/catch.hpp>
#include <limits>
#include "fbr/impurity_param.h"
#include "fbr/initial_state.h"

using namespace arma;
using namespace std;
using namespace fbr;

namespace {

/// A SIAM-like lattice: two spin chains interleaved on even/odd sites, each one
/// an impurity (site 0 for up, site 1 for dw) hybridized to a tight-binding
/// bath. `t_up`/`t_dw` scale the bath hopping of each spin, so passing
/// different values gives a model with no spin-flip symmetry.
mat interleavedChain(int L, double V, double t_up, double t_dw)
{
    mat K(L, L, fill::zeros);
    for (int i = 0; i < L-2; i++)
        K(i,i+2) = K(i+2,i) = (i%2==0) ? t_up : t_dw;
    K(0,2) = K(2,0) = V;
    K(1,3) = K(3,1) = V;
    return K;
}

/// The frame is what relates the star Hamiltonian back to the original one:
/// rot * Kstar * rot^T is the input Kmat.
void requireFrameRecoversInput(ImpurityParam const& star, mat const& Kinput)
{
    REQUIRE(norm(star.rot.t()*star.rot - mat(star.length(),star.length(),fill::eye),"fro")
            == Approx(0.0).margin(1e-12));
    REQUIRE(norm(star.rot*star.Kmat*star.rot.t() - Kinput,"fro") == Approx(0.0).margin(1e-10));
}

/// The bath block of a sector, i.e. everything but the impurity, must be diagonal.
void requireDiagonalBath(mat const& K, int a, int b)
{
    mat block = K.submat(a,a,b-1,b-1);
    REQUIRE(norm(block-diagmat(block.diag()),"fro") == Approx(0.0).margin(1e-12));
}

} // namespace

TEST_CASE("star transform, leading layout", "[param]")
{
    int L = 12;
    mat K(L, L, fill::zeros);
    for (int i = 1; i < L-1; i++) K(i,i+1) = K(i+1,i) = 0.5;
    K(0,1) = K(1,0) = 0.1;

    auto model = ImpurityParam{.Kmat=K, .Umat=mat(L,L,fill::zeros), .imp_pos={0,1}};
    model.to_star();

    REQUIRE(model.imp_pos == vector{0,1});
    requireDiagonalBath(model.Kmat, 2, L);
    requireFrameRecoversInput(model, K);
}

TEST_CASE("star transform, centered layouts", "[param]")
{
    int L = 12;
    double V = 0.3;

    SECTION("spin_symmetric puts the impurity at the center, one diagonal bath per spin")
    {
        mat K = interleavedChain(L, V, 0.5, 0.5);
        auto model = ImpurityParam{.Kmat=K, .Umat=mat(L,L,fill::zeros),
                                .imp_pos={0,1}, .layout=spin_symmetric};
        model.to_star();

        REQUIRE(model.imp_pos == vector{L/2-1, L/2});
        requireDiagonalBath(model.Kmat, 0, L/2-1);      // up bath
        requireDiagonalBath(model.Kmat, L/2+1, L);      // dw bath
        requireFrameRecoversInput(model, K);

        // the two sectors are mirror images of each other
        uvec irev = reverse(regspace<uvec>(0,L/2-1));
        REQUIRE(norm(model.Kmat.submat(irev,irev)
                     - model.Kmat.submat(L/2,L/2,L-1,L-1),"fro")
                == Approx(0.0).margin(1e-12));
    }

    SECTION("spin_symmetric rejects a model whose sectors are not mirror images")
    {
        mat K = interleavedChain(L, V, 0.5, 0.8);
        ImpurityParam param {.Kmat=K, .Umat=mat(L,L,fill::zeros),
                             .imp_pos={0,1}, .layout=spin_symmetric};
        REQUIRE_THROWS_AS(param.to_star(), std::invalid_argument);
    }

    SECTION("spin_block diagonalizes each spin bath on its own")
    {
        // up and dw baths differ, so the model has no spin-flip symmetry: a star
        // transform that mirrored one sector onto the other would replace the up
        // bath by a copy of the dw one.
        mat K = interleavedChain(L, V, 0.5, 0.8);
        auto model = ImpurityParam{.Kmat=K, .Umat=mat(L,L,fill::zeros),
                                .imp_pos={0,1}, .layout=spin_block};
        model.to_star();

        REQUIRE(model.imp_pos == vector{L/2-1, L/2});
        requireDiagonalBath(model.Kmat, 0, L/2-1);
        requireDiagonalBath(model.Kmat, L/2+1, L);
        requireFrameRecoversInput(model, K);

        // each sector keeps its own bath spectrum: eigenvalues of a
        // tight-binding chain of hopping t are 2t cos(k), so they scale with t.
        vec ek_up = sort(model.Kmat.diag().eval().rows(0,L/2-2));
        vec ek_dw = sort(model.Kmat.diag().eval().rows(L/2+1,L-1));
        REQUIRE(norm(ek_up-ek_dw) > 0.1);
        REQUIRE(norm(ek_up-ek_dw*(0.5/0.8)) == Approx(0.0).margin(1e-10));
    }
}


TEST_CASE("leading star transform preserves ordered impurities and interaction", "[param][regression]")
{
    int L=6;
    mat K=diagmat(vec{0.1,0.2,0.3,0.4,0.5,0.6});
    for (int i=0; i<L-1; ++i) K(i,i+1)=K(i+1,i)=0.15;
    vector<int> impurities;
    SECTION("overlapping swaps must not undo the impurity order") { impurities={1,0}; }
    SECTION("interaction must follow impurities moved from the bath") { impurities={4,2}; }

    mat U(L,L,fill::zeros);
    U(impurities[0],impurities[1])=0.7;
    auto model=ImpurityParam{.Kmat=K,.Umat=U,.imp_pos=impurities};
    model.to_star();

    REQUIRE(model.imp_pos == vector<int>{0,1});
    // Non-rotating columns must still identify the requested original sites,
    // in the requested order. Recovering K alone would not catch a wrong order.
    for (int i=0; i<2; ++i) {
        vec expected(L,fill::zeros);
        expected(impurities[i])=1;
        REQUIRE(norm(model.rot.col(i)-expected) == Approx(0).margin(1e-12));
    }
    mat expected_U(L,L,fill::zeros);
    expected_U(0,1)=0.7;
    REQUIRE(norm(model.Umat-expected_U,"fro") == Approx(0).margin(1e-12));
    REQUIRE(norm(model.rot*model.Umat*model.rot.t()-U,"fro") == Approx(0).margin(1e-12));
    requireFrameRecoversInput(model,K);
    requireDiagonalBath(model.Kmat,2,L);
}

TEST_CASE("Slater state accepts star models with omitted defaults", "[param][regression]")
{
    const auto model=ImpurityParam{.Kmat=diagmat(vec{-2,-1,1,2}),.imp_pos={0}};
    auto fb=slater<double>(model);
    REQUIRE(fb.rot.n_rows == 4);
    REQUIRE(norm(fb.rot-mat(4,4,fill::eye),"fro") == Approx(0).margin(1e-12));
    REQUIRE(norm(fb.correlator()-diagmat(vec{1,1,0,0}),"fro") == Approx(0).margin(1e-12));
    REQUIRE(model.rot.empty()); // constructing a state also works with a const model
}

TEST_CASE("model preparation checks inputs before constructing a state", "[param]")
{
    auto model=ImpurityParam{.Kmat=diagmat(vec{-2,-1,1,2}),.imp_pos={0}};
    SECTION("default matrices are initialized once") {
        model.prepare();
        REQUIRE(norm(model.rot-mat(4,4,fill::eye),"fro") == Approx(0).margin(1e-12));
        REQUIRE(norm(model.Umat,"fro") == Approx(0).margin(1e-12));
        model.rot.swap_cols(0,1);
        model.Umat(0,0)=0.5;
        model.prepare();
        REQUIRE(model.rot(1,0) == 1);
        REQUIRE(model.Umat(0,0) == 0.5);
        return;
    }
    SECTION("kinetic matrix must be square") { model.Kmat.zeros(4,3); }
    SECTION("rotation must match the model length") { model.rot.eye(3,3); }
    SECTION("interaction must match the model length") { model.Umat.zeros(3,3); }
    SECTION("impurity positions cannot repeat") { model.imp_pos={0,0}; }
    SECTION("impurity positions cannot be negative") { model.imp_pos={-1}; }
    SECTION("impurity positions must be inside the model") { model.imp_pos={4}; }
    SECTION("filling cannot be negative") { model.filling=-0.1; }
    SECTION("filling cannot exceed one") { model.filling=1.1; }
    SECTION("filling cannot be NaN") { model.filling=std::numeric_limits<double>::quiet_NaN(); }
    SECTION("centered models require even length") {
        model.Kmat.eye(5,5);
        model.imp_pos={1,2};
        model.layout=spin_block;
    }
    REQUIRE_THROWS_AS(model.prepare(),std::invalid_argument);
    REQUIRE_THROWS_AS(slater<double>(model),std::invalid_argument);
}

TEST_CASE("Slater state rejects energies and impurity positions inconsistent with the model", "[param]")
{
    auto model=ImpurityParam{.Kmat=diagmat(vec{-2,-1,1,2}),.imp_pos={0}};
    SECTION("wrong energy count") {
        REQUIRE_THROWS_AS(slater<double>(model,vec{-1,1}),std::invalid_argument);
    }
    SECTION("impurities must already occupy their layout positions") {
        model.imp_pos={2};
        REQUIRE_THROWS_AS(slater<double>(model),std::invalid_argument);
        model.to_star();
        REQUIRE_NOTHROW(slater<double>(model));
    }
}

TEST_CASE("star transform without bath preserves the entire impurity Hamiltonian", "[param]")
{
    auto model=ImpurityParam{.Kmat=diagmat(vec{-1,1}),.Umat=mat(2,2,fill::zeros),.imp_pos={1,0}};
    model.Umat(1,0)=0.7;
    model.to_star();
    REQUIRE(model.Kmat(0,0) == 1);
    REQUIRE(model.Umat(0,1) == 0.7);
    requireFrameRecoversInput(model,diagmat(vec{-1,1}));
}
