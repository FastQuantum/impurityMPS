#include <catch2/catch.hpp>
#include "fbr/impurity_param.h"

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

    auto model = Impurity {{.Kmat=K, .Umat=mat(L,L,fill::zeros), .impPos={0,1}}};

    REQUIRE(model.param.impPos == vector{0,1});
    requireDiagonalBath(model.param.Kmat, 2, L);
    requireFrameRecoversInput(model.param, K);
}

TEST_CASE("star transform, centered layouts", "[param]")
{
    int L = 12;
    double V = 0.3;

    SECTION("spin_symmetric puts the impurity at the center, one diagonal bath per spin")
    {
        mat K = interleavedChain(L, V, 0.5, 0.5);
        auto model = Impurity {{.Kmat=K, .Umat=mat(L,L,fill::zeros),
                                .impPos={0,1}, .layout=spin_symmetric}};

        REQUIRE(model.param.impPos == vector{L/2-1, L/2});
        requireDiagonalBath(model.param.Kmat, 0, L/2-1);      // up bath
        requireDiagonalBath(model.param.Kmat, L/2+1, L);      // dw bath
        requireFrameRecoversInput(model.param, K);

        // the two sectors are mirror images of each other
        uvec irev = reverse(regspace<uvec>(0,L/2-1));
        REQUIRE(norm(model.param.Kmat.submat(irev,irev)
                     - model.param.Kmat.submat(L/2,L/2,L-1,L-1),"fro")
                == Approx(0.0).margin(1e-12));
    }

    SECTION("spin_symmetric rejects a model whose sectors are not mirror images")
    {
        mat K = interleavedChain(L, V, 0.5, 0.8);
        ImpurityParam param {.Kmat=K, .Umat=mat(L,L,fill::zeros),
                             .impPos={0,1}, .layout=spin_symmetric};
        REQUIRE_THROWS_AS(param.toStar(), std::invalid_argument);
    }

    SECTION("spin_block diagonalizes each spin bath on its own")
    {
        // up and dw baths differ, so the model has no spin-flip symmetry: a star
        // transform that mirrored one sector onto the other would replace the up
        // bath by a copy of the dw one.
        mat K = interleavedChain(L, V, 0.5, 0.8);
        auto model = Impurity {{.Kmat=K, .Umat=mat(L,L,fill::zeros),
                                .impPos={0,1}, .layout=spin_block}};

        REQUIRE(model.param.impPos == vector{L/2-1, L/2});
        requireDiagonalBath(model.param.Kmat, 0, L/2-1);
        requireDiagonalBath(model.param.Kmat, L/2+1, L);
        requireFrameRecoversInput(model.param, K);

        // each sector keeps its own bath spectrum: eigenvalues of a
        // tight-binding chain of hopping t are 2t cos(k), so they scale with t.
        vec ek_up = sort(model.param.Kmat.diag().eval().rows(0,L/2-2));
        vec ek_dw = sort(model.param.Kmat.diag().eval().rows(L/2+1,L-1));
        REQUIRE(norm(ek_up-ek_dw) > 0.1);
        REQUIRE(norm(ek_up-ek_dw*(0.5/0.8)) == Approx(0.0).margin(1e-10));
    }
}
