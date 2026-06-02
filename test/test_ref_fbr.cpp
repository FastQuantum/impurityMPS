#include <catch2/catch.hpp>

#include "fbr/fbr_dyn_spin.h"

#include <armadillo>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace arma;
using namespace fbr;

namespace {

struct SnapshotData {
    vec ni;
    cx_mat cc;
};

struct Metrics {
    double niMax = 0;
    double ccMax = 0;
};

std::map<std::string, SnapshotData> loadChainReference()
{
    std::ifstream in;
    for (auto const &path : std::vector<std::string>{
             "example/output/chain_dyn_siam_center_ref.txt",
             "../example/output/chain_dyn_siam_center_ref.txt",
             "../../example/output/chain_dyn_siam_center_ref.txt"}) {
        in.open(path);
        if (in) break;
        in.clear();
    }
    if (!in) throw std::runtime_error("missing example/output/chain_dyn_siam_center_ref.txt");

    std::string magic, token;
    int L = 0, snapshots = 0;
    in >> magic >> token >> L >> token >> snapshots;
    if (magic != "chain_dyn_siam_center_ref_v1" || L <= 0 || snapshots <= 0)
        throw std::runtime_error("invalid chain reference header");

    std::map<std::string, SnapshotData> refs;
    for (int s = 0; s < snapshots; s++) {
        std::string label;
        in >> token >> label;
        if (token != "snapshot") throw std::runtime_error("invalid snapshot marker");

        SnapshotData data;
        data.ni.set_size(L);
        data.cc.set_size(L, L);

        in >> token;
        if (token != "ni") throw std::runtime_error("invalid ni marker");
        for (int i = 0; i < L; i++) in >> data.ni[i];

        in >> token;
        if (token != "cc") throw std::runtime_error("invalid cc marker");
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++) {
                double re = 0, im = 0;
                in >> re >> im;
                data.cc(i, j) = cmpx(re, im);
            }
        refs[label] = std::move(data);
    }
    return refs;
}

Fbr_dyn_spin makeFbrRun(int L, double dt)
{
    ImpuritySpin model;
    {
        double U = 0.2;
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
    auto fb = Fb_mps_spin<cmpx>::from_slater(model.param.rot * cmpx(1, 0), ek,
                                             model.param.nPart(), model.param.nImp());
    auto solver = Fbr_dyn_spin(model, fb, dt);
    solver.fb.tol = 1e-12;
    return solver;
}

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

cx_mat toChainOrder(cx_mat const &cc, uvec const &p)
{
    cx_mat out(cc.n_rows, cc.n_cols, fill::zeros);
    for (uword i = 0; i < p.n_elem; i++)
        for (uword j = 0; j < p.n_elem; j++)
            out(p[i], p[j]) = cc(i, j);
    return out;
}

Metrics compare(cx_mat const &fbrCcChain, SnapshotData const &chain)
{
    cx_mat dcc = fbrCcChain - chain.cc;
    vec dni = real(fbrCcChain.diag()) - chain.ni;
    return {abs(dni).max(), abs(dcc).max()};
}

struct RefComparison {
    Metrics initial;
    Metrics t01;
    Metrics t50;
};

RefComparison runRefComparison()
{
    constexpr int L = 100;
    constexpr double dt = 0.1;
    auto refs = loadChainReference();
    auto fbr = makeFbrRun(L, dt);
    auto p = fbrIndexToChainIndex(L);

    RefComparison out;
    out.initial = compare(toChainOrder(fbr.correlator_all(), p), refs.at("initial"));
    for (int step = 1; step <= 50; step++) {
        fbr.iterate({.nIter_diag = 8, .err_goal = 1e-8, .epsilonM = 0e-8, .nKrylov = 15});
        if (step == 1)
            out.t01 = compare(toChainOrder(fbr.correlator_all(), p), refs.at("t=0.1"));
        if (step == 50)
            out.t50 = compare(toChainOrder(fbr.correlator_all(), p), refs.at("t=5.0"));
    }
    return out;
}

RefComparison const &refComparison()
{
    static RefComparison result = runRefComparison();
    return result;
}

} // namespace

TEST_CASE("fbr vs saved chain center reference: initial GS", "[fb_ref_fbr]") {
    auto const &result = refComparison();
    REQUIRE(result.initial.niMax < 1e-8);
    REQUIRE(result.initial.ccMax < 1e-5);
}

TEST_CASE("fbr vs saved chain center reference: t=0.1", "[fb_ref_fbr]") {
    auto const &result = refComparison();
    REQUIRE(result.t01.niMax < 2e-6);
    REQUIRE(result.t01.ccMax < 4e-5);
}

TEST_CASE("fbr vs saved chain center reference: t=5.0", "[fb_ref_fbr]") {
    auto const &result = refComparison();
    REQUIRE(result.t50.niMax < 1e-4);
    REQUIRE(result.t50.ccMax < 2e-4);
}
