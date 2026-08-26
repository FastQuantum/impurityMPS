#pragma once

// Shared, TDVP-independent helpers for the reference-comparison tests
// (test_ref_fbr / test_ref_block / test_ref_ns). Each variant compiles into its
// own executable because the TDVP headers define non-inline functions; only the
// variant-specific FBR run lives in the .cpp, everything reusable lives here.

#include <armadillo>

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fbrtest {

using namespace arma;
using cmpx = std::complex<double>;

// One snapshot of a reference run: occupations and the full one-particle
// correlation matrix <c_i^dag c_j>, in original (chain-site) order.
struct SnapshotData {
    vec ni;
    cx_mat cc;
};

// Max abs deviation of occupations and of the correlation matrix.
struct Metrics {
    double niMax = 0;
    double ccMax = 0;
};

// A named set of reference snapshots (e.g. "chain" or "star").
using RefSet = std::map<std::string, SnapshotData>;

// Map a snapshot label to the TDVP step index it was taken at (dt = 0.1).
inline int stepOfLabel(std::string const &label)
{
    if (label == "initial") return 0;
    if (label == "t=0.1") return 1;
    if (label == "t=5.0") return 50;
    if (label == "t=10.0") return 100;
    if (label == "t=20.0") return 200;
    throw std::runtime_error("unknown snapshot label: " + label);
}

// Locate a reference file under test/ref/output/, trying the paths that work
// whether the test runs from the build dir, the repo root, or test/.
inline std::string findRef(std::string const &name)
{
    for (auto const &base : std::vector<std::string>{
             "test/ref/output/", "../test/ref/output/",
             "../../test/ref/output/"}) {
        std::ifstream in(base + name);
        if (in) return base + name;
    }
    throw std::runtime_error("missing reference file test/ref/output/" + name);
}

// Parse a chain_dyn_siam_center_ref_v1 file into a label -> SnapshotData map.
inline RefSet loadReference(std::string const &name)
{
    std::ifstream in(findRef(name));
    std::string magic, token;
    int L = 0, snapshots = 0;
    in >> magic >> token >> L >> token >> snapshots;
    if (magic != "chain_dyn_siam_center_ref_v1" || L <= 0 || snapshots <= 0)
        throw std::runtime_error("invalid reference header in " + name);

    RefSet refs;
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

// ---- Green function references (chain_green_irlm_ref_v1) ----

// One time slice of a Green function reference: the two impurity Green
// functions and the bond dimension the chain run needed there.
struct GreenSample {
    double t;
    cmpx G00, G01;
    int m;
};

// Parse a chain_green_irlm_ref_v1 file: a header line, a comment line, then one
// row per time step. The run may have stopped early (bond dimension), so the
// number of rows is whatever the file says.
inline std::vector<GreenSample> loadGreenReference(std::string const &name)
{
    std::ifstream in(findRef(name));
    std::string magic, token;
    int L = 0, steps = 0;
    double U = 0, V = 0, dt = 0;
    in >> magic >> token >> L >> token >> U >> token >> V >> token >> dt >> token >> steps;
    if (magic != "chain_green_irlm_ref_v1" || L <= 0 || steps <= 0)
        throw std::runtime_error("invalid Green reference header in " + name);
    std::getline(in, token);
    std::getline(in, token);   // the "# t ReG00 ..." comment line

    std::vector<GreenSample> rows;
    rows.reserve(steps);
    for (int s = 0; s < steps; s++) {
        GreenSample r;
        double re0 = 0, im0 = 0, re1 = 0, im1 = 0;
        in >> r.t >> re0 >> im0 >> re1 >> im1 >> r.m;
        if (!in) throw std::runtime_error("truncated Green reference " + name);
        r.G00 = cmpx(re0, im0);
        r.G01 = cmpx(re1, im1);
        rows.push_back(r);
    }
    return rows;
}

// Reorder a correlator from FBR site order to chain-reference site order. The
// permutation `p` is variant-specific (the spinful and spinless layouts differ),
// so each test .cpp supplies its own fbrIndexToChainIndex.
inline cx_mat toChainOrder(cx_mat const &cc, uvec const &p)
{
    cx_mat out(cc.n_rows, cc.n_cols, fill::zeros);
    for (uword i = 0; i < p.n_elem; i++)
        for (uword j = 0; j < p.n_elem; j++)
            out(p[i], p[j]) = cc(i, j);
    return out;
}

inline Metrics compare(cx_mat const &fbrCcChain, SnapshotData const &ref)
{
    cx_mat dcc = fbrCcChain - ref.cc;
    vec dni = real(fbrCcChain.diag()) - ref.ni;
    return {abs(dni).max(), abs(dcc).max()};
}

// Per-snapshot tolerances (niMax, ccMax) shared by all three FBR variants.
using Tol = std::pair<double, double>;

// FBR vs the CHAIN baseline: the tight gold standard. Observed max deviations are
// ~1e-4 (up to ~1e-3 at t=20 for U=0.2); these bounds sit ~2x above.
inline std::map<std::string, Tol> chainTol()
{
    return {{"initial", {1e-7, 2e-5}}, {"t=0.1", {5e-6, 1e-4}}, {"t=5.0", {2e-4, 4e-4}},
            {"t=10.0", {2e-4, 4e-4}},  {"t=20.0", {1e-3, 2.5e-3}}};
}

// Result for one FBR trajectory: label -> Metrics.
using TrajResult = std::map<std::string, Metrics>;

// Evolve a single FBR trajectory and, at each snapshot present in `ref`, compare
// its correlator against the reference. `iter(fbr)` advances one TDVP step;
// `corr(fbr)` returns the correlator in FBR (original) order, which is reordered
// to chain order before comparing.
template <class Fbr, class IterFn, class CorrFn>
TrajResult compareTrajectory(Fbr &fbr, IterFn iter, CorrFn corr, RefSet const &ref,
                             uvec const &p)
{
    int maxStep = 0;
    for (auto const &kv : ref) maxStep = std::max(maxStep, stepOfLabel(kv.first));
#ifndef FBR_ENABLE_LONG_TEST
    // Default: stop at t=5 (step 50). Snapshots at later times are simply not
    // reached, so they are not compared. Build with -DFBR_ENABLE_LONG_TEST=ON to
    // run the full t=20 sweep.
    maxStep = std::min(maxStep, 50);
#endif

    TrajResult out;
    auto doCompare = [&](int step) {
        cx_mat cc = toChainOrder(corr(fbr), p);
        for (auto const &kv : ref)
            if (stepOfLabel(kv.first) == step) out[kv.first] = compare(cc, kv.second);
    };

    doCompare(0);
    for (int step = 1; step <= maxStep; step++) {
        iter(fbr);
        doCompare(step);
    }
    return out;
}

} // namespace fbrtest
