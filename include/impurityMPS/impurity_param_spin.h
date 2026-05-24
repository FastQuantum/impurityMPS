#ifndef IMPURITY_PARAM_SPIN_H
#define IMPURITY_PARAM_SPIN_H

#include "impurity_param.h"
#include <armadillo>
#include <itensor/all.h>
#include <set>

/// Convention 2 (left-to-right in final layout):
///   impPos is a flat list of nImp impurity sites in the CURRENT Kmat layout,
///   ordered spatially as they should appear at the center of the chain:
///     impPos[0]        = outermost up  (next to up bath)
///     impPos[nUp-1]    = innermost up  (next to dw boundary at L/2)
///     impPos[nUp]      = innermost dw  (next to up boundary)
///     impPos[nImp-1]   = outermost dw  (next to dw bath)
///   After toStar(): impPos[i] == L/2 - nUp + i.
///
/// Up/dw membership is inferred from the connected components of Kmat
/// (graph::find_islands).  The first nUp = nImp/2 entries must lie in one
/// island; the rest must lie in the other.
///
/// Umat is L×L, indexed by site in the current Kmat layout: term
///   sum_{i,j} Umat(i,j) N_i N_j
struct ImpurityParamSpin {
    arma::mat Kmat;
    arma::mat Umat;
    std::vector<int> impPos;
    double filling=0.5;
    arma::mat rot;

    int length() const { return Kmat.n_rows; }
    int nImp() const { return impPos.size(); }
    int nPart() const { return filling*length()+0.5; }

    void validate()
    {
        int L = length();
        if (rot.empty()) rot = arma::mat(L, L, arma::fill::eye);
        if (Umat.empty()) Umat = arma::mat(L, L, arma::fill::zeros);
        if ((int)Umat.n_rows != L || (int)Umat.n_cols != L)
            throw std::invalid_argument("ImpurityParamSpin: Umat must be L×L");
        if (impPos.empty())
            throw std::invalid_argument("ImpurityParamSpin: impPos must be non-empty");
        if (impPos.size() % 2)
            throw std::invalid_argument("ImpurityParamSpin: impPos size must be even (nUp = nDw)");
    }

    /// Split sites into two ordered halves consistent with convention 2:
    ///   sites_up = [bath_up..., impPos[0], impPos[1], ..., impPos[nUp-1]]
    ///   sites_dw = [impPos[nUp], ..., impPos[nImp-1], bath_dw...]
    /// so that concatenation = pos_all maps new->old.
    std::pair<std::vector<int>, std::vector<int>> split_sites() const
    {
        int L = length();
        int nUp = nImp()/2;

        auto islands = graph::find_islands(Kmat);
        int comp_up = islands[impPos[0]];
        for (int i = 0; i < nUp; i++)
            if (islands[impPos[i]] != comp_up)
                throw std::invalid_argument("ImpurityParamSpin: impPos[0..nUp-1] must all be in one island");
        for (int i = nUp; i < nImp(); i++)
            if (islands[impPos[i]] == comp_up)
                throw std::invalid_argument("ImpurityParamSpin: impPos[nUp..] must all be in the other island");

        std::set<int> imp_set(impPos.begin(), impPos.end());

        std::vector<int> bath_up, bath_dw;
        for (int i = 0; i < L; i++) {
            if (imp_set.count(i)) continue;
            if (islands[i] == comp_up) bath_up.push_back(i);
            else                       bath_dw.push_back(i);
        }
        if ((int)(bath_up.size() + nUp) != L/2 || (int)(bath_dw.size() + nUp) != L/2)
            throw std::runtime_error("ImpurityParamSpin::split_sites: spin block size mismatch");

        std::vector<int> sites_up = bath_up;
        for (int i = 0; i < nUp; i++) sites_up.push_back(impPos[i]);
        std::vector<int> sites_dw;
        for (int i = nUp; i < nImp(); i++) sites_dw.push_back(impPos[i]);
        sites_dw.insert(sites_dw.end(), bath_dw.begin(), bath_dw.end());
        return {sites_up, sites_dw};
    }

    /// transform Kmat to star geometry (Hbath is diagonal per spin)
    void toStar()
    {
        validate();
        int L = length();
        int nUp = nImp()/2;

        // 1) reorder sites: bath_up | imp_up (outer..inner) | imp_dw (inner..outer) | bath_dw
        auto [sites_up, sites_dw] = split_sites();
        std::vector<int> all_sites = sites_up;
        all_sites.insert(all_sites.end(), sites_dw.begin(), sites_dw.end());
        arma::uvec pos_all = arma::conv_to<arma::uvec>::from(all_sites);

        Kmat = Kmat.submat(pos_all, pos_all).eval();
        Umat = Umat.submat(pos_all, pos_all).eval();
        rot  = rot.cols(pos_all).eval();
        for (int i = 0; i < nImp(); i++) impPos[i] = L/2 - nUp + i;

        // 2) diagonalize the dw bath: dw submatrix has impurities at positions [0..nUp-1]
        arma::mat Umat_half(nUp, nUp, arma::fill::zeros);
        ImpurityParam half = {.Kmat = Kmat.submat(L/2, L/2, L-1, L-1), .Umat = Umat_half};
        half.toStar();

        // 3) duplicate by reflection to the up side
        arma::uvec irev = arma::reverse(arma::regspace<arma::uvec>(0, L/2 - 1));
        Kmat.submat(L/2, L/2, L-1, L-1) = half.Kmat;
        Kmat.submat(irev, irev) = half.Kmat;
        rot.cols(L/2, L-1) = rot.cols(L/2, L-1).eval() * half.rot;
        rot.cols(irev)     = rot.cols(irev).eval() * half.rot;
    }
};

struct ImpuritySpin {
    ImpurityParamSpin param;
    ImpuritySpin() = default;
    ImpuritySpin(ImpurityParamSpin const& param_) : param(param_) { param.toStar(); }
};

#endif // IMPURITY_PARAM_SPIN_H
