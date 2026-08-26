#ifndef FBR_PARAM_H
#define FBR_PARAM_H

#include "graph.h"
#include "fb_mps.h"
#include <armadillo>
#include <itensor/all.h>
#include <set>

namespace fbr {

/// The parameters of an impurity model, together with the chain geometry its
/// star transformation produces.
///
/// `layout==leading` gives the spinless chain |imp|active|slater|: impPos is a
/// flat list of the interacting sites in the CURRENT Kmat layout, and after
/// toStar() the impurity sits at the beginning, impPos=={0,...,nImp-1}.
///
/// The centered layouts (`spin_symmetric`, `spin_block`) give
/// |slater_up|active_up|imp_up|imp_dw|active_dw|slater_dw|, with impPos ordered
/// spatially as it should appear at the center of the chain:
///     impPos[0]      = outermost up  (next to up bath)
///     impPos[nUp-1]  = innermost up  (next to dw boundary at L/2)
///     impPos[nUp]    = innermost dw  (next to up boundary)
///     impPos[nImp-1] = outermost dw  (next to dw bath)
/// so after toStar(): impPos[i] == L/2 - nUp + i. Up/dw membership is inferred
/// from the connected components of Kmat (graph::find_islands): the first
/// nUp = nImp/2 entries must lie in one island, the rest in the other.
///
/// Umat is L×L, indexed by site in the current Kmat layout: the term is
///   sum_{i,j} Umat(i,j) N_i N_j
struct ImpurityParam {
    arma::mat Kmat;           ///< the kinetic energy coefficient matrix
    arma::mat Umat;           ///< the Coulomb interaction coeff: U(i,j) ni nj
    std::vector<int> impPos;  ///< the positions of the interacting sites
    double filling=0.5;       ///< number of electrons per site
    arma::mat rot;            ///< (default => identity) the actual frame, such that rot*Kmat*rot.t() gives the original Kmat (in real space)
    Layout layout=leading;    ///< the chain geometry toStar() produces

    int length() const { return Kmat.n_rows; }
    int nImp() const { return impPos.size(); }
    int nPart() const { return filling*length()+0.5; }

    void validate()
    {
        int L=length();
        if (rot.empty()) rot=arma::mat(L,L, arma::fill::eye);
        if (Umat.empty()) Umat=arma::mat(L,L, arma::fill::zeros);
        if ((int)Umat.n_rows!=L || (int)Umat.n_cols!=L)
            throw std::invalid_argument("ImpurityParam: Umat must be L×L");
        if (impPos.empty())
            throw std::invalid_argument("ImpurityParam: impPos must be non-empty");
        if (layout!=leading && impPos.size()%2)
            throw std::invalid_argument("ImpurityParam: a centered layout needs an even impPos (nUp = nDw)");
    }

    /// transform Kmat to star geometry (Hbath is diagonal), in the geometry
    /// selected by `layout`
    void toStar() { layout==leading ? toStarLeading() : toStarCentered(); }

    /// Split sites into two ordered halves consistent with a centered layout:
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
                throw std::invalid_argument("ImpurityParam: impPos[0..nUp-1] must all be in one island");
        for (int i = nUp; i < nImp(); i++)
            if (islands[impPos[i]] == comp_up)
                throw std::invalid_argument("ImpurityParam: impPos[nUp..] must all be in the other island");

        std::set<int> imp_set(impPos.begin(), impPos.end());

        std::vector<int> bath_up, bath_dw;
        for (int i = 0; i < L; i++) {
            if (imp_set.count(i)) continue;
            if (islands[i] == comp_up) bath_up.push_back(i);
            else                       bath_dw.push_back(i);
        }
        if ((int)(bath_up.size() + nUp) != L/2 || (int)(bath_dw.size() + nUp) != L/2)
            throw std::runtime_error("ImpurityParam::split_sites: spin block size mismatch");

        std::vector<int> sites_up = bath_up;
        for (int i = 0; i < nUp; i++) sites_up.push_back(impPos[i]);
        std::vector<int> sites_dw;
        for (int i = nUp; i < nImp(); i++) sites_dw.push_back(impPos[i]);
        sites_dw.insert(sites_dw.end(), bath_dw.begin(), bath_dw.end());
        return {sites_up, sites_dw};
    }

private:
    /// transform Kmat to star geometry, impurity first: |imp|bath|
    ///
    /// If the bath block decomposes into disconnected islands (e.g. when the
    /// input lattice contains independent spin sectors interleaved on the same
    /// indices), each island is diagonalized separately and its eigenvectors
    /// are placed back at the original bath positions of that island. This
    /// preserves the geometric (and hence spin) structure: a bath site that
    /// was, say, even-indexed in the input stays even-indexed in the star
    /// geometry. Diagonalizing the full bath at once would mix the islands
    /// (e.g. via degenerate eigenvalues), breaking spin coherence in the
    /// resulting orbital ordering.
    void toStarLeading()
    {
        validate();
        int L=length();
        int nImp=this->nImp();

        for(auto i=0u; i<impPos.size(); i++) {  // put the impurity at the beginning
            Kmat.swap_cols(i,impPos[i]);
            Kmat.swap_rows(i,impPos[i]);
            rot.swap_cols(i,impPos[i]);
        }
        arma::mat Kbath=Kmat.submat(nImp,nImp,L-1,L-1).eval();
        int nB=L-nImp;
        auto labels = graph::find_islands(Kbath);
        int n_islands = *std::max_element(labels.begin(), labels.end()) + 1;

        arma::mat Kstar(L,L,arma::fill::zeros);
        Kstar.submat(0,0,nImp-1,nImp-1)=Kmat.submat(0,0,nImp-1,nImp-1);
        arma::mat evec_full(nB,nB,arma::fill::zeros);

        for(int k=0; k<n_islands; k++) {
            std::vector<arma::uword> pos_v;
            for(int i=0;i<nB;i++) if (labels[i]==k) pos_v.push_back(i);
            arma::uvec pos_k(pos_v);

            arma::mat Kk = Kbath.submat(pos_k, pos_k);
            arma::vec ek1;
            arma::mat evec1;
            eig_sym(ek1, evec1, Kk);
            arma::uvec iek = arma::stable_sort_index(arma::abs(ek1));
            arma::mat evec = evec1.cols(iek);
            arma::vec ek   = ek1.rows(iek);

            arma::uvec pos_k_full = pos_k + nImp;
            arma::mat vk = Kmat.submat(arma::regspace<arma::uvec>(0, nImp-1), pos_k_full) * evec;

            for(auto j=0u; j<ek.n_elem; j++) {
                int jj = nImp + pos_k[j];
                Kstar(jj, jj) = ek[j];
                for(int i=0; i<nImp; i++)
                    Kstar(i, jj) = Kstar(jj, i) = vk(i, j);
            }
            for(auto j=0u; j<pos_k.n_elem; j++)
                for(auto i=0u; i<pos_k.n_elem; i++)
                    evec_full(pos_k[i], pos_k[j]) = evec(i, j);
        }

        Kmat = Kstar;
        rot.cols(nImp,L-1) = rot.cols(nImp,L-1).eval() * evec_full;
        impPos=iota(nImp);   // the impurity now sits at the beginning
    }

    /// transform Kmat to star geometry with the impurity at the center, one
    /// diagonal bath per spin: |bath_up|imp_up|imp_dw|bath_dw|
    void toStarCentered()
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
        arma::mat Umat_half(L/2, L/2, arma::fill::zeros);
        ImpurityParam half = {.Kmat = Kmat.submat(L/2, L/2, L-1, L-1), .Umat = Umat_half, .impPos=iota(nUp)};
        half.toStar();   // a leading star on the half chain

        // 3) duplicate by reflection to the up side
        arma::uvec irev = arma::reverse(arma::regspace<arma::uvec>(0, L/2 - 1));
        Kmat.submat(L/2, L/2, L-1, L-1) = half.Kmat;
        Kmat.submat(irev, irev) = half.Kmat;
        rot.cols(L/2, L-1) = rot.cols(L/2, L-1).eval() * half.rot;
        rot.cols(irev)     = rot.cols(irev).eval() * half.rot;
    }
};

struct Impurity {
    ImpurityParam param;

    Impurity() = default;
    Impurity(ImpurityParam const& param_) : param(param_) { param.toStar(); }

    /// A Slater state in the model's own frame, filling and layout.
    /// ek defaults to the diagonal of Kmat; pass your own to force a particular
    /// occupation (e.g. ek[i]=-10 to fill orbital i).
    template<class T=double>
    Fb_mps<T> slater(arma::vec ek={}) const
    {
        if (ek.empty()) ek=arma::vec {param.Kmat.diag()};
        return Fb_mps<T>::from_slater(arma::conv_to<arma::Mat<T>>::from(param.rot), ek,
                                      param.nPart(), param.nImp(), param.layout);
    }
};

} // namespace fbr

#endif // FBR_PARAM_H
