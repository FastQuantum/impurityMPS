#ifndef FBR_PARAM_H
#define FBR_PARAM_H

#include "graph.h"
#include "layout.h"
#include <armadillo>
#include <itensor/all.h>
#include <set>

namespace fbr {

/// The parameters of an impurity model, together with the chain geometry its
/// star transformation produces.
///
/// `layout==leading` gives the spinless chain |imp|active|slater|: imp_pos is a
/// flat list of the interacting sites in the CURRENT Kmat layout, and after
/// to_star() the impurity sits at the beginning, imp_pos=={0,...,n_imp-1}.
///
/// The centered layouts (`spin_symmetric`, `spin_block`) give
/// |slater_up|active_up|imp_up|imp_dw|active_dw|slater_dw|, with imp_pos ordered
/// spatially as it should appear at the center of the chain:
///     imp_pos[0]      = outermost up  (next to up bath)
///     imp_pos[nUp-1]  = innermost up  (next to dw boundary at L/2)
///     imp_pos[nUp]    = innermost dw  (next to up boundary)
///     imp_pos[n_imp-1] = outermost dw  (next to dw bath)
/// so after to_star(): imp_pos[i] == L/2 - nUp + i. Up/dw membership is inferred
/// from the connected components of Kmat (graph::find_islands): the first
/// nUp = n_imp/2 entries must lie in one island, the rest in the other.
///
/// Umat is L×L, indexed by site in the current Kmat layout: the term is
///   sum_{i,j} Umat(i,j) N_i N_j
struct ImpurityParam {
    arma::mat Kmat;           ///< the kinetic energy coefficient matrix
    arma::mat Umat;           ///< the Coulomb interaction coeff: U(i,j) ni nj
    std::vector<int> imp_pos;  ///< the positions of the interacting sites
    double filling=0.5;       ///< number of electrons per site
    arma::mat rot;            ///< (default => identity) the actual frame, such that rot*Kmat*rot.t() gives the original Kmat (in real space)
    Layout layout=leading;    ///< the chain geometry to_star() produces

    int length() const { return Kmat.n_rows; }
    int n_imp() const { return imp_pos.size(); }
    int n_part() const { return filling*length()+0.5; }

    void validate()
    {
        int L=length();
        if (rot.empty()) rot=arma::mat(L,L, arma::fill::eye);
        if (Umat.empty()) Umat=arma::mat(L,L, arma::fill::zeros);
        if ((int)Umat.n_rows!=L || (int)Umat.n_cols!=L)
            throw std::invalid_argument("ImpurityParam: Umat must be L×L");
        if (imp_pos.empty())
            throw std::invalid_argument("ImpurityParam: imp_pos must be non-empty");
        if (layout!=leading && imp_pos.size()%2)
            throw std::invalid_argument("ImpurityParam: a centered layout needs an even imp_pos (nUp = nDw)");
    }

    /// transform Kmat to star geometry (Hbath is diagonal), in the geometry
    /// selected by `layout`
    void to_star() { layout==leading ? to_star_leading() : to_star_centered(); }

    /// Split sites into two ordered halves consistent with a centered layout:
    ///   sites_up = [bath_up reversed..., imp_pos[0], ..., imp_pos[nUp-1]]
    ///   sites_dw = [imp_pos[nUp], ..., imp_pos[n_imp-1], bath_dw...]
    /// so that concatenation = pos_all maps new->old.
    ///
    /// The up bath is listed in reverse so that reading the up half from the
    /// center outwards visits its sites in the same order as the dw half does:
    /// the two halves are then mirror images as matrices, which is what lets
    /// to_star_centered() reflect one onto the other.
    std::pair<std::vector<int>, std::vector<int>> split_sites() const
    {
        int L = length();
        int nUp = n_imp()/2;

        auto islands = graph::find_islands(Kmat);
        int comp_up = islands[imp_pos[0]];
        for (int i = 0; i < nUp; i++)
            if (islands[imp_pos[i]] != comp_up)
                throw std::invalid_argument("ImpurityParam: imp_pos[0..nUp-1] must all be in one island");
        for (int i = nUp; i < n_imp(); i++)
            if (islands[imp_pos[i]] == comp_up)
                throw std::invalid_argument("ImpurityParam: imp_pos[nUp..] must all be in the other island");

        std::set<int> imp_set(imp_pos.begin(), imp_pos.end());

        std::vector<int> bath_up, bath_dw;
        for (int i = 0; i < L; i++) {
            if (imp_set.count(i)) continue;
            if (islands[i] == comp_up) bath_up.push_back(i);
            else                       bath_dw.push_back(i);
        }
        if ((int)(bath_up.size() + nUp) != L/2 || (int)(bath_dw.size() + nUp) != L/2)
            throw std::runtime_error("ImpurityParam::split_sites: spin block size mismatch");

        std::vector<int> sites_up(bath_up.rbegin(), bath_up.rend());
        for (int i = 0; i < nUp; i++) sites_up.push_back(imp_pos[i]);
        std::vector<int> sites_dw;
        for (int i = nUp; i < n_imp(); i++) sites_dw.push_back(imp_pos[i]);
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
    void to_star_leading()
    {
        validate();
        int L=length();
        int n_imp=this->n_imp();

        for(auto i=0u; i<imp_pos.size(); i++) {  // put the impurity at the beginning
            Kmat.swap_cols(i,imp_pos[i]);
            Kmat.swap_rows(i,imp_pos[i]);
            rot.swap_cols(i,imp_pos[i]);
        }
        arma::mat Kbath=Kmat.submat(n_imp,n_imp,L-1,L-1).eval();
        int nB=L-n_imp;
        auto labels = graph::find_islands(Kbath);
        int n_islands = *std::max_element(labels.begin(), labels.end()) + 1;

        arma::mat Kstar(L,L,arma::fill::zeros);
        Kstar.submat(0,0,n_imp-1,n_imp-1)=Kmat.submat(0,0,n_imp-1,n_imp-1);
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

            arma::uvec pos_k_full = pos_k + n_imp;
            arma::mat vk = Kmat.submat(arma::regspace<arma::uvec>(0, n_imp-1), pos_k_full) * evec;

            for(auto j=0u; j<ek.n_elem; j++) {
                int jj = n_imp + pos_k[j];
                Kstar(jj, jj) = ek[j];
                for(int i=0; i<n_imp; i++)
                    Kstar(i, jj) = Kstar(jj, i) = vk(i, j);
            }
            for(auto j=0u; j<pos_k.n_elem; j++)
                for(auto i=0u; i<pos_k.n_elem; i++)
                    evec_full(pos_k[i], pos_k[j]) = evec(i, j);
        }

        Kmat = Kstar;
        rot.cols(n_imp,L-1) = rot.cols(n_imp,L-1).eval() * evec_full;
        imp_pos=iota(n_imp);   // the impurity now sits at the beginning
    }

    /// transform Kmat to star geometry with the impurity at the center, one
    /// diagonal bath per spin: |bath_up|imp_up|imp_dw|bath_dw|
    void to_star_centered()
    {
        validate();
        int L = length();
        int nUp = n_imp()/2;

        // 1) reorder sites: bath_up | imp_up (outer..inner) | imp_dw (inner..outer) | bath_dw
        auto [sites_up, sites_dw] = split_sites();
        std::vector<int> all_sites = sites_up;
        all_sites.insert(all_sites.end(), sites_dw.begin(), sites_dw.end());
        arma::uvec pos_all = arma::conv_to<arma::uvec>::from(all_sites);

        Kmat = Kmat.submat(pos_all, pos_all).eval();
        Umat = Umat.submat(pos_all, pos_all).eval();
        rot  = rot.cols(pos_all).eval();
        for (int i = 0; i < n_imp(); i++) imp_pos[i] = L/2 - nUp + i;

        // 2) diagonalize the dw bath. Read from the center outwards each half is
        // a leading chain of its own -- impurities first, then its bath -- which
        // is the |imp|bath| shape to_star_leading() expects. For dw that reading is
        // the submatrix itself; for up it is the submatrix under `irev`.
        arma::uvec irev = arma::reverse(arma::regspace<arma::uvec>(0, L/2 - 1));
        auto half_star=[&](arma::mat const& Khalf) {
            ImpurityParam half = {.Kmat = Khalf,
                                  .Umat = arma::mat(L/2, L/2, arma::fill::zeros),
                                  .imp_pos = iota(nUp)};
            half.to_star();   // a leading star on the half chain
            return half;
        };
        if (layout==spin_symmetric) {   // the layout only holds for a symmetric model
            arma::mat asym = Kmat.submat(irev,irev) - Kmat.submat(L/2, L/2, L-1, L-1);
            if (asym.max() > 1e-10 || asym.min() < -1e-10)
                throw std::invalid_argument("ImpurityParam: spin_symmetric needs the two spin "
                                            "sectors to be mirror images; use spin_block instead");
        }
        auto half = half_star(Kmat.submat(L/2, L/2, L-1, L-1));
        Kmat.submat(L/2, L/2, L-1, L-1) = half.Kmat;
        rot.cols(L/2, L-1) = rot.cols(L/2, L-1).eval() * half.rot;

        // 3) and the up bath. Under spin_symmetric the two halves are the same
        // matrix, so reflecting the dw result is both cheaper and exact -- and it
        // keeps the mirror symmetry that layout relies on free of any eigenvector
        // sign the two diagonalizations could disagree on. spin_block assumes no
        // such symmetry, so its up bath is diagonalized on its own.
        auto const& up_half = (layout==spin_symmetric) ? half
                                                       : half_star(Kmat.submat(irev, irev));
        Kmat.submat(irev, irev) = up_half.Kmat;
        rot.cols(irev)         = rot.cols(irev).eval() * up_half.rot;
    }
};

struct Impurity {
    ImpurityParam param;

    Impurity() = default;
    Impurity(ImpurityParam const& param_) : param(param_) { param.to_star(); }
};

} // namespace fbr

#endif // FBR_PARAM_H
