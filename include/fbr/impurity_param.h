#ifndef FBR_PARAM_H
#define FBR_PARAM_H

#include "graph.h"
#include <armadillo>
#include <itensor/all.h>

namespace fbr {

struct ImpurityParam {
    arma::mat Kmat;           ///< the kinetic energy coefficient matrix
    arma::mat Umat;           ///< the Coulomb interaction coeff: U(i,j) ni nj
    std::vector<int> impPos;  ///< (default => {0,1,...,nImp-1}) the positions of interacting sites
    double filling=0.5;       ///< number of electrons per site
    arma::mat rot;            ///< (default => identity) the actual frame, such that F*Kmat*F.t() gives the original Kmat (in real space)

    int length() const { return Kmat.n_rows; }
    int nImp() const { return impPos.size(); }
    int nPart() const { return filling*length()+0.5; }

    void initializeDefault()
    {
        //TODO : verify correctness
        if (rot.empty()) rot=arma::mat(length(),length(), arma::fill::eye);
        if (impPos.empty()) throw std::invalid_argument("ImpurityParam::impPos shoul be initialize");
    }

    /// transform Kmat to star geometry (Hbath is diagonal)
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
    void toStar()
    {
        initializeDefault();
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
        // impPos=iota(nImp);
    }
};

struct Impurity {
    ImpurityParam param;

    Impurity() = default;
    Impurity(ImpurityParam const& param_) : param(param_) { param.toStar(); }
};

} // namespace fbr

#endif // FBR_PARAM_H
