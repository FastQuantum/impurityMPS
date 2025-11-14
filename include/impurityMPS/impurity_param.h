#ifndef IMPURITY_PARAM_H
#define IMPURITY_PARAM_H

#include "impurityMPS/fermionic.h"
#include <armadillo>
#include <itensor/all.h>


struct ImpurityParam {
    arma::sp_mat Kstar;     ///< the kinetic energy coefficient matrix in star geometry (Hbath is diagonal)
    arma::sp_mat Umat;      ///< the Coulomb interaction coeff: U(i,j) ni nj
    double filling=0.5;     ///< number of electrons per site
    arma::mat F;            ///< (optional) the actual frame, such that F*Kstar*F.t() gives the original Kmat (in real space)

    int length() const { return Kstar.n_rows; }
    int nImp() const { return Umat.n_rows; }
    int nPart() const { return filling*length()+0.5; }

    /// helper: transform Kmat to star geometry (Hbath is diagonal)
    /// return Kstar and the rotation F performed: Kstar=F.t()*Kmat*F
    static std::pair<arma::sp_mat,arma::mat> to_star_kin(arma::mat const& Kmat, int nImp)
    {
        int L=Kmat.n_rows;
        arma::mat Kbath=Kmat.submat(nImp,nImp,L-1,L-1).eval();
        arma::mat evec;
        arma::vec ek;
        arma::eig_sym(ek,evec,Kbath);
        // auto [ek,evec]=FullDiagonalizeTridiagonal(Kbath.diag().eval(),Kbath.diag(1).eval());
        arma::mat vk=(Kmat.submat(0,nImp,nImp-1,L-1)*evec);

        arma::sp_mat Kstar(L,L);
        Kstar.submat(0,0,nImp-1,nImp-1)=Kmat.submat(0,0,nImp-1,nImp-1);
        for(auto jj=0u;jj<ek.size();jj++) {
            Kstar(jj+nImp,jj+nImp)=ek[jj];
            for(auto i=0; i<nImp; i++)
                Kstar(i,jj+nImp)=Kstar(jj+nImp,i)=vk(i,jj);
        }
        arma::mat F(L,L,arma::fill::eye);  // the full rotation performed
        F.submat(nImp,nImp,L-1,L-1)=evec;
        return {Kstar,F};
    }

    /// (deprecated!)helper: transform Kmat to star geometry (Hbath is diagonal) for tridiagonal matrix Kmat
    static arma::sp_mat to_star_kin_tridiag(arma::sp_mat const& Kmat, int nImp) // TODO
    {
        int L=Kmat.n_rows;
        auto Kbath=Kmat.submat(nImp,nImp,L-1,L-1).eval();
        auto [ek,evec]=FullDiagonalizeTridiagonal(arma::vec {Kbath.diag()}, arma::vec {Kbath.diag(1)});
        arma::mat vk=(Kmat.submat(0,nImp,nImp-1,L-1)*evec);

        arma::sp_mat Kstar(L,L);
        Kstar.submat(0,0,nImp-1,nImp-1)=Kmat.submat(0,0,nImp-1,nImp-1);
        for(auto jj=0u;jj<ek.size();jj++) {
            Kstar(jj+nImp,jj+nImp)=ek[jj];
            for(auto i=0; i<nImp; i++)
                Kstar(i,jj+nImp)=Kstar(jj+nImp,i)=vk(i,jj);
        }
        return Kstar;
    }
};

#endif // IMPURITY_PARAM_H
