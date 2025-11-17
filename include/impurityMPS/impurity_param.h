#ifndef IMPURITY_PARAM_H
#define IMPURITY_PARAM_H

#include "impurityMPS/fermionic.h"
#include <armadillo>
#include <itensor/all.h>

struct ImpurityParam {
    arma::mat Kmat;           ///< the kinetic energy coefficient matrix
    arma::mat Umat;           ///< the Coulomb interaction coeff: U(i,j) ni nj
    std::vector<int> impPos;  ///< (default => {0,1,...,nImp-1}) the positions of interacting sites
    double filling=0.5;       ///< number of electrons per site
    arma::mat F;              ///< (default => identity) the actual frame, such that F*Kmat*F.t() gives the original Kmat (in real space)

    int length() const { return Kmat.n_rows; }
    int nImp() const { return Umat.n_rows; }
    int nPart() const { return filling*length()+0.5; }

    void validate()
    {
        if (F.empty()) F=arma::mat(length(),length(), arma::fill::eye);
        if (impPos.empty()) impPos=iota(nImp());
    }

    /// transform Kmat to star geometry (Hbath is diagonal)
    void toStar()
    {
        validate();
        // TODO : if the matrix is already in star then return *this;
        int L=length();
        int nImp=this->nImp();

        for(auto i=0u; i<impPos.size(); i++) {  // put the impurity at the beginning
            Kmat.swap_cols(i,impPos[i]);
            Kmat.swap_rows(i,impPos[i]);
            F.swap_cols(i,impPos[i]);
        }
        arma::mat Kbath=Kmat.submat(nImp,nImp,L-1,L-1).eval();
        arma::mat evec;
        arma::vec ek;
        arma::eig_sym(ek,evec,Kbath);
        arma::mat vk=(Kmat.submat(0,nImp,nImp-1,L-1)*evec);

        arma::mat Kstar(L,L,arma::fill::zeros);
        Kstar.submat(0,0,nImp-1,nImp-1)=Kmat.submat(0,0,nImp-1,nImp-1);
        for(auto j=0u;j<ek.size();j++) {
            Kstar(j+nImp,j+nImp)=ek[j];
            for(auto i=0; i<nImp; i++)
                Kstar(i,j+nImp)=Kstar(j+nImp,i)=vk(i,j);
        }
        Kmat=Kstar;
        F.cols(nImp,L-1)=F.cols(nImp,L-1).eval()*evec;
        impPos=iota(nImp);
    }


    /// (deprecated!) helper: transform Kmat to star geometry (Hbath is diagonal) for tridiagonal matrix Kmat
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

struct Impurity {
    ImpurityParam param;
    Impurity(ImpurityParam const& param_) : param(param_) { param.toStar(); }
};

#endif // IMPURITY_PARAM_H
