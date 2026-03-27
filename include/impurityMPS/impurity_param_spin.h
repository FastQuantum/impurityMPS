#ifndef IMPURITY_PARAM_SPIN_H
#define IMPURITY_PARAM_SPIN_H

#include "impurity_param.h"
#include "impurityMPS/fermionic.h"
#include <armadillo>
#include <itensor/all.h>

struct ImpurityParamSpin {
    arma::mat Kmat;           ///< the kinetic energy coefficient matrix
    arma::mat Umat;           ///< the Coulomb interaction coeff: U(i,j) ni nj
    std::vector<int> impPos;  ///< (default => {0,1,...,nImp-1}) the positions of interacting sites
    double filling=0.5;       ///< number of electrons per site
    arma::mat rot;            ///< (default => identity) the actual frame, such that F*Kmat*F.t() gives the original Kmat (in real space)

    int length() const { return Kmat.n_rows; }
    int nImp() const { return Umat.n_rows; }
    int nPart() const { return filling*length()+0.5; }

    void initializeDefault()
    {
        //TODO : verify correctness
        if (rot.empty()) rot=arma::mat(length(),length(), arma::fill::eye);
        if (impPos.empty()) impPos=iota(nImp());
    }

    arma::umat split_sites() const // TODO <--------- general disconnected components
    {
        int L=length();
        arma::umat out(L/2,2);
        for(auto i=0; i<L/2; i++) {
            out(i,0)=2*i;
            out(i,1)=2*i+1;
        }
        return out;
    }

    /// transform Kmat to star geometry (Hbath is diagonal)
    void toStar()
    {
        initializeDefault();        
        // TODO : if the matrix is already in star then return *this;

        //create a non-spin version
        arma::umat split=split_sites();
        arma::uvec pos0=split.col(0);
        arma::mat Umat(1,1);
        ImpurityParam half={.Kmat=Kmat(pos0,pos0), .Umat=Umat};
        half.toStar();

        split.print("split");


        // duplicate non-spin by reflexion
        int L=length();
        arma::mat Kstar(L,L,arma::fill::zeros);
        Kstar.submat(L/2,L/2,L-1,L-1)=half.Kmat;
        auto i_left=iota(L/2);
        std::reverse(i_left.begin(), i_left.end());
        auto iset=arma::conv_to<arma::uvec>::from(i_left);
        Kstar.submat(iset,iset)=half.Kmat;

        Kmat=Kstar;
        impPos[0]=L/2-1;
        impPos[1]=L/2;
        //TODO: update the rotation using split

    }
};

struct ImpuritySpin {
    ImpurityParamSpin param;
    ImpuritySpin(ImpurityParamSpin const& param_) : param(param_) { param.toStar(); }
};

#endif // IMPURITY_PARAM_SPIN_H
