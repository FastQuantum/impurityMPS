#ifndef IMPURITY_PARAM_SPIN_H
#define IMPURITY_PARAM_SPIN_H

#include "impurity_param.h"
#include <armadillo>
#include <itensor/all.h>

struct ImpurityParamSpin {
    arma::mat Kmat;           ///< the kinetic energy coefficient matrix
    arma::mat Umat;           ///< the Coulomb interaction coeff: U(i,j) ni nj
    std::vector<int> impPos1_up;  ///< the positions of spin-up interacting sites (after toStar)
    std::vector<int> impPos1_dw;  ///< the positions of spin-dw interacting sites (after toStar)
    std::vector<int> impPos0_up;  ///< the positions of spin-up interacting sites (before toStar)
    std::vector<int> impPos0_dw;  ///< the positions of spin-dw interacting sites (before toStar)
    double filling=0.5;       ///< number of electrons per site
    arma::mat rot;            ///< (default => identity) the actual frame, such that F*Kmat*F.t() gives the original Kmat (in real space)

    int length() const { return Kmat.n_rows; }
    int nImp() const { return Umat.n_rows; }
    int nPart() const { return filling*length()+0.5; }

    void validate()
    {
        //TODO : verify correctness
        if (rot.empty()) rot=arma::mat(length(),length(), arma::fill::eye);
        if (impPos1_up.size() != impPos1_dw.size()) throw std::invalid_argument("ImpurityParamSpin: impPos_up != impPos_dw");
        if (impPos1_up.empty()) {
            for(auto i=0; i<nImp()/2; i++) {
                impPos1_up.push_back(2*i);
                impPos1_dw.push_back(2*i+1);
            }
        }
        impPos0_up.resize(nImp()/2);
        impPos0_dw.resize(nImp()/2);
    }

    /// return all (up first) the positions of the impurity
    std::vector<int> impPos() const
    {
        std::vector<int> out;
        for(int i=0; i<nImp()/2; i++) out.push_back(impPos1_up[i]);
        for(int i=0; i<nImp()/2; i++) out.push_back(impPos1_dw[i]);
        return out;
    }

    /// return all (up first) the positions of the impurity
    std::vector<int> impPos0() const
    {
        std::vector<int> out;
        for(int i=0; i<nImp()/2; i++) out.push_back(impPos0_up[i]);
        for(int i=0; i<nImp()/2; i++) out.push_back(impPos0_dw[i]);
        return out;
    }

    /// Split all sites into two spin columns using graph connected components of Kmat.
    /// Column 0 = spin-up component (the one containing impPos1_up[0]).
    /// Column 1 = spin-dw component.
    /// Within each column the impurity sites are placed first.
    arma::umat split_sites() const
    {
        int L=length();

        auto islands = graph::find_islands(Kmat);
        int comp_up = islands[impPos1_up[0]];

        std::vector<int> sites_up, sites_dw;
        for (int i=0; i<L; i++) {
            if (islands[i] == comp_up) sites_up.push_back(i);
            else                        sites_dw.push_back(i);
        }
        if ((int)sites_up.size() != L/2 || (int)sites_dw.size() != L/2)
            throw std::runtime_error("split_sites: each spin component must have exactly L/2 sites");

        arma::umat out(L/2, 2);
        for (int i=0; i<L/2; i++) {
            out(i,0) = sites_up[i];
            out(i,1) = sites_dw[i];
        }

        for(auto i=0; i<nImp()/2; i++)
        {
            int id_up=arma::find(out.col(0).eval() == (arma::uword)impPos1_up[i]).eval()[0];
            int id_dw=arma::find(out.col(1).eval() == (arma::uword)impPos1_dw[i]).eval()[0];
            out.col(0).swap_rows(id_up,i);
            out.col(1).swap_rows(id_dw,i);
        }

        return out;
    }

    /// transform Kmat to star geometry (Hbath is diagonal)
    void toStar()
    {
        validate();
        // TODO : if the matrix is already in star then return *this;
        int L=length();
        { // reorganize the sites
            arma::umat split=split_sites();
            arma::uvec pos_all=arma::join_vert(arma::reverse(split.col(0)),split.col(1));
            Kmat=Kmat.submat(pos_all,pos_all).eval();
            rot=rot.cols(pos_all).eval();

            for(auto i=0; i<nImp()/2; i++) {
                impPos0_up[i]=impPos1_up[i];
                impPos0_dw[i]=impPos1_dw[i];
                impPos1_up[i]=L/2-i-1;
                impPos1_dw[i]=L/2+i;
            }
        }

        // build an artificial impurity with one of the spin
        arma::mat Umat_half(nImp()/2,nImp()/2, arma::fill::zeros);
        ImpurityParam half={.Kmat=Kmat.submat(L/2,L/2,L-1,L-1), .Umat=Umat_half};
        half.toStar();

        // duplicate the artificial impurity by reflexion
        auto irev=arma::regspace<arma::uvec>(L/2-1,0);
        Kmat.submat(L/2,L/2,L-1,L-1)=half.Kmat;
        Kmat.submat(irev,irev)=half.Kmat;
        rot.cols(L/2,L-1)=rot.cols(L/2,L-1).eval()*half.rot;
        rot.cols(irev)=rot.cols(irev).eval()*half.rot;
    }
};

struct ImpuritySpin {
    ImpurityParamSpin param;
    ImpuritySpin()=default;
    ImpuritySpin(ImpurityParamSpin const& param_) : param(param_) { param.toStar(); }
};

#endif // IMPURITY_PARAM_SPIN_H
