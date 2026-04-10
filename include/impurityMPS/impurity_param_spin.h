#ifndef IMPURITY_PARAM_SPIN_H
#define IMPURITY_PARAM_SPIN_H

#include "impurity_param.h"
#include <armadillo>
#include <itensor/all.h>

struct ImpurityParamSpin {
    arma::mat Kmat;           ///< the kinetic energy coefficient matrix
    arma::mat Umat;           ///< the Coulomb interaction coeff: U(i,j) ni nj
    std::vector<int> impPos_up;  ///< (default => {0,2,...,nImp-2}) the positions of interacting sites
    std::vector<int> impPos_dw;  ///< (default => {1,3,...,nImp-1}) the positions of interacting sites
    double filling=0.5;       ///< number of electrons per site
    arma::mat rot;            ///< (default => identity) the actual frame, such that F*Kmat*F.t() gives the original Kmat (in real space)

    int length() const { return Kmat.n_rows; }
    int nImp() const { return Umat.n_rows; }
    int nPart() const { return filling*length()+0.5; }

    void validate()
    {
        //TODO : verify correctness
        if (rot.empty()) rot=arma::mat(length(),length(), arma::fill::eye);
        if (impPos_up.size() != impPos_dw.size()) throw std::invalid_argument("ImpurityParamSpin: impPos_up != impPos_dw");
        if (impPos_up.empty()) {
            for(auto i=0; i<nImp()/2; i++) {
                impPos_up.push_back(2*i);
                impPos_dw.push_back(2*i+1);
            }
        }
    }

    /// return all (up first) the positions of the impurity
    std::vector<int> impPos() const
    {
        std::vector<int> out;
        for(int i=0; i<nImp()/2; i++) {
            out.push_back(impPos_up[i]);
            out.push_back(impPos_dw[i]);
        }
        return out;
    }

    /// return the positions of each spin up/dw in columns 0/1
    /// The impurity will be at beginning of the positions
    arma::umat split_sites() const // TODO <--------- general disconnected components. Use impPos to classify up/dw
    {
        int L=length();
        // TODO this block will be replaced by graph algorithm
        arma::umat out(L/2,2);
        for(auto i=0; i<L/2; i++) {
            out(i,0)=2*i;
            out(i,1)=2*i+1;
        }

        // put the impurity at the beginning of the positions
        // TODO: take Umat as a graph instead
        for(auto i=0; i<nImp()/2; i++)
        {
            int id_up=arma::find(out.col(0).eval() == impPos_up[i]).eval()[0];
            int id_dw=arma::find(out.col(1).eval() == impPos_dw[i]).eval()[0];
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
            split.print("split");
            arma::uvec pos_all=arma::join_vert(arma::reverse(split.col(0)),split.col(1));
            Kmat=Kmat.submat(pos_all,pos_all).eval();
            rot=rot.cols(pos_all).eval();

            /// TODO: reorder the Umat accordingly
            // using namespace arma;
            // auto ip_up=conv_to<uvec>::from(impPos_up);
            // auto ip_dw=conv_to<uvec>::from(impPos_dw);
            // for(auto i=0; i<nImp()/2; i++) {
            //     int id_up_new=L/2-i-1;
            //     int id_dw_new=L/2+i;
            //     int id_up=arma::find(ip_up == id_up_new).eval()[0];
            //     int id_dw=arma::find(ip_dw == id_dw_new).eval()[0];
            //     Umat.swap_cols(id_up,id_up);
            //     Umat.swap_rows(id_up,id_up);
            //     Umat.swap_cols(id_dw,id_dw);
            //     Umat.swap_rows(id_dw,id_dw);
            // }

        }

        // build an artificial impurity with one of the spin
        arma::mat Umat(nImp()/2,nImp()/2);
        ImpurityParam half={.Kmat=Kmat.submat(L/2,L/2,L-1,L-1), .Umat=Umat};
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
