#ifndef FB_MPS_H
#define FB_MPS_H

#include "givens_rotation.h"
#include "fermionic.h"

#include <armadillo>
#include <itensor/all.h>

template<class T>
struct Fb_mps
{
    itensor::Fermion sites;
    itensor::MPS psi;
    arma::Sp_mat<T> cc;
    int nActive;
    double tol=1e-10;

    struct from_slater {};

    Fb_mps(from_slater tag,vec ek, int nPart, int nActive_)
    : sites (itensor::Fermion(param.length(), {"ConserveNf",true}))
    , cc (arma::Mat<T>(ek.size(), ek.size(), arma::fill::zeros))
    , nActive (nActive_)
    {                
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            cc(k,k)=1;
        }
        psi=itensor::MPS(state);
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    std::vector<GivensRot<T>> extract_representative(arma::Mat<T>& K, int nRef)
    {
        // itensor::cpu_time t0;
        arma::vec ni_bath=cc.diag().eval().rows(nActive,param.length()-1);
        arma::vec delta_n_bath=arma::abs(ni_bath-nRef);
        arma::uvec pos0=arma::find(delta_n_bath<0.5).eval()+nActive ;
        if (pos0.empty()) { std::cout<<"warning: no Slater?\n"; return; }
        auto k12 = K.head_rows(nActive).eval().cols(pos0).eval();
        arma::vec s;
        arma::Mat<T> U, V;
        svd_econ(U,s,V, k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);

        auto Kcol=K.cols(pos0).eval();
        applyGivens(Kcol,givens);
        K.cols(pos0)=Kcol;
        // std::cout<<" givens to K 1"<<t0.sincemark()<<std::endl; t0.mark();

        {
            arma::inplace_trans(K);
            auto Kcol=K.cols(pos0).eval();
            applyGivens(Kcol,givens);
            K.cols(pos0)=Kcol;
            arma::inplace_trans(K);
            // auto Krow=K.rows(pos0).eval();
            // applyGivens(GivensDagger(givens),Krow);
            // K.rows(pos0)=Krow;
        }
        // std::cout<<" givens to K 2"<<t0.sincemark()<<std::endl; t0.mark();
        // no need to update cc
        for(auto i=0; i<nSv; i++) {
            SlaterSwap(nActive,pos0.at(i));
            nActive++;
        }
        return givens;
    }

    /// compute the cc in the active sector using the psi
    void compute_cc()
    {
        auto ccz=correlationMatrix(psi, sites,"Cdag","C",itensor::range1(nActive));
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc(i,j)=ccz.at(i).at(j);
    }

    /// return the rotation Q applied to the active sector: ci=Qij*dj (where ci are the old destroy operators)
    arma::Mat<T> rotateToNaturalOrbitals()
    {
        auto cc1=cc.submat(param.nImp(),param.nImp(),nActive-1, nActive-1).eval();
        auto givens=GivensRotForCC_right(cc1);
        for(auto& g:givens) g.b+=param.nImp();
        auto gates=Fermionic::NOGates(sites,givens);
        gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
        auto rot1=matrot_from_Givens(givens,nActive);
        cc.cols(0,nActive-1)=cc.cols(0,nActive-1).eval()*rot1.t();
        cc.rows(0,nActive-1)=rot1*cc.rows(0,nActive-1).eval();
        auto ni_bath=arma::real(cc.diag()).eval().rows(param.nImp(),param.length()-1).eval();
        nActive=arma::find(ni_bath>tol && ni_bath<1-tol).eval().size()+param.nImp();
        return rot1.st();
    }

    double SlaterEnergy() const
    {
        double energy=0;
        for(auto i=nActive; i<cc.n_rows; i++)
            energy += cc(i,i)*K(i,i);
        return energy;
    }

};


#endif // FB_MPS_H


