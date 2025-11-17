#include"impurity_gs.h"

struct Impurity_dyn {
    ImpurityParam param;
    itensor::Fermion sites;
    itensor::AutoMPO hImp;
    double dt;
    double tol=1e-10;

    /// these quantities are updated during the iterations
    arma::cx_mat K;
    arma::cx_mat rotF;
    itensor::MPS psi;
    double energy=-1000;
    arma::cx_mat cc;
    int nActive;

    Impurity_dyn(itensor::MPS const& psi_, const ImpurityParam& param_, double dt_, double tol_=1e-10)
        : param(param_)
        , sites(itensor::Fermion(param.length(), {"ConserveNf",true}))
        , hImp (sites)
        , dt(dt_)
        , tol(tol_) 
        , K(arma::mat(param_.Kmat))
        , rotF( arma::cx_mat(param_.lenght(),param_.length(), arma::fill::eye) )
        , psi(psi_)
        , nActive(param.nImp())
    {
        for(auto i=0; i<param.nImp(); i++)
            for(auto j=0; j<param.nImp(); j++)
                if (std::abs(param.Umat(i,j))>1e-15)
                    hImp += param.Umat(i,j), "N", i+1, "N", j+1;
    }

    void iterate(DmrgParam args={})
    {
        extract_representative(0);
        extract_representative(1);
        doDmrg(args);
        rotateToNaturalOrbitals();
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef)
    {
        // itensor::cpu_time t0;
        arma::cx_vec ni_bath=cc.diag().eval().rows(nActive,param.length()-1);
        arma::cx_vec delta_n_bath=arma::abs(ni_bath-nRef);
        arma::uvec pos0=arma::find(delta_n_bath<0.5).eval()+nActive ;
        if (pos0.empty()) { std::cout<<"warning: no Slater?\n"; return; }
        auto k12 = K.head_rows(nActive).eval().cols(pos0).eval();
        arma::vec s;
        arma::cx_mat U, V;
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
    }

};