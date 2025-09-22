#ifndef IMPURITY_GS_H
#define IMPURITY_GS_H

#include "givens_rotation.h"
#include "fermionic.h"

#include <armadillo>
#include <itensor/all.h>

struct ImpurityParam {
    arma::sp_mat Kstar;     ///< the kinetic energy coefficient matrix in star geometry (Hbath is diagonal)
    arma::sp_mat Umat;      ///< the Coulomb interaction coeff: U(i,j) ni nj
    double filling=0.5;     ///< number of electrons per site

    int length() const { return Kstar.n_rows; }
    int nImp() const { return Umat.n_rows; }
    int nPart() const { return filling*length()+0.5; }

    /// helper: transform Kmat to star geometry (Hbath is diagonal)
    static arma::sp_mat to_star_kin(arma::mat const& Kmat, int nImp)
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
        return Kstar;
    }

    /// helper: transform Kmat to star geometry (Hbath is diagonal) for tridiagonal matrix Kmat
    static arma::sp_mat to_star_kin_tridiag(arma::sp_mat const& Kmat, int nImp)
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


struct Impurity_gs {
    ImpurityParam param;
    itensor::Fermion sites;
    itensor::AutoMPO hImp;
    double tol=1e-10;

    /// these quantities are updated during the iterations
    arma::mat K;
    itensor::MPS psi;
    double energy=-1000;
    arma::mat cc;
    int nActive;

    explicit Impurity_gs(const ImpurityParam& param_, double tol_=1e-10)
        : param(param_)
        , sites(itensor::Fermion(param.length(), {"ConserveNf",false}))
        , hImp (sites)
        , tol(tol_)
        , K(arma::mat(param_.Kstar))
        , nActive(param.nImp())
    {
        prepareSlaterGs(K.diag(), param.length()/2);
        for(auto i=0; i<param.nImp(); i++)
            for(auto j=0; j<param.nImp(); j++)
                if (std::abs(param.Umat(i,j))>1e-15)
                    hImp += param.Umat(i,j), "N", i+1, "N", j+1;
    }

    void iterate(DmrgParam args={})
    {
        extract_f(0.0);
        extract_f(1.0);
        doDmrg(args);
        rotateToNaturalOrbitals();
    }

    /// extract f orbital of the sites with ni=0 or 1
    void extract_f(double nRef)
    {
        itensor::cpu_time t0;
        arma::vec ni_bath=cc.diag().eval().rows(nActive,param.length()-1);
        arma::vec delta_n_bath=arma::abs(ni_bath-nRef);
        arma::uvec pos0=arma::find(delta_n_bath<0.5).eval()+nActive ;
        if (pos0.empty()) { std::cout<<"warning: no Slater?\n"; return; }
        auto k12 = K.head_rows(nActive).eval().cols(pos0).eval();
        arma::vec s;
        arma::mat U, V;
        svd_econ(U,s,V, k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);

        auto Kcol=K.cols(pos0).eval();
        applyGivens(Kcol,givens);
        K.cols(pos0)=Kcol;
        std::cout<<" givens to K 1"<<t0.sincemark()<<std::endl; t0.mark();

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
        std::cout<<" givens to K 2"<<t0.sincemark()<<std::endl; t0.mark();
        // no need to update cc
        for(auto i=0; i<nSv; i++) {
            SlaterSwap(nActive,pos0.at(i));
            nActive++;
        }
    }

    void doDmrg(DmrgParam args={})
    {
        auto mpo=fullHamiltonian( K.submat(0,0,nActive-1,nActive-1) );
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(psi,mpo,sweeps, {"MaxSite",nActive,"Quiet", true, "Silent", true});
        energy += SlaterEnergy();
        auto ccz=correlationMatrix(psi, sites,"Cdag","C",itensor::range1(nActive));
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc(i,j)=ccz.at(i).at(j);
    }

    void rotateToNaturalOrbitals()
    {
        auto cc1=cc.submat(param.nImp(),param.nImp(),nActive-1, nActive-1).eval();
        auto givens=GivensRotForCC_right(cc1);
        for(auto& g:givens) g.b+=param.nImp();
        auto gates=Fermionic::NOGates(sites,givens);
        gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
        auto rot1=matrot_from_Givens(givens,nActive);
        cc.cols(0,nActive-1)=cc.cols(0,nActive-1).eval()*rot1.t();
        cc.rows(0,nActive-1)=rot1*cc.rows(0,nActive-1).eval();
        K.cols(0,nActive-1)=K.cols(0,nActive-1).eval()*rot1.st();
        K.rows(0,nActive-1)=rot1.st().t()*K.rows(0,nActive-1).eval();
        auto ni_bath=arma::real(cc.diag()).eval().rows(param.nImp(),param.length()-1).eval();
        nActive=arma::find(ni_bath>tol && ni_bath<1-tol).eval().size()+param.nImp();
    }

    void prepareSlaterGs(arma::vec ek, int nPart)
    {
        cc=arma::mat(ek.size(), ek.size(), arma::fill::zeros);        
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        double energy=0;
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            energy += ek[k];
            cc(k,k)=1;
        }
        psi=itensor::MPS(state);
    }

    double SlaterEnergy() const
    {
        double energy=0;
        for(auto i=nActive; i<cc.n_rows; i++)
            energy += cc(i,i)*K(i,i);
        return energy;
    }

    /// return the mpo of the Hamiltoninan given by himp and the kinetic energy kin
    itensor::MPO fullHamiltonian(arma::mat const& kin) const
    {
        auto h=hImp;
        for(auto i=0; i<kin.n_rows; i++)
            for(auto j=0; j<kin.n_cols; j++)
                if (std::abs(kin(i,j))>tol)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }

private:
    /// Swap to sites inside the Slater part
    void SlaterSwap(int i,int j)
    {
        if (i==j) return;
        if (i<nActive || j<nActive) throw std::runtime_error("SlaterSwap for active orbitals");
        if (std::abs(cc(i,i)-cc(j,j))<0.5) throw std::runtime_error("SlaterSwap for equal occupations");
        K.swap_cols(i,j);
        K.swap_rows(i,j);

        auto flip=[&](int p) {
            auto G = cc(p,p)>0.5 ? sites.op("A",p+1) : sites.op("Adag",p+1) ;
            auto newA = G*psi(p+1);
            newA.noPrime();
            psi.set(p+1,newA);
        };
        flip(i);
        flip(j);

        cc.swap_cols(i,j);
        cc.swap_rows(i,j);
    }

};


#endif // IMPURITY_GS_H
