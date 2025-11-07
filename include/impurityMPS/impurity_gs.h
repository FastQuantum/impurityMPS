#ifndef IMPURITY_GS_H
#define IMPURITY_GS_H

#include "givens_rotation.h"
#include "fermionic.h"
#include "fb_mps.h"

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

    /// these quantities are updated during the iterations
    Fb_mps<double> fb;
    arma::mat K;
    double energy=-1000;    

    explicit Impurity_gs(const ImpurityParam& param_)
        : param(param_)
        , fb { prepareSlater(param_) }
        , K(arma::mat(param_.Kstar))
    {}

    void iterate(DmrgParam args={})
    {
        extract_representative(0);
        extract_representative(1);
        doDmrg(args);
        rotateToNaturalOrbitals();
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef); }

    void doDmrg(DmrgParam args={})
    {
        auto mpo=fullHamiltonian( K.submat(0,0,fb.nActive-1,fb.nActive-1) );
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"MaxSite",fb.nActive,"Quiet", true, "Silent", true});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    void rotateToNaturalOrbitals()
    {
        int nA=fb.nActive; // it will change
        auto rot1=fb.rotateToNaturalOrbitals(param.nImp());
        K.cols(0,nA-1)=K.cols(0,nA-1).eval()*rot1;
        K.rows(0,nA-1)=rot1.t()*K.rows(0,nA-1).eval();
    }

    void prepareSlaterGs(arma::vec const& ek) { fb=Fb_mps<double>::from_slater(ek,param.nPart(),param.nImp()); }

    double SlaterEnergy() const { return fb.SlaterEnergy(K); }

    /// return the mpo of the Hamiltoninan given by himp and the kinetic energy kin
    itensor::MPO fullHamiltonian(arma::mat const& kin) const
    {
        itensor::AutoMPO h(fb.sites);
        for(auto i=0; i<param.nImp(); i++)
            for(auto j=0; j<param.nImp(); j++)
                if (std::abs(param.Umat(i,j))>1e-15)
                    h += param.Umat(i,j), "N", i+1, "N", j+1;
        for(auto i=0; i<kin.n_rows; i++)
            for(auto j=0; j<kin.n_cols; j++)
                if (std::abs(kin(i,j))>fb.tol)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }

private:
    static Fb_mps<double> prepareSlater(ImpurityParam const& param)
    {
        arma::vec ek=arma::vec(param.Kstar.diag());
        return Fb_mps<double>::from_slater(ek, param.nPart(), param.nImp());
    }
};


#endif // IMPURITY_GS_H
