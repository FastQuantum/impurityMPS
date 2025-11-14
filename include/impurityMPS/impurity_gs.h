#ifndef IMPURITY_GS_H
#define IMPURITY_GS_H

#include "givens_rotation.h"
#include "fermionic.h"
#include "impurity_param.h"
#include "fb_mps.h"

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
