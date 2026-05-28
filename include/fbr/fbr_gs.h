#ifndef FBR_GS_H
#define FBR_GS_H

#include "itensor_utils.h"
#include "fbr_param.h"
#include "fb_mps.h"

namespace fbr {

struct Fbr_gs {
    FbrParam param;

    /// these quantities are updated during the iterations
    Fb_mps<double> fb;
    arma::mat K;
    double energy=-1000;

    Fbr_gs(Fbr const& imp, Fb_mps<double> const& fb_)
        : param(imp.param)
        , fb { fb_ }
        , K(param.Kmat)
    {}

    void iterate(DmrgParam args={})
    {
        extract_representative(0);
        extract_representative(1);
        doDmrg(args);
        rotateToNaturalOrbitals();
    }

    void extract_representative(int nRef){ fb.extract_representative(K,nRef,fb.nActive); }

    void doDmrg(DmrgParam args={})
    {
        int nA=fb.nActive;
        auto mpo=fullHamiltonian(K.submat(0,0,nA-1,nA-1));
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"MaxSite",nA,"Quiet", true, "Silent", true});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    void rotateToNaturalOrbitals()
    {
        int nA=fb.nActive;
        auto rot1=fb.rotateToNaturalOrbitals(param.nImp());
        K.cols(0,nA-1)=K.cols(0,nA-1).eval()*rot1;
        K.rows(0,nA-1)=rot1.t()*K.rows(0,nA-1).eval();
    }

    itensor::MPO fullHamiltonian(arma::mat const& kin) const
    {
        itensor::AutoMPO h(fb.sites);
        int L=param.length();
        for (int i=0; i<L; i++)
            for (int j=0; j<L; j++)
                if (std::abs(param.Umat(i,j))>1e-15)
                    h += param.Umat(i,j),"N",i+1,"N",j+1;
        for (int i=0; i<(int)kin.n_rows; i++)
            for (int j=0; j<(int)kin.n_cols; j++)
                if (std::abs(kin(i,j))>fb.tol)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }
};

} // namespace fbr

#endif // FBR_GS_H
