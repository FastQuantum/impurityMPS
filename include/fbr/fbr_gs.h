#ifndef FBR_GS_H
#define FBR_GS_H

#include "itensor_utils.h"
#include "impurity_param.h"
#include "fb_mps.h"
#include "initial_state.h"

namespace fbr {

struct Fbr_gs {
    ImpurityParam param;

    /// these quantities are updated during the iterations
    Fb_mps<double> fb;
    arma::mat K;
    double energy=-1000;

    Fbr_gs(Impurity const& imp, Fb_mps<double> const& fb_)
        : param(imp.param)
        , fb { fb_ }
        , K(param.Kmat)
    {}

    void iterate(DmrgParam args={})
    {
        applyPlan(fb.planRepresentative(K,0,/*use_active=*/true));
        applyPlan(fb.planRepresentative(K,1,/*use_active=*/true));
        doDmrg(args);
        applyPlan(fb.planNaturalOrbitals(fb.cc));
    }

    void applyPlan(OrbitalUpdate<double> const& update)
    {
        update.applyAsBasis(K);
        fb.applyUpdate(update);
    }

    void doDmrg(DmrgParam args={})
    {
        int nA=fb.nActive();
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
