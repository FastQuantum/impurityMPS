#ifndef FBR_GS_SPIN_H
#define FBR_GS_SPIN_H

#include "itensor_utils.h"
#include "impurity_param_spin.h"
#include "fb_mps.h"

namespace fbr {

struct Fbr_gs_spin {
    ImpurityParamSpin param;

    /// these quantities are updated during the iterations
    Fb_mps<double> fb;
    arma::mat K;
    double energy=-1000;

    Fbr_gs_spin(ImpuritySpin const& imp, Fb_mps<double> const& fb_)
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
        fb.ensure_symmetry(K);
        fb.applyUpdate(update);
    }

    void doDmrg(DmrgParam args={})
    {
        auto [a,b]=fb.interval_active_full();
        auto mpo=fullHamiltonian(a,b);
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        fb.psi.position(a+1);
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"Minb",a+1,"MaxSite",b,"Quiet", true, "Silent", true});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    /// return the mpo of the Hamiltoninan given by himp and the kinetic energy kin
    itensor::MPO fullHamiltonian(int a,int b) const
    {
        itensor::AutoMPO h(fb.sites);
        int L = param.length();
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (std::abs(param.Umat(i,j)) > 1e-15)
                    h += param.Umat(i,j), "N", i+1, "N", j+1;

        for(auto i=a; i<b; i++)
            for(auto j=a; j<b; j++)
                if (std::abs(K(i,j))>fb.tol)
                    h += K(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }
};


} // namespace fbr

#endif // FBR_GS_SPIN_H
