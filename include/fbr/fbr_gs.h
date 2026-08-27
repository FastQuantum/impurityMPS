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
        apply_plan(fb.plan_representative(K,0,/*use_active=*/true));
        apply_plan(fb.plan_representative(K,1,/*use_active=*/true));
        do_dmrg(args);
        apply_plan(fb.plan_natural_orbitals(fb.cc));
    }

    void apply_plan(OrbitalUpdate<double> const& update)
    {
        update.apply_as_basis(K);
        fb.apply(update);
    }

    void do_dmrg(DmrgParam args={})
    {
        int nA=fb.n_active();
        auto mpo=full_hamiltonian(K.submat(0,0,nA-1,nA-1));
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.n_iter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"MaxSite",nA,"Quiet", true, "Silent", true});
        energy += fb.slater_energy(K);
        fb.update_cc();
    }

    itensor::MPO full_hamiltonian(arma::mat const& kin) const
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
