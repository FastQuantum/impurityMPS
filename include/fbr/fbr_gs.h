#ifndef FBR_GS_H
#define FBR_GS_H

#include "itensor_utils.h"
#include "impurity_param.h"
#include "fb_mps.h"
#include "initial_state.h"

namespace fbr {

/// Ground state of an impurity model: a DMRG sweep on the active window,
/// then the orbital rotations that move the window's edges. One solver for the
/// three layouts -- the window [a,b) is [0,n_active) for `leading`, and
/// ensure_symmetry only does anything under `spin_symmetric`.
struct Fbr_gs {
    ImpurityParam param;

    /// these quantities are updated during the iterations
    Fb_mps<double> fb;
    arma::mat K;
    double energy=-1000;

    Fbr_gs(ImpurityParam const& param_, Fb_mps<double> const& fb_)
        : param(param_)
        , fb { fb_ }
        , K(param.Kmat)
    {
        param.validate();   // a model built directly in star geometry never saw to_star()
    }

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
        fb.ensure_symmetry(K);
        fb.apply(update);
    }

    void do_dmrg(DmrgParam args={})
    {
        auto [a,b]=fb.range(Part::active);
        auto mpo=full_hamiltonian(a,b);
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.n_iter_diag;
        sweeps.noise() = args.noise;
        fb.psi.position(a+1);
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"Minb",a+1,"MaxSite",b,"Quiet", true, "Silent", true});
        energy += fb.slater_energy(K);
        fb.update_cc();
    }

    /// the MPO of the interacting Hamiltonian: the full Umat plus the kinetic
    /// block [a,b) of the current K
    itensor::MPO full_hamiltonian(int a,int b) const
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

#endif // FBR_GS_H
