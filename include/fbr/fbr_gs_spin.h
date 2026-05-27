#ifndef FBR_GS_SPIN_H
#define FBR_GS_SPIN_H

#include "itensor_utils.h"
#include "fbr_param_spin.h"
#include "fb_mps_spin.h"

namespace fbr {

struct Fbr_gs_spin {
    FbrParamSpin param;

    /// these quantities are updated during the iterations
    Fb_mps_spin<double> fb;
    arma::mat K;
    double energy=-1000;

    Fbr_gs_spin(FbrSpin const& imp, Fb_mps_spin<double> const& fb_)
        : param(imp.param)
        , fb { fb_ }
        , K(param.Kmat)
    {}

    void iterate(DmrgParam args={})
    {
        itensor::cpu_time t0;
        extract_representative(0);
        extract_representative(1);
        doDmrg(args);
        rotateToNaturalOrbitals();
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef,/*use_active=*/true); }

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

    void rotateToNaturalOrbitals()
    {
        auto [a,b]=fb.interval_active_full(); // the interval will change
        auto rot1=fb.rotateToNaturalOrbitals();
        K.cols(a,b-1)=K.cols(a,b-1).eval()*rot1;
        K.rows(a,b-1)=rot1.t()*K.rows(a,b-1).eval();
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
