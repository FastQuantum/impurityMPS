#ifndef IT_TDVP_H
#define IT_TDVP_H

#include "fermionic.h"
#include "tdvp.h"
#include "basisextension.h"

template<class HamS>
struct it_tdvp {
    int bond_dim=64;
    int nIter_diag=32;
    double rho_cutoff=1e-10;
    double epsilonM=1e-8;
    std::complex<double> dt={0, 0.1};
    bool do_normalize=true;
    double err_goal=1e-8;
    bool silent=true;
    bool enrichByFit=false;
    int nKrylov=3;

    int nsweep=0;
    double energy=0;
    HamS hamsys;
    itensor::MPS psi;

    it_tdvp(HamS const& hamsys_, itensor::MPS const& psi_) : hamsys(hamsys_), psi(psi_)
    {
        psi.replaceSiteInds(hamsys.sites.inds());
        if constexpr (std::is_same_v<HamS,HamSys>)
            energy=std::real(itensor::innerC(psi, hamsys.ham, psi));
        else {
            energy=0;
            for(auto const& x:hamsys.ham)
                energy += std::real(itensor::innerC(psi, x, psi));
        }

    }

    std::complex<double> time() const { return dt * static_cast<double>(nsweep); }

    void iterate()
    {
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = bond_dim;
        sweeps.cutoff() = rho_cutoff;
        sweeps.niter() = nIter_diag;

        if (epsilonM != 0)
        {
            // Global subspace expansion
           std::vector<double> epsilonK(nKrylov,1E-8);
           //std::vector<int> maxDimK(5,0.5*itensor::maxLinkDim(psi));

            if (hamsys.hamEnrich.length()==0) throw std::invalid_argument("hamsys.hamEnrich need to be defined for tdvp");
            itensor::addBasis(psi,hamsys.ham,epsilonK,
                     {"Cutoff",epsilonM,
                      "Method",enrichByFit ? "Fit" : "DensityMatrix",
                      "KrylovOrd",nKrylov,
                      "DoNormalize", do_normalize,
                      "Quiet",true,
                      "Silent",silent});
        }

        // TDVP sweep
        energy = itensor::tdvp(psi,hamsys.ham,-dt,sweeps,
                      {"Truncate",true,
                       "DoNormalize", false,
                       "Quiet",true,
                       "Silent",silent,
                       "NumCenter",2,
                       "ErrGoal", err_goal});

        nsweep++;
    }
};

#endif // IT_TDVP_H
