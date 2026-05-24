#ifndef IMPURITY_SPIN_INIT_H
#define IMPURITY_SPIN_INIT_H

#include "impurity_param_spin.h"
#include "fb_mps_spin.h"

/// Initialize both `model` (ImpuritySpin) and `fb` (Fb_mps_spin) from
/// Kmat and Umat given in **arbitrary site ordering** (convention 2).
///
/// `impPos` lists all nImp impurity sites in spatial order for the final
/// star layout:
///   impPos[0..nUp-1]      = up impurities, outer -> inner
///   impPos[nUp..nImp-1]   = dw impurities, inner -> outer
/// Up/dw membership is inferred from the connected components of Kmat.
///
/// `Umat` is L×L, indexed by site position in the input Kmat layout:
///   sum_{i,j} Umat(i,j) N_i N_j
struct ImpuritySpinInit {
    ImpuritySpin model;
    Fb_mps_spin<cmpx> fb;

    /// @param Kmat     single-particle Hamiltonian, L×L
    /// @param Umat     L×L interaction matrix in the same site ordering as Kmat
    /// @param impPos   nImp impurity site indices, ordered per convention 2
    /// @param filling  electrons per site (default 0.5)
    ImpuritySpinInit(arma::mat const& Kmat, arma::mat const& Umat,
                     std::vector<int> const& impPos, double filling=0.5)
    {
        ImpurityParamSpin param;
        param.Kmat    = Kmat;
        param.Umat    = Umat;
        param.impPos  = impPos;
        param.filling = filling;

        model = ImpuritySpin(param);   // calls param.toStar() internally
        fb    = makeFb(model.param);
    }

private:
    static Fb_mps_spin<cmpx> makeFb(ImpurityParamSpin const& param)
    {
        arma::vec ek = param.Kmat.diag();
        auto rot = param.rot * cmpx(1,0);
        return Fb_mps_spin<cmpx>::from_slater(rot, ek, param.nPart(), param.nImp());
    }
};

#endif // IMPURITY_SPIN_INIT_H
