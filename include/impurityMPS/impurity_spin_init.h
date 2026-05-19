#ifndef IMPURITY_SPIN_INIT_H
#define IMPURITY_SPIN_INIT_H

#include "impurity_param_spin.h"
#include "fb_mps_spin.h"

/// Initialize both `model` (ImpuritySpin) and `fb` (Fb_mps_spin) from
/// Kmat and Umat given in **arbitrary site ordering**.
///
/// The two spin sectors are identified automatically via the graph connected
/// components of Kmat (one component per spin).  The impurity sites within
/// each component are specified by `impPos` (flat list of all impurity site
/// indices in the original Kmat ordering); graph membership determines which
/// ones are spin-up and which are spin-down.
///
/// After construction the internal Kmat is in star geometry (Hbath diagonal),
/// and `fb` holds the corresponding Slater initial state.
struct ImpuritySpinInit {
    ImpuritySpin model;
    Fb_mps_spin<cmpx> fb;

    /// @param Kmat     single-particle Hamiltonian in arbitrary site ordering (L×L)
    /// @param Umat     interaction matrix indexed by impurity sites (nImp×nImp)
    /// @param impPos   indices (in Kmat ordering) of all nImp impurity sites.
    ///                 **Convention**: within each spin block the physically-interacting
    ///                 sites (those appearing off-diagonally in Umat) must come before
    ///                 any buffer/auxiliary sites.  This ensures the rotation produced
    ///                 by toStar() is consistent with the Kmat diagonal after the
    ///                 spin-reflection step.
    /// @param filling  electrons per site (default 0.5)
    ImpuritySpinInit(arma::mat const& Kmat, arma::mat const& Umat,
                     std::vector<int> const& impPos, double filling=0.5)
    {
        // Split impPos into up/dw using graph connected components of Kmat
        auto islands = graph::find_islands(Kmat);
        int comp_up = islands[impPos[0]];

        std::vector<int> imp_up, imp_dw;
        for (int p : impPos) {
            if (islands[p] == comp_up) imp_up.push_back(p);
            else                       imp_dw.push_back(p);
        }
        if (imp_up.size() != imp_dw.size())
            throw std::invalid_argument("ImpuritySpinInit: unequal number of up/dw impurity sites");

        ImpurityParamSpin param;
        param.Kmat     = Kmat;
        param.Umat     = Umat;
        param.filling  = filling;
        param.impPos1_up = imp_up;
        param.impPos1_dw = imp_dw;

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
