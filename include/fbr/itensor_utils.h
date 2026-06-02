#ifndef FBR_ITENSOR_UTILS_H
#define FBR_ITENSOR_UTILS_H

#include "givens_rotation.h"
#include <itensor/all.h>

namespace fbr {

struct DmrgParam {
    int max_bond_dim=512;
    int nIter_diag=4;
    double noise=1e-8;
};

/// Control parameters for a TDVP time step: bond-dimension limits, local Krylov evolution, and addBasis local basis expansion.
struct TdvpParam {
    // --- pure TDVP parameters ---
    int max_bond_dim=1024;  ///< Maximum MPS bond dimension during the TDVP sweep.
    double noise=0;         ///< ITensor sweep noise term.
    int nIter_diag=16;      ///< Krylov iterations used to apply exp(-i * Heff * dt) locally.
    double err_goal=1e-6;   ///< TDVP local evolution error goal.
    // --- addBasis (global subspace expansion) parameters ---
    double epsilonM=1e-5;   ///< addBasis density-matrix cutoff; set to 0 to skip basis expansion.
    int nKrylov=3;          ///< Krylov order of the addBasis global subspace expansion
    double epsilonK=1e-6;   ///< add basis cutoff for each Krylov-vector
};

inline arma::cx_mat getCc(itensor::Fermion const& sites, itensor::MPS const& psi)
{
    arma::cx_mat cc(sites.length(), sites.length());
    auto ccz=correlationMatrixC(psi, sites,"Cdag","C");
    for(auto i=0u; i<ccz.size(); i++)
        for(auto j=0u; j<ccz[i].size(); j++)
            cc(i,j)=ccz.at(i).at(j);
    return cc;
}

inline arma::vec getNi(itensor::Fermion const& sites, itensor::MPS const& psi)
{
    arma::vec ni(sites.length());
    auto niz=expectC(psi, sites,"N");
    for(auto i=0u; i<sites.length(); i++)
        ni[i]=std::real(niz[i]);
    return ni;
}

template<class T>
std::vector<itensor::BondGate> NOGates(itensor::Fermion const& sites, std::vector<GivensRot<T>> const& gs)
{
    using itensor::BondGate;
    std::vector<itensor::BondGate> gates;
    for(const GivensRot<T>& g : gs)
    {
        int b=g.b+1;
        auto rot=g.matrix().t().eval();

        auto s1 = itensor::dag(sites(b));
        auto s2 = itensor::dag(sites(b+1));
        auto s1p = prime(sites(b));
        auto s2p = prime(sites(b+1));
        itensor::ITensor hterm(s1,s2,s1p,s2p);
        hterm.set(s1(1),s2(1),s1p(1),s2p(1), 1);
        hterm.set(s1(2),s2(2),s1p(2),s2p(2), 1);
        hterm.set(s1(2),s2(1),s1p(2),s2p(1), rot(0,0));
        hterm.set(s1(2),s2(1),s1p(1),s2p(2), rot(0,1));
        hterm.set(s1(1),s2(2),s1p(2),s2p(1), rot(1,0));
        hterm.set(s1(1),s2(2),s1p(1),s2p(2), rot(1,1));

        if (hterm) {
            auto bg=BondGate(sites,b,b+1,hterm);
            gates.push_back(bg);
        }
    }
    return gates;
}

} // namespace fbr

#endif // FBR_ITENSOR_UTILS_H
