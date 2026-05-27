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

struct TdvpParam {
    int max_bond_dim=1024;
    int nIter_diag=16;
    double noise=0;
    double epsilonM=1e-8;
    int nKrylov=15;
    double err_goal=1e-8;
};

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
