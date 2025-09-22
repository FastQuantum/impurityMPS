#include<catch2/catch.hpp>
#include<itensor/all.h>
#include<armadillo>

using namespace arma;
using namespace std;

itensor::MPO getHamiltonian(itensor::SpinHalf sites)
{
    auto ampo = itensor::AutoMPO(sites);
    for(int j = 1; j < sites.length(); ++j)
    {
        ampo += "Sz",j,"Sz",j+1;
        ampo += 0.5,"S+",j,"S-",j+1;
        ampo += 0.5,"S-",j,"S+",j+1;
    }
    return toMPO(ampo);
}

itensor::MPS prepapeState(itensor::SpinHalf sites)
{
    using namespace itensor;

    auto state = itensor::InitState(sites);
    for(int i = 1; i <= sites.length(); ++i)
    {
        if(i%2 == 0) state.set(i,"Up");
        else         state.set(i,"Dn");
    }

    auto H=getHamiltonian(sites);

    auto sweeps = itensor::Sweeps(10);
    sweeps.maxdim() = 10,20,100,100,200;
    sweeps.cutoff() = 1E-10;
    sweeps.niter() = 2;
    sweeps.noise() = 1E-7,1E-8,0.0;
    println(sweeps);

    auto [en0,psi0] = dmrg(H,randomMPS(state),sweeps,{"Quiet=",true});
    cout<<"energy "<<en0<<endl;
    return psi0;
}

auto addTWoSites(itensor::SpinHalf sites, itensor::MPS const& psi)
{
    using namespace itensor;
    // psi.replaceSiteInds(hamsys.sites.inds());
    int N=psi.length();
    auto state = itensor::InitState(sites);
    for(int i = 1; i <= sites.length(); ++i)
    {
        if(i%2 == 0) state.set(i,"Up");
        else         state.set(i,"Dn");
    }
    auto psi2 = itensor::MPS(state);
    for(int i = 1; i+1 <= psi.length(); ++i)
        psi2.Aref(i)=psi(i);
    { // manually copy the last tensor
        auto a=commonIndex(psi(N-1), psi(N));
        auto s=uniqueIndex(psi(N),psi(N-1));
        auto b=commonIndex(psi2(N),psi2(N+1));
        auto T=ITensor(a.dag(),s,b);
        for(auto ai:range1(a))
            for(auto si:range1(s)) {
                auto value=psi(N).eltC(a(ai), s(si));
                if (std::abs(value)==0.0) continue;
                T.set(a(ai), s(si), b(1), value);
            }
        psi2.set(N,T);
    }
    psi2.replaceSiteInds(sites.inds());
    return psi2;
}

TEST_CASE("itensor copy mps") {
    cout<<"copying mps"<<endl;

    int N = 20;
    auto sites = itensor::SpinHalf(N,{"ConserveQNs=",true});
    itensor::MPS psi=prepapeState(sites);
    cout<<"energy1 "<<itensor::inner(psi,getHamiltonian(sites),psi)<<endl;
    {
        auto sites = itensor::SpinHalf(psi.length()+2,{"ConserveQNs=",true});
        auto psi2=addTWoSites(sites,psi);
        cout<<"energy2 "<<itensor::inner(psi2,getHamiltonian(sites),psi2)<<endl;
    }
}
