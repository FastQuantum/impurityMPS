#include "impurityMPS/irlm_gs.h"
#include "impurityMPS/it_dmrg.h"
// #include "impurityMPS/it_tdvp.h"
#include "impurityMPS/irlm_gs.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include<itensor/all.h>
#include <nlohmann/json.hpp>

using namespace std;

auto computeGS(itensor::Fermion const& sites, itensor::MPO const& ham, itensor::MPS *psi=nullptr)
{
    cout<<"bond dimension of H: "<< maxLinkDim(ham) << endl;
    it_dmrg sol_gs = psi ? it_dmrg {HamSys{.sites=sites,.ham=ham}, *psi} :
                           it_dmrg {HamSys{.sites=sites,.ham=ham}};
    sol_gs.bond_dim=64;
    sol_gs.noise=1e-3;
    sol_gs.silent=true;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<32; i++) {
        if (i==10) { sol_gs.noise=1e-7; sol_gs.bond_dim=256; }
        else if (i==26) { sol_gs.noise=1e-9; sol_gs.bond_dim=512; }
        else if (i==28) sol_gs.noise=0;
        sol_gs.iterate();
        cout<<i+1<<" "<<maxLinkDim(sol_gs.psi)<<" "<<sol_gs.energy<<endl;
    }
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";
    return sol_gs;
}


NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IrlmData,L,t,V,U,ed,connected)

int main()
{    
    nlohmann::json j;
    {
        std::ifstream in("param.json");
        if (!in) throw invalid_argument("Please provide the param.json file");
        j=nlohmann::json::parse(in);
    }

    itensor::cpu_time t0;
    t0.mark();
    auto model0=Irlm_gs {j.at("irlm_gs")};
    cout<<"initialization: "<<t0.sincemark().wall<<endl;
    t0.mark();

    cout<<"iteration nActive energy time\n"<<setprecision(12);
    for(auto i=0;i<30;i++){
        model0.extract_f(0.0);
        model0.extract_f(1.0);
        cout<<"extract f: "<<t0.sincemark()<<"  "; t0.mark();
        model0.doDmrg();
        cout<<"dmrg: "<<t0.sincemark()<<"  "; t0.mark();
        model0.rotateToNaturalOrbitals();
        cout<<"nat orb: "<<t0.sincemark()<<"  "; t0.mark();
        cout<<i+1<<" "<<model0.nActive<<" "<<model0.energy<<endl;
    }

    // cout<<"\n\nNormal dmrg\n";
    // auto sol_gs=computeGS(model0.sites,model0.fullHamiltonian(true));

    return 0;
}
