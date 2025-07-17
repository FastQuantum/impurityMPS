#include "impurityMPS/irlm.h"
#include "impurityMPS/it_dmrg.h"
#include "impurityMPS/it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;


auto computeGS(HamSys const& sys, itensor::MPS psi={})
{
    cout<<"bond dimension of H: "<< maxLinkDim(sys.ham) << endl <<setprecision(14);
    auto sol_gs = psi ? it_dmrg {sys,psi} : it_dmrg {sys};
    sol_gs.bond_dim=512;
    sol_gs.noise=1e-3;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<30; i++) {
        if (i==10) sol_gs.noise=1e-7;
        else if (i==26) sol_gs.noise=1e-9;
        else if (i==28) sol_gs.noise=0;
        sol_gs.iterate();
        //sol_gs.psi.orthogonalize({"Cutoff",1e-9});
        cout<<i+1<<" "<<maxLinkDim(sol_gs.psi)<<" "<<sol_gs.energy<<endl;
    }
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";
    return sol_gs;
}


/// ./irlm_star <len> [star=true] [dt=0.1] [U=0.25]
int main(int argc, char **argv)
{
    bool star=true;
    int len=20;
    double dt=0.1;
    double U=0.25;
    if (argc>=2) len=atoi(argv[1]);
    if (argc>=3) star=atoi(argv[2]);
    if (argc>=4) dt=atof(argv[3]);
    if (argc>=5) U=atof(argv[4]);
    auto model0=IRLM {.L=len, .t=0.5, .V=0.0, .U=U, .ed=-10};
    auto model1=IRLM {.L=len, .t=0.5, .V=0.0, .U=U, .ed=-10, .connected=false};
    auto model2=IRLM {.L=len, .t=0.5, .V=0.1, .U=U, .ed=0.0};

    cout<<"\n-------------------------- find the gs1 ----------------\n";

    auto sol_gs0=computeGS(star ? model0.HamStar() : model0.Ham());
    auto sol_gs=computeGS(star ? model1.HamStar() : model1.Ham(), sol_gs0.psi);

    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n";


    auto sys2= star ? model2.HamStar() : model2.Ham();
    cout<<"bond dimension of H: "<< maxLinkDim(sys2.ham) << endl;
    it_tdvp sol {sys2, sol_gs.psi};
    sol.dt={0,dt};
    sol.bond_dim=1024;
    sol.rho_cutoff=1e-12;
    sol.epsilonM=0; //1e-10;
    sol.silent=true;
    sol.enrichByFit = false;
    ofstream out("irlm_star_L"s+to_string(sol.hamsys.ham.length())+".txt");
    out<<"sweep bond-dim energy n0\n"<<setprecision(14);
    double n0=itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real();
    auto cdOp=[&](itensor::Fermion const& sites) {
        itensor::AutoMPO ampo(sites);
        ampo += "Cdag",1,"C",2;
        ampo += "Cdag",2,"C",1;
        return itensor::toMPO(ampo);
    };
    double cd=itensor::innerC(sol.psi, cdOp(sol.hamsys.sites), sol.psi).real();
    out<<"0 "<< maxLinkDim(sys2.ham)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<" "<<n0<<" "<<cd<<endl;
    for(auto i=0; i*dt<=len; i++) {
        // sol.epsilonM=(i%1==0) ? 1e-7 : 0;
        sol.iterate();
        if (false && i%100==0) {
            auto cc=Fermionic::cc_matrix(sol.psi, sol.hamsys.sites);
            string filename="eval_L"s+to_string(sol.hamsys.ham.length())+"_t"+to_string(i)+".txt";
            arma::eig_sym(cc).save(filename,arma::raw_ascii);
        }

        double n0=itensor::expectC(sol.psi, sol.hamsys.sites, "N",{1}).at(0).real();
        cd=itensor::innerC(sol.psi, cdOp(sol.hamsys.sites), sol.psi).real();
        out<<(i+1)*abs(sol.dt)<<" "<< maxLinkDim(sys2.ham)<<" "<<maxLinkDim(sol.psi)<<" "<<sol.energy<<" "<<n0<<" "<<cd<<endl;
    }

    return 0;
}

