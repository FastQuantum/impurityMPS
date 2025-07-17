#include "impurityMPS/irlm.h"
#include "impurityMPS/it_dmrg.h"
#include "impurityMPS/it_tdvp.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include<itensor/all.h>
#include <nlohmann/json.hpp>

using namespace std;

struct State {
    itensor::MPS psi;
    itensor::Fermion sites;
};

auto computeGS(HamSys const& sys)
{
    cout<<"bond dimension of H: "<< maxLinkDim(sys.ham) << endl;
    it_dmrg sol_gs {sys};
    sol_gs.bond_dim=64;
    sol_gs.noise=1e-3;
    cout<<"\nsweep bond-dim energy\n";
    for(auto i=0u; i<30; i++) {
        if (i==10) { sol_gs.noise=1e-7; sol_gs.bond_dim=256; }
        else if (i==26) { sol_gs.noise=1e-9; sol_gs.bond_dim=512; }
        else if (i==28) sol_gs.noise=0;
        if (false && i==11) {
            // Global subspace expansion
//            std::vector<double> epsilonK(3,1E-8);
            std::vector<int> maxDimK(3,0.5*itensor::maxLinkDim(sol_gs.psi));

            itensor::addBasis(sol_gs.psi,sol_gs.hamsys.ham,maxDimK,
                     {"Cutoff", 1e-9,
                      "Method", "DensityMatrix",
                      "KrylovOrd",3,
                      "DoNormalize", true,
                      "Quiet",true,
                      "Silent",true});
        }
        sol_gs.iterate();
        //sol_gs.psi.orthogonalize({"Cutoff",1e-9});
        cout<<i+1<<" "<<maxLinkDim(sol_gs.psi)<<" "<<sol_gs.energy<<endl;
    }
    for(auto i=0; i<sol_gs.psi.length(); i++)
        cout<<itensor::leftLinkIndex(sol_gs.psi,i+1).dim()<<" ";
    cout << "\n";
    return sol_gs;
}


auto prepareSlater(itensor::Fermion const& sites, arma::vec ek, int nPart)
{
    auto state = itensor::InitState(sites,"0");
    arma::uvec iek=arma::sort_index(ek);
    double energy=0;
    for(int j = 0; j < nPart; j++) {
        state.set(iek[j]+1,"1");
        energy += ek[iek[j]];
    }
    std::cout << " Slater energy: " << energy << std::endl;
    return itensor::MPS(state);
}

/// return a rotation rot such that rot(:,B).t()*kin*rot(:,B) is diagonal where B are the inactive eigenvectors of cc (with eval=0 or 1).
template<class T>
arma::Mat<T> MagicRotation(arma::Mat<T> const& cc, arma::Mat<T> const& kin, double tolWannier)
{
    using namespace arma;
    Mat<T> rot(size(cc), fill::eye);
    uint nInactive=0;
    {// inactive evec of cc
       Mat<T> evec;
       vec eval;
       eig_sym(eval,evec,cc);
       { //sort by activity
           arma::vec activity(eval.size());
           for(auto i=0u; i<eval.size(); i++)
               activity[i]=-std::min(eval[i], -eval[i]+1);
           arma::uvec iev=arma::stable_sort_index(activity.clean(1e-15)); //activity sorting
           eval=eval(iev);
           evec=evec.cols(iev);
       }
       uvec active=arma::find(eval>tolWannier && eval<1-tolWannier);
       uvec inactive=arma::find(eval<=tolWannier || eval>=1-tolWannier);
       rot.head_cols(active.size())=evec.cols(active);                      // TODO: Wannierize the active sector (?)
       rot.tail_cols(inactive.size())=evec.cols(inactive);
       nInactive=inactive.size();
    }
    {// diagonalize kin in the subspace of inactive orbitals
        auto rot1=rot.tail_cols(nInactive);  // this is a view!
        Mat<T> X=rot1.t() * kin * rot1;
        Mat<T> evec;
        vec eval;
        eig_sym(eval,evec,X);
        if (false){ //sort by position
            arma::mat xOp=arma::diagmat(arma::regspace(0,cc.n_rows-1));
            vec x=arma::real((evec.t()*rot1.t()*xOp*rot1*evec).eval().diag());
            uvec iev=stable_sort_index(x);
            evec=evec.cols(iev).eval();
        }
        rot1=rot1*evec;
    }

    // cout<<"n inactive = "<<nInactive<<endl;
    // cout<<"rot.t * rot-1 ="<<norm(rot.t()*rot-mat(size(rot),fill::eye))<<endl;
    // (rot.t()*cc*rot).eval().clean(1e-15).print("cc rotated");
    // (rot.t()*kin*rot).eval().clean(1e-15).print("kin rotated");

    // fix the sign
    for(auto j=0u; j<rot.n_cols; j++) {
        auto i=arma::index_max(arma::abs(rot.col(j)));
        rot.col(j) /= arma::sign(rot(i,j));
    }

    cout<<"norm(1-rot)="<<norm(rot-arma::mat(size(rot),fill::eye))<<endl;

    return rot;
}


NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(IRLM,L,t,V,U,ed,connected)


int main()
{
    using namespace itensor;
    using namespace arma;
    using namespace nlohmann;
    using matriz=arma::cx_mat;

    json j;
    {
        std::ifstream in("param.json");
        if (!in) throw invalid_argument("Please provide the param.json file");
        j=json::parse(in);
    }

    cout<<setw(4)<<j;

    bool save=j.value("save",false);
    bool verbose=j.at("verbose");
    int circuit_nImp=j.at("circuit").at("nImp");
    int circuit_nSite=j.at("circuit").at("nSite");
    double circuit_dt=j.at("circuit").at("dt");
    double circuit_tol=j.at("circuit").at("tol");
    double dt=j.at("tdvp").at("dt");

    IRLM m1 = j.at("irlm");
    int len=m1.L, nExclude=2;
    itensor::Fermion sites(len, {"ConserveNf",true});
    auto model0=IRLM_ip {sites, j.at("irlm0"), dt};
    auto model=IRLM_ip {sites, j.at("irlm"), dt};

    cout<<"\n-------------------------- solve the gs1 ----------------\n" << setprecision(15);

    arma::mat K=model0.irlm.matrices().first;
    arma::cx_mat rot = model0.irlm.rotStar() * cx_double(1,0);

    arma::vec ek=arma::real((rot.t()*K*rot).eval().diag().as_col());
    ek[1] += model0.irlm.U;
    auto psi=prepareSlater(sites, ek, len/2);
    matriz cc=Fermionic::cc_matrix(psi, sites);
    auto cck=Fermionic::cc_matrix_kondo(psi, sites);

    if (save) cc.save("cc_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);
    if (save) cck.save("cck_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);
    if (save) rot.save("orb_L"s+to_string(len)+"_t"+to_string(0)+".txt",arma::raw_ascii);


    cout<<"\n-------------------------- evolve the psi with new Hamiltonian ----------------\n"; cout.flush();
    double tolActivity=j.at("ip").at("tolActivity");

    ofstream out("irlm_no_L"s+to_string(len)+".txt");
    out<<"time M m energy n0 cd nActive\n"<<setprecision(14);
    double n0=itensor::expectC(psi, sites, "N",{1}).at(0).real();
    auto cdOp=[&](itensor::Fermion const& sites) {
        itensor::AutoMPO ampo(sites);
        ampo += "Cdag",1,"C",2;
        ampo += "Cdag",2,"C",1;
        return itensor::toMPO(ampo);
    };
    double cd=itensor::innerC(psi, cdOp(sites), psi).real();
    vec ni=arma::real(cc.diag());
    arma::uvec active=arma::find(ni>tolActivity && ni<1-tolActivity);
    out<<"0 "<< 0 <<" "<<maxLinkDim(psi)<<" "<<0<<" "<<n0<<" "<<cd<<" "<<active.size()<<endl;
    auto model2_ip=IRLM_ip{model};
    for(auto i=0; i*dt<=len; i++) {
        if (verbose) cout<<"-------------------------- iteration "<<i+1<<" --------\n";
        itensor::cpu_time t0;
        int nImpIp=map<string,int> {{"circuit", circuit_nImp},
                                    {"activity", std::max(circuit_nImp,(int)active.size())},
                                    {"none", len}
                                   }.at(j.at("ip").at("type"));
        //auto [sys2,givens,Kip] = model2_ip.HamIP_f(rot,nImpIp,dt, j.at("extract_f"));
        auto hip=model2_ip.HamIP_f3(rot, nImpIp, ni, dt, j.at("extract_f"),tolActivity);
//        auto [sys2,givens,Kip] = model2_ip.HamIPS(rot, nImpIp, dt, j.at("extract_f"));
        psi.replaceSiteInds(hip.ham.sites.inds());
        if (hip.from != hip.to && hip.to!=len) { // swap sites
            itensor::AutoMPO ampo(sites);
            ampo += "Cdag",hip.from+1, "C",hip.to+1;
            ampo += "Cdag",hip.to+1,"C",hip.from+1;
            auto op = itensor::toMPO(ampo);
            psi = applyMPO(op,psi);
            psi.replaceSiteInds(hip.ham.sites.inds());
            cc.swap_cols(hip.from, hip.to);
            cc.swap_rows(hip.from, hip.to);
            if (save) {
                cck.swap_cols(hip.from, hip.to);
                cck.swap_rows(hip.from, hip.to);
            }
        }
        // arma::real(Fermionic::cc_matrix(psi, hip.ham.sites).diag()).print("ni when hip");
        if (nImpIp!=len) rot = hip.rot;
//        if (nImpIp!=len) { rot = rot * matrot_from_Givens(givens,len).st();  }
        if (verbose) cout<<"Hamiltonian mpo:"<<t0.sincemark()<<endl;
        t0.mark();

        if (j.at("extract_f")) {// the circuit to extract f orbitals
            auto gates=Fermionic::NOGates(hip.ham.sites,hip.givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",circuit_tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
            if (verbose) cout<<"circuit-f:"<<t0.sincemark()<<endl;
            t0.mark();
        }

        // if (verbose) arma::real(Fermionic::cc_matrix(psi, hip.ham.sites).diag()).print("ni after f");

        if (j.at("use_tdvp")) {// tdvp
            it_tdvp sol {hip.ham, psi};
            sol.dt={0,dt};
            sol.bond_dim=512;
            sol.rho_cutoff=1e-14;
            sol.silent=true;
            sol.epsilonM=j.at("tdvp").at("epsilonM");
            sol.nKrylov=j.at("tdvp").at("nKrylov");
            sol.enrichByFit = false; //(i%10!=0);

            sol.iterate();
            psi=sol.psi;
        }
        else {
            //Kip.clean(1e-13).print("Kip");
            //cout<<"U="<<model2_ip.irlm.U<<endl;
            auto gates=model2_ip.TrotterGatesExp(hip.Kip,3,dt);
            // auto gates=model2_ip.TrotterGates(Kip,3,dt);
            //Time evolve, overwriting psi when done
            gateTEvol(gates,1,1,psi,{"Cutoff=",circuit_tol,"Quiet=",true, "Normalize",false,"ShowPercent",false});
        }

        // if (verbose) arma::real(Fermionic::cc_matrix(psi, hip.ham.sites).diag()).print("ni after Trotter");

        if (false && j.at("extract_f")) {// the circuit to extract f orbitals applied in reverse
            auto givens=hip.givens;
            std::reverse(givens.begin(), givens.end());
            for(auto& g:givens) g=g.dagger();
            auto gates=Fermionic::NOGates(hip.ham.sites,givens);
            gateTEvol(gates,1,1,psi,{"Cutoff",circuit_tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
            if (verbose) cout<<"circuit-f:"<<t0.sincemark()<<endl;
            t0.mark();
        }

        if (verbose) cout<<"tdvp time"<<t0.sincemark()<<endl;
        t0.mark();


        // cc=Fermionic::cc_matrix(psi, hip.ham.sites);
        cc.submat(0,0,hip.from,hip.from)=Fermionic::cc_matrix(psi, hip.ham.sites, hip.from+1)* cx_double(1,0);
        if (save) {
            cck=cc*cc(0,0);
            cck.submat(0,0,hip.from,hip.from)=Fermionic::cc_matrix_kondo(psi, hip.ham.sites, hip.from+1)* cx_double(1,0);
        }
        if (verbose) cout<<"cc computation:"<<t0.sincemark()<<endl;
        t0.mark();

        if (std::abs(i*dt-std::round(i*dt/circuit_dt)*circuit_dt) < 0.5*dt) {
            auto givens=Fermionic::NOGivensRot(cc,circuit_nImp,circuit_nSite,tolActivity, hip.from);
//            auto givens=Fermionic::GivensRotForMatrix(cc,circuit_nImp,20);
            //real((rot1 * cc * rot1.t()).eval().clean(1e-10).submat(circuit_nImp,circuit_nImp,cc.n_rows-1,cc.n_cols-1)).print("rot1*cc*rot1.t()");
            auto gates=Fermionic::NOGates(hip.ham.sites,givens);
            if (verbose) cout<<"circuit1 algebra:"<<t0.sincemark()<<endl;
            gateTEvol(gates,1,1,psi,{"Cutoff",circuit_tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
            if (verbose) cout<<"circuit1:"<<t0.sincemark()<<endl;
            t0.mark();
            //matriz ccr=Fermionic::cc_matrix(psi, sol.hamsys.sites);
            //real(ccr.clean(1e-7).submat(circuit_nImp,circuit_nImp,cc.n_rows-1,cc.n_cols-1)).print("cc after rot");
            auto rot1=matrot_from_Givens(givens,hip.from+1);
            rot.cols(0,hip.from) = rot.cols(0,hip.from).eval()*rot1.st();
            cc.cols(0,hip.from)=cc.cols(0,hip.from).eval()*rot1.t();
            cc.rows(0,hip.from)=rot1*cc.rows(0,hip.from).eval();
            if (save) {
                cck.cols(0,hip.from)=cck.cols(0,hip.from).eval()*rot1.t();
                cck.rows(0,hip.from)=rot1*cck.rows(0,hip.from).eval();
            }
            //psi.orthogonalize({"Cutoff",circuit_tol});
            ni=arma::real(cc.diag());
            // ni=arma::real(Fermionic::cc_matrix(psi, hip.ham.sites).diag());
            active=arma::find(ni>tolActivity && ni<1-tolActivity);

            if (verbose) {
                cout<<"active: "<<active.size()<<endl;
                for(auto i=0; i<psi.length(); i++)
                    cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
                cout<<endl;
            }

            // if (verbose) arma::real(Fermionic::cc_matrix(psi, hip.ham.sites).diag()).print("ni after circuit");
        }


        double n0=itensor::expectC(psi, hip.ham.sites, "N",{1}).at(0).real();
        cd=itensor::innerC(psi, cdOp(hip.ham.sites), psi).real();
        out<<(i+1)*abs(dt)<<" "<< maxLinkDim(hip.ham.ham) <<" "<<maxLinkDim(psi)<<" "<<0<<" "<<n0<<" "<<cd<<" "<<active.size()<<endl;

        if (save && (i>0) && (i%100==0)) {
            cc.save("cc_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            cck.save("cck_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            rot.save("orb_L"s+to_string(len)+"_t"+to_string(i)+".txt",arma::raw_ascii);
            //exportPsi(psi,"psi_t"s+to_string(i)+".txt");
        }        
    }

    return 0;
}
