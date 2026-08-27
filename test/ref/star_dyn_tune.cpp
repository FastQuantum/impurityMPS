// Parameter-tuning driver for the STAR-geometry SIAM dynamics.
// Same physics/model as star_dyn_siam_center.cpp, but the TDVP subspace-expansion
// parameters and the number of steps are taken from the command line so a sweep can
// be run without recompiling. Snapshots are written to a scratch file whose name
// encodes the parameters; compare against the committed chain reference with
// tune_compare.py.
//
// Usage:
//   star_dyn_tune <U> <n_krylov> <epsilon_M> <epsilon_K> <err_goal> <maxSteps> <outPath>
#include <itensor/all.h>
#include <fbr/itensor_utils.h>
#include <tdvp.h>
#include <basisextension.h>
#include <armadillo>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace arma;

struct Snap {
    string label;
    arma::vec ni;
    arma::cx_mat cc;
};

void writeReference(string const& path, vector<Snap> const& snaps, int L)
{
    ofstream out(path);
    out << setprecision(17);
    out << "chain_dyn_siam_center_ref_v1 L " << L
        << " snapshots " << snaps.size() << "\n";
    for (auto const& s : snaps) {
        out << "snapshot " << s.label << "\n";
        out << "ni";
        for (int i = 0; i < L; i++) out << " " << s.ni[i];
        out << "\n";
        out << "cc";
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                out << " " << s.cc(i, j).real() << " " << s.cc(i, j).imag();
        out << "\n";
    }
}

auto computeKstar(mat K, int n_imp)
{
    int L=K.n_rows;
    int nBath=L/2-n_imp/2;

    mat Kstar(L,L,arma::fill::zeros);
    mat rot(L,L,fill::eye);

    auto pos_up=regspace<uvec>(0,L/2-1);
    auto pos_dw=regspace<uvec>(L/2,L-1);

    for(int s : {0,1})
    {
        uvec pos   = s==0 ? pos_up : pos_dw;
        uvec pos_bath = s==0 ? pos.head(nBath)   : pos.tail(nBath);
        uvec pos_impu = s==0 ? pos.tail(n_imp/2)  : pos.head(n_imp/2);

        mat Kbath=K.submat(pos_bath,pos_bath);
        mat evec1; vec ek1;
        eig_sym(ek1,evec1,Kbath);
        uvec iek = s==0 ? sort_index(abs(ek1),"descend") : sort_index(abs(ek1));
        arma::mat evec=evec1.cols(iek);
        arma::vec ek=ek1.rows(iek);

        arma::mat vk=K.submat(pos_impu,pos_bath).eval()*evec;
        Kstar.submat(pos_impu,pos_impu)=K.submat(pos_impu,pos_impu);

        for(auto j=0u;j<ek.size();j++) {
            int jj=pos_bath[j];
            Kstar(jj,jj)=ek[j];
            for(auto i=0u;i<pos_impu.size();i++) {
                int ii=pos_impu[i];
                Kstar(ii,jj)=Kstar(jj,ii)=vk(i,j);
            }
        }
        rot.cols(pos_bath)=rot.cols(pos_bath).eval()*evec;
    }

    return make_pair(Kstar, rot);
}

void do_tdvp(itensor::MPS &psi, itensor::MPO const mpo, double dt,
            fbr::TdvpParam const& args, double tol=1e-12)
{
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = args.max_bond_dim;
    sweeps.cutoff() = tol;
    sweeps.niter() = args.n_iter_diag;
    sweeps.noise() = args.noise;

    std::vector<double> epsilon_K(args.n_krylov, args.epsilon_K);
    itensor::addBasis(psi, mpo, epsilon_K,
                      {"Cutoff", args.epsilon_M,
                       "Method", "DensityMatrix",
                       "KrylovOrd", args.n_krylov,
                       "DoNormalize", true,
                       "Quiet", true,
                       "Silent", true});

    using cmpx=complex<double>;
    itensor::tdvp(psi,mpo, -cmpx(0,1)*dt, sweeps,
                  {"Truncate", true,
                   "DoNormalize", true,
                   "Quiet", true,
                   "Silent", true,
                   "NumCenter", 2,
                   "ErrGoal", args.err_goal});
}

itensor::MPO getHamiltonian(itensor::Fermion sites, mat const& K, mat const& Umat)
{
    double tol=1e-12;
    int L=K.n_rows;
    int n_imp=Umat.n_rows;
    int nBath=L/2-n_imp/2;
    itensor::AutoMPO h(sites);
    for(auto i=0; i<n_imp; i++)
        for(auto j=0; j<n_imp; j++) {
            int ii=nBath+i;
            int jj=nBath+j;
            if (std::abs(Umat(i,j))>1e-15)
                h += Umat(i,j), "N", ii+1, "N", jj+1;
        }

    for(auto i=0; i<L; i++)
        for(auto j=0; j<L; j++)
            if (std::abs(K(i,j))>tol)
                h += K(i,j),"Cdag",i+1,"C",j+1;
    return itensor::toMPO(h);
}

int main(int argc, char** argv)
{
    int L=100;
    int n_imp=4;
    double dt=0.1;
    int nBath=L/2-n_imp/2;

    double U        = argc>1 ? std::stod(argv[1]) : 0.2;
    fbr::TdvpParam args;
    args.err_goal = argc>5 ? std::stod(argv[5]) : 1e-8;
    args.epsilon_M = argc>3 ? std::stod(argv[3]) : 1e-7;
    args.n_krylov  = argc>2 ? std::stoi(argv[2]) : 15;
    args.epsilon_K = argc>4 ? std::stod(argv[4]) : 1e-8;
    int maxSteps  = argc>6 ? std::stoi(argv[6]) : 200;
    string out    = argc>7 ? argv[7] : "output/tune_star_ref.txt";

    int maxBondDim = 1024;

    mat Kstar, Umat;
    mat rot;
    {
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        {
            for(auto i=0; i<L/2-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            for(auto i=L/2; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            K(nBath+n_imp/2-1, nBath+n_imp/2-1)=-U/2;
            K(L/2, L/2)=-U/2;
            K(nBath, nBath+n_imp/2-1)=K(nBath+n_imp/2-1, nBath)=V;
            K(L/2, L/2+n_imp/2-1)=K(L/2+n_imp/2-1, L/2)=V;
        }
        Umat.zeros(n_imp,n_imp);
        Umat(n_imp/2-1,n_imp/2)=U;

        std::tie(Kstar, rot) = computeKstar(K, n_imp);
    }
    cx_mat cxrot = conv_to<cx_mat>::from(rot);

    itensor::Fermion sites=itensor::Fermion(L, {"ConserveNf",true});
    itensor::MPS psi;
    {
        auto ek=arma::vec {Kstar.diag()};
        ek[nBath+n_imp/2-1]=ek[L/2]=-10;
        ek[nBath]=ek[L/2+n_imp/2-1]=10;

        int n_part=L/2;
        sites=itensor::Fermion(ek.size(), {"ConserveNf",true});
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        for(int j = 0; j < n_part; j++) {
            int k=iek[j];
            state.set(k+1,"1");
        }
        psi=itensor::MPS(state);
    }

    auto mpo=getHamiltonian(sites,Kstar,Umat);

    vector<pair<int,string>> wanted = {
        {0, "initial"}, {1, "t=0.1"}, {50, "t=5.0"}, {100, "t=10.0"}, {200, "t=20.0"},
    };
    int nSteps = maxSteps;
    vector<Snap> snaps;

    auto capture = [&](int step, string const& label) {
        cx_mat cc = cxrot * fbr::get_cc(sites, psi) * cxrot.t();
        snaps.push_back({label, arma::real(cc.diag()), cc});
        writeReference(out, snaps, L);
    };
    capture(0, "initial");

    cerr<<"# U="<<U<<" n_krylov="<<args.n_krylov<<" epsM="<<args.epsilon_M
        <<" epsK="<<args.epsilon_K<<" errGoal="<<args.err_goal<<" maxSteps="<<maxSteps<<"\n";
    cerr<<"time m n_dw n_dw_bf wall\n"<<setprecision(12);
    itensor::cpu_time t0;
    for(auto i=0;i<nSteps;i++){
        do_tdvp(psi,mpo,dt,args);
        int step=i+1;
        int m=itensor::maxLinkDim(psi);
        double n_dw=itensor::expectC(psi,sites,"N",{nBath+n_imp/2+1})[0].real();
        double n_dw_bf=itensor::expectC(psi,sites,"N",{nBath+n_imp/2+2})[0].real();
        cerr<<step*dt<<" "<<m<<" "<<n_dw<<" "<<n_dw_bf<<" "<<t0.sincemark().wall<<endl;
        t0.mark();
        for(auto const& w: wanted)
            if(w.first==step) capture(step, w.second);
        if(m>maxBondDim){
            cerr<<"# bond dim "<<m<<" exceeded "<<maxBondDim<<" at t="<<step*dt<<"\n";
            break;
        }
    }
    cerr<<"wrote "<<out<<" with "<<snaps.size()<<" snapshots\n";
    return 0;
}
