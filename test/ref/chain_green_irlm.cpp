// Trusted CHAIN baseline for the impurity Green functions of the spinless IRLM,
// used by test/test_ref_green.cpp to validate the FBR (active-window) result.
//
//     G(i,j,t) = -i <psi0| c_i(t) c_j^dag(0) |psi0>,   i,j in {0,1}
//
// Everything is done in the real-space chain, which is nearest-neighbour and so
// MPS-friendly: plain DMRG for the ground state and plain two-site TDVP for the
// evolution, with no orbital rotation and no interaction picture. Writing
// A=|psi0> and B_j=c_j^dag|psi0> and evolving both,
//     G(i,j,t) = -i <A(t)| c_i |B_j(t)> = -i <c_i^dag A(t) | B_j(t)>,
// which is a plain MPS overlap. The three states A, B_0, B_1 are evolved with
// the same MPO.
//
// The model is the one of example/fbr_dyn_irlm.cpp: L=100 chain of hopping 0.5,
// impurity sites 0 and 1, hybridization V=0.1 between them, e_imp=-U/2 and a
// Hubbard U between the two impurity sites.
//
// Writes one line per time step to output/chain_green_irlm_U<U>_ref.txt, and
// stops at t=20 or as soon as the bond dimension of any of the three states
// reaches 1024, keeping what it has. At U=0 the run also reports its deviation
// from the analytic free-fermion Green function, as a check on itself.
//
// TDVP settings: the chain is nearest-neighbour, so two-site TDVP grows the
// bonds by itself and the global subspace expansion is nearly free of effect
// here. Measured at U=0 against the analytic Green function (L=60, t<=2), all of
// n_krylov=15, n_krylov=2 and no expansion at all agree to 4.3e-8, at 536s, 324s
// and 248s. We keep the expansion at the tuned n_krylov=2: same accuracy as the
// overkill set, and it is still there for the interacting runs, where the
// entanglement grows and it is no longer free.
//
// Usage: chain_green_irlm [U]        (default 0.2)
// Env: GREEN_L, GREEN_NSTEP, GREEN_NKRYLOV, GREEN_EPSM override the defaults.

#include <itensor/all.h>
#include <fbr/itensor_utils.h>
#include <tdvp.h>
#include <basisextension.h>
#include <armadillo>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace arma;
using cmpx=std::complex<double>;

namespace {

/// One time slice of the run.
struct Sample {
    double t;
    cmpx G00, G01;
    int m;
};

void writeReference(string const& path, vector<Sample> const& rows,
                    int L, double U, double V, double dt)
{
    ofstream out(path);
    out<<setprecision(17);
    out<<"chain_green_irlm_ref_v1 L "<<L<<" U "<<U<<" V "<<V
       <<" dt "<<dt<<" steps "<<rows.size()<<"\n";
    out<<"# t ReG00 ImG00 ReG01 ImG01 m\n";
    for (auto const& r : rows)
        out<<r.t<<" "<<r.G00.real()<<" "<<r.G00.imag()<<" "
           <<r.G01.real()<<" "<<r.G01.imag()<<" "<<r.m<<"\n";
}

/// The true fermionic c_site^dag, Jordan-Wigner string included.
void applyCdag(itensor::Fermion const& sites, itensor::MPS& psi, int site)
{
    for (int k=1; k<=site; k++) {          // F is diagonal and unitary
        auto A=sites.op("F",k)*psi(k);
        A.noPrime();
        psi.set(k,A);
    }
    psi.position(site+1);
    auto A=sites.op("Cdag",site+1)*psi(site+1);
    A.noPrime();
    psi.set(site+1,A);
}

itensor::MPO getHamiltonian(itensor::Fermion const& sites, mat const& K, double U)
{
    int L=K.n_rows;
    itensor::AutoMPO h(sites);
    if (std::abs(U)>1e-15) h += U,"N",1,"N",2;      // impurity sites 0 and 1
    for (int i=0; i<L; i++)
        for (int j=0; j<L; j++)
            if (std::abs(K(i,j))>1e-12)
                h += K(i,j),"Cdag",i+1,"C",j+1;
    return itensor::toMPO(h);
}

void findGs(itensor::MPS& psi, itensor::MPO const& mpo)
{
    auto sweeps=itensor::Sweeps(1);
    sweeps.maxdim()=1024;
    sweeps.cutoff()=1e-12;
    sweeps.niter()=4;
    sweeps.noise()=1e-8;
    cout<<"# dmrg sweep m energy"<<endl<<setprecision(12);
    for (int i=0; i<24; i++) {
        double e=itensor::dmrg(psi,mpo,sweeps,{"Quiet",true,"Silent",true});
        if (i%4==3) cout<<"#   "<<i+1<<" "<<itensor::maxLinkDim(psi)<<" "<<e<<endl;
    }
}

double envD(char const* k, double d) { return getenv(k) ? std::stod(getenv(k)) : d; }
int    envI(char const* k, int d)    { return getenv(k) ? std::stoi(getenv(k)) : d; }

void do_tdvp(itensor::MPS& psi, itensor::MPO const& mpo, double dt)
{
    fbr::TdvpParam args{.err_goal=1e-8,
                        .epsilon_M=envD("GREEN_EPSM",1e-4),
                        .n_krylov=envI("GREEN_NKRYLOV",2),
                        .epsilon_K=1e-4};
    auto sweeps=itensor::Sweeps(1);
    sweeps.maxdim()=args.max_bond_dim;
    sweeps.cutoff()=1e-12;
    sweeps.niter()=args.n_iter_diag;
    sweeps.noise()=args.noise;

    if (args.epsilon_M!=0) {
    std::vector<double> epsilon_K(args.n_krylov,args.epsilon_K);
    itensor::addBasis(psi,mpo,epsilon_K,
                      {"Cutoff",args.epsilon_M,
                       "Method","DensityMatrix",
                       "KrylovOrd",args.n_krylov,
                       "DoNormalize",true,
                       "Quiet",true,
                       "Silent",true});
    }

    itensor::tdvp(psi,mpo,-cmpx(0,1)*dt,sweeps,
                  {"Truncate",true,
                   "DoNormalize",true,
                   "Quiet",true,
                   "Silent",true,
                   "NumCenter",2,
                   "ErrGoal",args.err_goal});
}

} // namespace

int main(int argc, char** argv)
{
    int L=envI("GREEN_L",100);
    double dt=0.1;
    double V=0.1;
    double U = argc>1 ? std::stod(argv[1]) : 0.2;
    int nStep = getenv("GREEN_NSTEP") ? std::stoi(getenv("GREEN_NSTEP")) : 200;  // t = 20
    int maxBondDim=1024;       // stop as soon as any state reaches this
    int n_part=L/2;

    mat K(L,L,fill::zeros);
    for (int i=1; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
    K(0,1)=K(1,0)=V;
    K(0,0)=K(1,1)=-U/2;

    // analytic free-fermion Green function, for the U=0 self-check
    vec ek; mat evec;
    eig_sym(ek,evec,K);
    auto G_free=[&](int i,int j,double t) {
        cmpx g=0;
        for (int a=n_part; a<L; a++)      // unoccupied modes only
            g += std::exp(-cmpx(0,1)*ek[a]*t)*evec(i,a)*evec(j,a);
        return -cmpx(0,1)*g;
    };

    auto sites=itensor::Fermion(L,{"ConserveNf",true});
    auto mpo=getHamiltonian(sites,K,U);

    itensor::MPS psi;
    {
        auto state=itensor::InitState(sites,"0");
        for (int j=0; j<n_part; j++) state.set(2*j+1,"1");
        psi=itensor::MPS(state);
    }
    findGs(psi,mpo);

    auto A=psi;
    auto B0=psi; applyCdag(sites,B0,0);
    auto B1=psi; applyCdag(sites,B1,1);
    double nrm0=std::sqrt(std::real(itensor::innerC(B0,B0)));
    double nrm1=std::sqrt(std::real(itensor::innerC(B1,B1)));
    B0.normalize();
    B1.normalize();
    A*=cmpx(1,0); B0*=cmpx(1,0); B1*=cmpx(1,0);

    string us = argc>1 ? argv[1] : "0.2";
    string out = "chain_green_irlm_U"+us+"_ref.txt";
    vector<Sample> rows;
    double freeErr=0;

    cout<<"# t ReG00 ImG00 ReG01 ImG01 m"<<endl<<setprecision(12);
    for (int step=0; step<=nStep; step++) {
        double t=step*dt;
        auto A0=A; applyCdag(sites,A0,0);          // <A| c_0
        cmpx G00=-cmpx(0,1)*nrm0*itensor::innerC(A0,B0);
        cmpx G01=-cmpx(0,1)*nrm1*itensor::innerC(A0,B1);
        int m=std::max({itensor::maxLinkDim(A),itensor::maxLinkDim(B0),
                        itensor::maxLinkDim(B1)});
        rows.push_back({t,G00,G01,m});
        writeReference(out,rows,L,U,V,dt);         // keep the file valid throughout

        if (U==0) freeErr=std::max({freeErr,std::abs(G00-G_free(0,0,t)),
                                            std::abs(G01-G_free(0,1,t))});
        cout<<t<<" "<<G00.real()<<" "<<G00.imag()<<" "
            <<G01.real()<<" "<<G01.imag()<<" "<<m<<endl;

        if (m>=maxBondDim) {
            cout<<"# bond dim "<<m<<" reached "<<maxBondDim<<" at t="<<t
                <<"; stopping and keeping what we have"<<endl;
            break;
        }
        if (step<nStep)
            for (auto* p : {&A,&B0,&B1}) do_tdvp(*p,mpo,dt);
    }

    if (U==0) cout<<"# max |G - G_free| = "<<scientific<<freeErr<<endl;
    cout<<"# wrote "<<out<<" with "<<rows.size()<<" rows"<<endl;
    return 0;
}
