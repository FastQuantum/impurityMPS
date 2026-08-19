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

// One measured time slice of the trusted chain run: occupations and the full
// one-particle correlation matrix <c_i^dag c_j>, in chain-site order.
struct Snap {
    string label;
    arma::vec ni;
    arma::cx_mat cc;
};

// Serialize snapshots in the chain_dyn_siam_center_ref_v1 format read by
// test/test_ref_*.cpp (whitespace-token stream: header, then per snapshot a
// label, an "ni" vector of length L, and a row-major "cc" of L*L (re im) pairs).
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

void findGs(itensor::MPS &psi, itensor::MPO const mpo, double tol=1e-12)
{
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = 1024;
    sweeps.cutoff() = 1e-10;
    sweeps.niter() = 4;
    sweeps.noise() = 1e-8;

    cout<<"iteration m energy\n"<<setprecision(12);
    for(auto i=0;i<20;i++){
        double energy=itensor::dmrg(psi,mpo,sweeps, {/*"MaxSite",fb.nActive,*/"Quiet", true, "Silent", true});
        cout<<i+1<<" "<<itensor::maxLinkDim(psi)<<" "<<energy<<endl;
    }
}

void doTdvp(itensor::MPS &psi, itensor::MPO const mpo, double dt, double tol=1e-12)
{
    fbr::TdvpParam args{.err_goal=1e-8, .epsilonM=1e-4, .nKrylov=15, .epsilonK=1e-4};
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = args.max_bond_dim;
    sweeps.cutoff() = tol;
    sweeps.niter() = args.nIter_diag;
    sweeps.noise() = args.noise;

    std::vector<double> epsilonK(args.nKrylov, args.epsilonK);
    itensor::addBasis(psi, mpo, epsilonK,
                      {"Cutoff", args.epsilonM,
                       "Method", "DensityMatrix",
                       "KrylovOrd", args.nKrylov,
                       "DoNormalize", true,
                       "Quiet", true,
                       "Silent", true});

    using cmpx=std::complex<double>;
    itensor::tdvp(psi,mpo, -cmpx(0,1)*dt, sweeps,          // TDVP sweep
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
    int nImp=Umat.n_rows;
    int nBath=L/2-nImp/2;  // impurity cluster occupies sites [nBath, nBath+nImp)
    itensor::AutoMPO h(sites);
    for(auto i=0; i<nImp; i++)
        for(auto j=0; j<nImp; j++) {
            int ii=nBath+i;  // these positions change with nImp and nBath
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
    int nImp=4;
    double dt=0.1;
    int nBath=L/2-nImp/2;  // =4 for L=12, nImp=4
    double U = argc>1 ? std::stod(argv[1]) : 0.2;   // Hubbard U (default 0.2)
    int maxBondDim = 1024;   // stop the run (and keep snapshots so far) if exceeded

    mat Kchain, Umat; // define the Hamiltonian
    {
        double V=0.1;
        arma::mat K(L,L, arma::fill::zeros);
        {
            // spin-up chain (sites 0..L/2-1) and spin-down chain (sites L/2..L-1)
            for(auto i=0; i<L/2-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            for(auto i=L/2; i<L-1; i++) K(i,i+1)=K(i+1,i)=0.5;
            // impurity on-site energies: spin-up imp at site nBath+nImp/2-1, spin-down at L/2
            K(nBath+nImp/2-1, nBath+nImp/2-1)=-U/2;
            K(L/2, L/2)=-U/2;
            // hybridization V: overwrites the chain hop between buffer and physical impurity
            K(nBath, nBath+nImp/2-1)=K(nBath+nImp/2-1, nBath)=V;      // spin-up
            K(L/2, L/2+nImp/2-1)=K(L/2+nImp/2-1, L/2)=V;              // spin-down
        }
        Umat.zeros(nImp,nImp);
        Umat(nImp/2-1,nImp/2)=U;  // Hubbard U between spin-up imp (cluster idx nImp/2-1) and spin-down imp (nImp/2)

        //std::tie(Kstar,rot) = computeKstar(K, nImp);
        Kchain=K;
    }

    itensor::Fermion sites=itensor::Fermion(L, {"ConserveNf",true});
    itensor::MPS psi;  // should be  bath--|0110|--bath
    {
        int nPart=L/2;
        auto state = itensor::InitState(sites,"0");
        for(int j = 0; j < nPart; j++) state.set(2*j+1,"1");
        state.set(L/2-1,"0");
        state.set(L/2,"1");
        state.set(L/2+1,"1");
        state.set(L/2+2,"0");
        psi=itensor::MPS(state);
    }

    {  //psi should be the gs of  bath--|0110|--bath
        auto Kfake=Kchain;
        Kfake(nBath,nBath-1)=Kfake(nBath-1,nBath)=0;
        int b2=L/2+nImp/2;
        Kfake(b2-1,b2)=Kfake(b2,b2-1)=0;
        Kfake(nBath, nBath+nImp/2-1) = Kfake(nBath+nImp/2-1, nBath) = 0; // decouple spin-up buffer and imp
        Kfake(L/2, L/2+nImp/2-1) = Kfake(L/2+nImp/2-1, L/2) = 0;         // decouple spin-down imp and buffer
        // Bias cluster sites so DMRG cannot change the occupation away from |0110|
        Kfake(L/2-2,L/2-2)=10;;
        Kfake(L/2-1,L/2-1)=-10;
        Kfake(L/2,L/2)=-10;
        Kfake(L/2+1,L/2+1)=10;

        auto mpo=getHamiltonian(sites,Kfake,Umat);
        findGs(psi,mpo);        
    }

    auto mpo=getHamiltonian(sites,Kchain,Umat);

    // Steps at which to snapshot the full state (dt=0.1): t = 0, 0.1, 5, 10, 20.
    // The first three reproduce the committed reference; t=10 and t=20 extend it
    // to catch long-time drift between the chain baseline and the FBR run.
    vector<pair<int,string>> wanted = {
        {0, "initial"}, {1, "t=0.1"}, {50, "t=5.0"}, {100, "t=10.0"}, {200, "t=20.0"},
    };
    int nSteps = wanted.back().first;
    vector<Snap> snaps;

    // Rewrite the whole file after each snapshot so a run interrupted early (e.g.
    // killed once the bond dimension gets too large) still leaves a valid file
    // with every snapshot collected so far.
    string out = "chain_dyn_siam_center_U" + string(argc>1?argv[1]:"0.2") + "_ref.txt";
    auto capture = [&](int step, string const& label) {
        snaps.push_back({label, fbr::getNi(sites, psi), fbr::getCc(sites, psi)});
        writeReference(out, snaps, L);
    };
    // t=0 snapshot: the prepared ground state, before any time evolution.
    capture(0, "initial");

    cout<<"time m n_dw n_dw_bf\n"<<setprecision(12);
    for(auto i=0;i<nSteps;i++){
        doTdvp(psi,mpo,dt);
        int step=i+1;
        int m=itensor::maxLinkDim(psi);
        double n_dw=itensor::expectC(psi,sites,"N",{nBath+nImp/2+1})[0].real();
        double n_dw_bf=itensor::expectC(psi,sites,"N",{nBath+nImp/2+2})[0].real();
        cout<<step*dt<<" "<<m<<" "<<n_dw<<" "<<n_dw_bf<<endl;
        for(auto const& w: wanted)
            if(w.first==step) capture(step, w.second);
        if(m>maxBondDim){
            cout<<"# bond dim "<<m<<" exceeded "<<maxBondDim
                <<" at t="<<step*dt<<"; stopping and keeping snapshots so far\n";
            break;
        }
    }

    cout<<"wrote "<<out<<" with "<<snaps.size()<<" snapshots\n";
    return 0;
}
