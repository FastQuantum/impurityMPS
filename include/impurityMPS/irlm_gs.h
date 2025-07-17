#ifndef IRLM_GS_H
#define IRLM_GS_H

#include "givens_rotation.h"
#include "fermionic.h"

#include <armadillo>
#include <itensor/all.h>


std::pair<arma::vec,arma::mat> FullDiagonalizeTridiagonal(arma::vec an, arma::vec bn);

struct IrlmData {
    int L=20;
    double t=0.5;
    double V=0.15;
    double U=-0.5;
    double ed=0;
    bool connected=true;

    /// Kinetic energy matrix
    arma::sp_mat kin_mat() const
    {
        arma::sp_mat K(L,L);
        for(auto i=1+!connected; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=t;
        K(0,1)=K(1,0)=V;
        K(0,0)=ed;
        K(1,1)=-ed;

        K(1,1)=-U/2;
        K(0,0)+=-U/2;

        return K;
    }

    /// U ni nj
    arma::mat U_mat() const
    {
        arma::mat Umat(2,2,arma::fill::zeros);
        Umat(0,1)=U;
        return Umat;
    }

    /// Kinetic energy coeff in star geometry (Hbath is diagonal)
    arma::mat star_kin() const
    {
        auto K=kin_mat();
        arma::mat Kbath=arma::mat {K.submat(2,2,L-1,L-1).eval()};
        // arma::mat evec;
        // arma::vec ek;
        // arma::eig_sym(ek,evec,Kbath);
        auto [ek,evec]=FullDiagonalizeTridiagonal(Kbath.diag().eval(),Kbath.diag(1).eval());
        arma::vec vk=(K.submat(1,2,1,L-1)*evec).as_col();

        arma::mat Kstar(L,L,arma::fill::zeros);
        Kstar.submat(0,0,1,1)=K.submat(0,0,1,1);
        for(auto i=0u;i<ek.size();i++) {
            Kstar(i+2,i+2)=ek[i];
            Kstar(1,i+2)=Kstar(i+2,1)=vk[i];
        }
        return Kstar;
    }
};

struct DmrgParams {
    int max_bond_dim=512;
    int nIter_diag=4;
    double noise=1e-8;
};


struct Irlm_gs {
    IrlmData irlm;
    itensor::Fermion sites;
    itensor::AutoMPO hImp;
    double tol=1e-10;

    /// these quantities are updated during the iterations
    arma::mat K;
    itensor::MPS psi;
    double energy=-1000;
    arma::mat cc;
    int nActive=2;


    explicit Irlm_gs(const IrlmData& irlm_, double tol_=1e-10)
        : irlm(irlm_)
        , sites(itensor::Fermion(irlm.L, {"ConserveNf",true}))
        , hImp (sites)
        , tol(tol_)
        , K(irlm_.star_kin())
    {
        prepareSlaterGs(sites, arma::vec {K.diag()}, irlm.L/2);

        std::vector<int> impPos={0,1};
        auto Umat=irlm.U_mat();
        for(auto i:impPos)
            for(auto j:impPos)
                if (std::abs(Umat(i,j))>1e-15)
                    hImp += Umat(i,j),"N", i+1,"N", j+1;
    }

    void iterate(DmrgParams args={})
    {
        extract_f(0.0);
        extract_f(1.0);
        doDmrg(args);
        rotateToNaturalOrbitals();
    }

    /// extract f orbital of the sites with ni=0 or 1
    void extract_f(double nRef)
    {
        arma::vec ni=cc.diag();
        arma::vec nSlater=arma::abs(ni.rows(nActive,irlm.L-1)-nRef).eval();
        arma::uvec pos0=arma::find(nSlater<0.5).eval()+nActive ;
        if (pos0.empty()) { std::cout<<"warning: no slater?\n"; return; }
        auto k12 = K.head_rows(nActive).eval().cols(pos0).eval();
        arma::vec s;
        arma::mat U, V;
        svd(U,s,V, k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);
        auto Kcol=K.cols(pos0).eval();
        applyGivens(Kcol,givens);
        K.cols(pos0)=Kcol;
        itensor::cpu_time t0;
        std::cout<<" givens to K 1"<<t0.sincemark()<<std::endl; t0.mark();

        {
            arma::inplace_trans(K);
            auto Kcol=K.cols(pos0).eval();
            applyGivens(Kcol,givens);
            K.cols(pos0)=Kcol;
            arma::inplace_trans(K);
            // auto Krow=K.rows(pos0).eval();
            // applyGivens(GivensDagger(givens),Krow);
            // K.rows(pos0)=Krow;
        }
        std::cout<<" givens to K 2"<<t0.sincemark()<<std::endl; t0.mark();
        // no need to update cc
        for(auto i=0; i<nSv; i++) {
            SlaterSwap(nActive,pos0.at(i));
            nActive++;
        }
    }

    void doDmrg(DmrgParams args={})
    {
        auto h=hImp;
        for(auto i=0; i<nActive; i++)
            for(auto j=0; j<nActive; j++)
            if (std::abs(K(i,j))>tol)
                h += K(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);

        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(psi,mpo,sweeps, {"MaxSite",nActive,"Quiet", true, "Silent", true});
        energy += SlaterEnergy();
        auto ccz=correlationMatrix(psi, sites,"Cdag","C",itensor::range1(nActive));
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc(i,j)=ccz.at(i).at(j);
    }

    void rotateToNaturalOrbitals()
    {
        auto cc1=cc.submat(2,2,nActive-1, nActive-1).eval();
        auto givens=GivensRotForCC_right(cc1);
        for(auto& g:givens) g.b+=2;
        auto gates=Fermionic::NOGates(sites,givens);
        gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
        auto rot1=matrot_from_Givens(givens,nActive);
        cc.cols(0,nActive-1)=cc.cols(0,nActive-1).eval()*rot1.t();
        cc.rows(0,nActive-1)=rot1*cc.rows(0,nActive-1).eval();
        K.cols(0,nActive-1)=K.cols(0,nActive-1).eval()*rot1.st();
        K.rows(0,nActive-1)=rot1.st().t()*K.rows(0,nActive-1).eval();
        auto nib=arma::real(cc.diag()).eval().rows(2,irlm.L-1).eval();
        nActive=arma::find(nib>tol && nib<1-tol).eval().size()+2;
    }

    void prepareSlaterGs(itensor::Fermion const& sites, arma::vec ek, int nPart)
    {
        cc=arma::mat(irlm.L, irlm.L, arma::fill::zeros);
        ek[0]=-10; // force ocupation |10>
        ek[1]=10;
        auto state = itensor::InitState(sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        double energy=0;
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            energy += ek[k];
            cc(k,k)=1;
        }
        psi=itensor::MPS(state);
    }

    double SlaterEnergy()
    {
        double energy=0;
        for(auto i=nActive; i<irlm.L; i++)
            energy += cc(i,i)*K(i,i);
        return energy;
    }

    itensor::MPO fullHamiltonian(bool real_space=true) const
    {
        auto kin= real_space ? arma::mat {irlm.kin_mat()} : K;
        auto h=hImp;
        for(auto i=0; i<irlm.L; i++)
            for(auto j=0; j<irlm.L; j++)
            if (std::abs(kin(i,j))>tol)
                h += kin(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }

private:

    /// Swap to sites inside the Slater part
    void SlaterSwap(int i,int j)
    {
        if (i==j) return;
        if (i<nActive || j<nActive) throw std::runtime_error("SlaterSwap for active orbitals");
        if (std::abs(cc(i,i)-cc(j,j))<0.5) {
            std::cout<<" swap "<<i<<" "<<j<<"\n";
            std::cout.flush();
            throw std::runtime_error("SlaterSwap for equal occupations");
        }
        K.swap_cols(i,j);
        K.swap_rows(i,j);

        auto flip=[&](int p) {
            auto G = cc(p,p)>0.5 ? sites.op("A",p+1) : sites.op("Adag",p+1) ;
            auto newA = G*psi(p+1);
            newA.noPrime();
            psi.set(p+1,newA);
        };
        flip(i);
        flip(j);

        cc.swap_cols(i,j);
        cc.swap_rows(i,j);
    }
};

#include<mkl_lapacke.h>

inline std::pair<arma::vec,arma::mat> FullDiagonalizeTridiagonal(arma::vec an, arma::vec bn)
{
    lapack_int n=an.size(), M;
    arma::vec eval(n);
    arma::mat evec(n,n);
    std::vector<lapack_int> ifail(n);
    int info=LAPACKE_dstevr(LAPACK_COL_MAJOR,'V','A', n, an.memptr(), bn.memptr(),
                              0.0, 0.0,1,1,2e-11,&M,eval.memptr(),evec.memptr(),n,ifail.data());
    if (info!=0) throw
            std::runtime_error("LAPACKE_dstevx inside DiagonalizeTridiagonal, info!=0");
    return std::make_pair(eval,evec);
}

#endif // IRLM_GS_H
