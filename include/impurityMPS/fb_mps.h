#ifndef FB_MPS_H
#define FB_MPS_H

#include "givens_rotation.h"
#include "fermionic.h"

#include <armadillo>
#include <itensor/all.h>

/// This class store a few body state.
template<class T>
struct Fb_mps
{
    itensor::Fermion sites;     ///< the sites of the network from ITensor
    itensor::MPS psi;           ///< the mps state
    arma::SpMat<T> cc;          ///< the correlation matrix or one-particle density matrix
    int nActive;                ///< the number of active orbitals (the rest nActive...sites.length() is considered Slater)
    double tol=1e-10;           ///< the tolerance used for both applying the gates and defining active orbitals.


    /// construct a Fb_mps as a Slater state.
    /// ek is the energy of every site, nPart is the number of particles.
    /// The number of active orbitals is initialized with nActive
    static Fb_mps<T> from_slater(arma::vec ek, int nPart, int nActive)
    {
        Fb_mps<T> fb;
        fb.sites=itensor::Fermion(ek.size(), {"ConserveNf",true});
        fb.cc=arma::Mat<T>(ek.size(), ek.size(), arma::fill::zeros);
        fb.nActive=nActive;
        auto state = itensor::InitState(fb.sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            fb.cc(k,k)=1;
        }
        fb.psi=itensor::MPS(state);
        return fb;
    }

    /// extract representative orbitals of the sites with ni=nRef where nRef can be 0 or 1.
    /// Return the Givens rotations used.
    std::vector<GivensRot<T>> extract_representative(arma::Mat<T>& K, int nRef)
    {
        // 1. find the orbitals with the occupation nref
        arma::vec ni_bath=arma::vec(cc.diag().eval().rows(nActive, cc.n_rows-1));
        arma::vec delta_n_bath=arma::abs(ni_bath-nRef);
        arma::uvec pos0=arma::find(delta_n_bath<0.5).eval()+nActive ;
        if (pos0.empty()) { std::cout<<"warning: no Slater?\n"; return {}; }

        // 2. find the Givens rotations for them
        auto k12 = K.head_rows(nActive).eval().cols(pos0).eval();
        arma::vec s;
        arma::Mat<T> U, V;
        svd_econ(U,s,V, k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);

        // 3. rotate K
        auto Kcol=K.cols(pos0).eval();
        applyGivens(Kcol,givens);
        K.cols(pos0)=Kcol;
        {
            arma::inplace_trans(K);
            auto Kcol=K.cols(pos0).eval();
            applyGivens(Kcol,givens);
            K.cols(pos0)=Kcol;
            arma::inplace_trans(K);
        }        
        // no need to update cc
        // 4. move the nSv representative orbitals to the beginning of the Slater
        for(auto i=0; i<nSv; i++) {
            SlaterSwap (nActive,pos0.at(i));
            K.swap_cols(nActive,pos0.at(i));
            K.swap_rows(nActive,pos0.at(i));
            nActive++;
        }
        return givens; // TODO: wrong, we need to add swap gates
    }

    /// update the cc in the active sector using the psi
    void update_cc()
    {
        auto ccz=correlationMatrix(psi, sites,"Cdag","C",itensor::range1(nActive));
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc(i,j)=ccz.at(i).at(j);
    }

    /// Diagonalize the cc submatrix from [start,nActive). Rotate psi, and update the nActive, accordingly.
    /// Return the rotation Q applied: ci=Qij*dj (where ci are the old orbitals)
    arma::Mat<T> rotateToNaturalOrbitals(int start)
    {
        auto cc1=arma::Mat<T>(cc.submat(start,start,nActive-1, nActive-1).eval());
        auto givens=GivensRotForCC_right(cc1);
        for(auto& g:givens) g.b+=start;
        auto gates=Fermionic::NOGates(sites,givens);
        gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
        auto rot1=matrot_from_Givens(givens,nActive);
        cc.cols(0,nActive-1)=cc.cols(0,nActive-1).eval()*rot1.t();
        cc.rows(0,nActive-1)=rot1*cc.rows(0,nActive-1).eval();
        auto ni_bath=arma::vec( arma::real(cc.diag()).eval().rows(start,cc.n_rows-1).eval() );
        nActive=arma::find(ni_bath>tol && ni_bath<1-tol).eval().size()+start;
        return rot1.st();
    }

    /// Energy of the Slater part. K is the kinetic energy matrix
    double SlaterEnergy(arma::Mat<T> const& K) const
    {
        double energy=0;
        for(auto i=nActive; i<cc.n_rows; i++)
            energy += cc(i,i)*K(i,i);
        return energy;
    }

    /// Swap to sites inside the Slater part
    void SlaterSwap(int i,int j)
    {
        if (i==j) return;
        if (i<nActive || j<nActive) throw std::runtime_error("SlaterSwap for active orbitals");
        if (std::abs(cc(i,i)-cc(j,j))<0.5) throw std::runtime_error("SlaterSwap for equal occupations");

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


#endif // FB_MPS_H


