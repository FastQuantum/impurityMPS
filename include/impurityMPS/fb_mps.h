#ifndef FB_MPS_H
#define FB_MPS_H

#include "givens_rotation.h"
#include "fermionic.h"

#include <armadillo>
#include <itensor/all.h>

/// This class stores a few body state.
template<class T>
struct Fb_mps
{
    itensor::Fermion sites;     ///< the sites of the network from ITensor
    itensor::MPS psi;           ///< the mps state
    arma::Mat<T> rot;           ///< the actual rotation frame
    arma::SpMat<T> cc;          ///< the correlation matrix or one-particle density matrix
    int nActive;                ///< the number of active orbitals (the rest nActive...sites.length() is considered Slater)
    double tol=1e-10;           ///< the tolerance used for both applying the gates and defining active orbitals.


    /**
     * @brief Construct a Fb_mps as a Slater state.
     * @param rot is the rotation to get the ek
     * @param ek is the energy of every site,
     * @param nPart is the number of particles,
     * @param nActive is the (for now artifitially imposed) number of active orbitals.
     *        the part that will not rotated later
     */
    static Fb_mps<T> from_slater(arma::Mat<T> const& rot,arma::vec const& ek, int nPart, int nActive)
    {
        Fb_mps<T> fb;
        fb.sites=itensor::Fermion(ek.size(), {"ConserveNf",true});
        fb.cc=arma::Mat<T>(ek.size(), ek.size(), arma::fill::zeros);
        auto state = itensor::InitState(fb.sites,"0");
        arma::uvec iek=arma::sort_index(ek);
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            fb.cc(k,k)=1;
        }
        fb.psi=itensor::MPS(state);
        fb.rot=rot;
        fb.nActive=nActive;
        return fb;
    }

    /// convert to complex values. There is an specialization for `double` below.
    Fb_mps<cmpx> to_complex() const { return *this; }

    /// extract representative orbitals of the sites with ni=nRef where nRef can be 0 or 1.
    /// Return the Givens rotations used.
    void extract_representative(arma::Mat<T>& K, int nRef) { extract_representative(K,nRef,nActive); }

    /// extract representative orbitals of the sites with ni=nRef where nRef can be 0 or 1,
    /// @param start is the first site to consider
    /// Return the Givens rotations used.
    void extract_representative(arma::Mat<T>& K, int nRef, int nRows)
    {
        // 1. find the orbitals with the occupation nref
        auto ni_bath=arma::vec( arma::real( cc.diag().eval().rows(nActive, cc.n_rows-1) ) );
        arma::vec delta_n_bath=arma::abs(ni_bath-nRef);
        arma::uvec pos0=arma::find(delta_n_bath<0.5).eval()+nActive ;
        if (pos0.empty()) { std::cout<<"warning: no Slater?\n"; /*return {};*/ }

        // 2. find the Givens rotations for them
        auto k12 = K.head_rows(nRows).eval().cols(pos0).eval();
        arma::vec s;
        arma::Mat<T> U, V;
        svd_econ(U,s,V, k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);

        // arma::Mat<T> rot1=matrot_from_Givens(givens, k12.n_cols)/*.st()*/;
        // K.cols(pos0)=K.cols(pos0).eval()*rot1;
        // K.rows(pos0)=rot1.t()*K.rows(pos0).eval();
        // rot.cols(pos0)=rot.cols(pos0)*rot1;

        // 3. rotate K and cc
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
        auto Rcol=rot.cols(pos0).eval();
        applyGivens(Rcol,givens);
        rot.cols(pos0)=Rcol;

        // no need to update cc
        // 4. move the nSv representative orbitals to the beginning of the Slater
        for(auto i=0; i<nSv; i++) {
            SlaterWaveFunctionSwap (nActive,pos0.at(i));
            K.swap_cols(nActive,pos0.at(i));
            K.swap_rows(nActive,pos0.at(i));
            rot.swap_cols(nActive,pos0.at(i));            
            cc.swap_cols(nActive,pos0.at(i));
            cc.swap_rows(nActive,pos0.at(i));
            nActive++;
        }
        //return givens; // TODO: wrong, we need to add swap gates
    }

    void extract_representative_final(arma::Mat<T>& K, int start, int end )
    {
        // 1. find the interval for the transformation
        int p1=start;  // first position
        int p2=end-1;  // last position

        // 2. find the Givens rotations
        auto k12=K.submat(0,p1,p1-1,p2);
        arma::vec s;
        arma::Mat<cmpx> U, V;
        svd_econ(U,s,V,k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();  // it should be nSv==nChannel
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);

        // 3. update K, rot and cc
        arma::cx_mat rot1=matrot_from_Givens(givens, k12.n_cols)/*.st()*/;
        K.cols(p1,p2)=K.cols(p1,p2).eval()*rot1;
        K.rows(p1,p2)=rot1.t()*K.rows(p1,p2).eval();
        rot.cols(p1,p2)=rot.cols(p1,p2)*rot1;
        cc.cols(p1,p2)=cc.cols(p1,p2).eval()*rot1.st().t();
        cc.rows(p1,p2)=rot1.st()*cc.rows(p1,p2).eval();
        // do not update nActive

        // 4. update the mps
        for(auto& g:givens) g.b+=p1;
        auto gates=Fermionic::NOGates(sites, GivensDagger(givens));
        gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
    }

    /// update the cc in the active sector using the psi
    void update_cc()
    {
        if constexpr (std::is_same<T,double>::value) {
            auto ccz=correlationMatrix(psi, sites,"Cdag","C",itensor::range1(nActive));
            for(auto i=0u; i<ccz.size(); i++)
                for(auto j=0u; j<ccz[i].size(); j++)
                    cc(i,j)=ccz.at(i).at(j);
        }
        else {
            auto ccz=correlationMatrixC(psi, sites,"Cdag","C",itensor::range1(nActive));
            for(auto i=0u; i<ccz.size(); i++)
                for(auto j=0u; j<ccz[i].size(); j++)
                    cc(i,j)=ccz.at(i).at(j);
        }        
    }

    /// Diagonalize the `cc` submatrix in the interval [start,nActive).
    /// Rotate `psi`, and update the `nActive`, accordingly.
    /// @return the rotation Q applied: ci=Qij*dj (where ci are the old orbitals)
    arma::Mat<T> rotateToNaturalOrbitals(int start)
    {
        auto cc1 = arma::Mat<T>( cc.submat(start,start,nActive-1, nActive-1).eval() );
        auto givens=GivensRotForCC_right(cc1);
        for(auto& g:givens) g.b+=start;
        auto gates=Fermionic::NOGates(sites,givens);
        gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
        auto rot1=matrot_from_Givens(givens,nActive);
        rot.cols(0,nActive-1)=rot.cols(0,nActive-1).eval()*rot1.st();
        cc.cols(0,nActive-1)=cc.cols(0,nActive-1).eval()*rot1.t();
        cc.rows(0,nActive-1)=rot1*cc.rows(0,nActive-1).eval();
        auto ni_bath = arma::vec( arma::real(cc.diag()).eval().rows(start,cc.n_rows-1).eval() );
        nActive=arma::find(ni_bath>tol && ni_bath<1-tol).eval().size()+start;
        return rot1.st();
    }

    /// Energy of the Slater part. K is the kinetic energy matrix
    double SlaterEnergy(arma::Mat<T> const& K) const
    {
        double energy=0;
        for(auto i=nActive; i<cc.n_rows; i++)
            energy += std::real(cc(i,i)*K(i,i));
        return energy;
    }

    // arma::vec occupations_ni() const { return arma::vec( arma::real(cc.diag()) );}

    arma::vec occupations_ni2() const
    {
        arma::vec ni(cc.n_rows);
        auto niv=itensor::expectC(psi,sites,"N");
        for(auto i=0u; i<ni.size(); i++)
            ni[i]=niv[i].real();
        return ni;
    }

    /// compute all the correlator <ci^ cj> where i and j are original sites (i.e. before the rotation).
    arma::Mat<T> correlator_all() const
    {
        arma::Mat<T> Qinv=rot.st().t();
        return Qinv.t() * cc * Qinv;
    }

    /// compute the correlator <ci^ cj> where i and j are original sites (i.e. before the rotation).
    T correlator(int i, int j) const
    {
        arma::Mat<T> Qinv=rot.st().t();
        arma::Col<T> ccQinv=cc*Qinv.col(j);
        return arma::cdot(Qinv.col(i), ccQinv);
    }

    /// compute the correlator <ci^ cj> for all i, where i and j are original sites (i.e. before the rotation).
    arma::Col<T> correlator_all_i(int j) const
    {
        arma::Mat<T> Qinv=rot.st().t();
        arma::Col<T> ccQinv=cc*Qinv.col(j);
        return Qinv.t() * ccQinv;
    }

    /// compute the correlator <ci^ cj> for all j, where i and j are original sites (i.e. before the rotation).
    arma::Col<T> correlator_all_j(int i) const
    {
        arma::Mat<T> Qinv=rot.st().t();
        arma::Col<T> Qinv_t_cc=Qinv.col(i).t()*cc;
        return Qinv_t_cc*Qinv;
    }

private:

    /// Swap to sites inside the Slater part
    void SlaterWaveFunctionSwap(int i,int j)
    {
        if (i==j) return;
        if (i<nActive || j<nActive) throw std::runtime_error("SlaterSwap for active orbitals");
        T ni=cc(i,i), nj=cc(j,j);
        if (std::abs(ni-nj)<0.5) throw std::runtime_error("SlaterSwap for equal occupations");

        auto flip=[&](int p) {
            T np=cc(p,p);
            auto G = std::abs(np)>0.5 ? sites.op("A",p+1) : sites.op("Adag",p+1) ;
            auto newA = G*psi(p+1);
            newA.noPrime();
            psi.set(p+1,newA);
        };
        flip(i);
        flip(j);
    }

};

template<>
Fb_mps<cmpx> Fb_mps<double>::to_complex() const
{
    Fb_mps<cmpx> fb;
    fb.sites = sites;
    fb.psi = psi * cmpx(1,0);
    fb.rot = rot * cmpx(1,0);
    fb.cc = cc * cmpx(1,0);
    fb.nActive = nActive;
    fb.tol = tol;
    return fb;
}


#endif // FB_MPS_H


