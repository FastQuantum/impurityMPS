#ifndef FB_MPS_H
#define FB_MPS_H

#include "givens_rotation.h"
#include "fermionic.h"

#include <armadillo>
#include <itensor/all.h>

enum Spin{up, dw};

/// This class stores a few body state with spin up/down on the left/right of the active orbitals.
/// |spin_up|--|impurity|---|spin_dw|
template<class T>
struct Fb_mps_spin
{
    itensor::Fermion sites;     ///< the sites of the network from ITensor
    itensor::MPS psi;           ///< the mps state
    arma::Mat<T> rot;           ///< the actual rotation frame
    arma::SpMat<T> cc;          ///< the correlation matrix or one-particle density matrix
    int imp_size;               ///< the impurity size. These orbitals go to the center and they will not be rotated
    int p1, p2;                 ///< the active orbitals are in [p1,p2)
    int natOrbDepth=-1;         ///< the depth of the circuit used to extract the natural orbitals (-1 means the to use an exact circuit)
    double tol=1e-10;           ///< the tolerance used for both applying the gates and defining active orbitals.

    /**
     * @brief Construct a Fb_mps as a Slater state.
     * @param rot is the rotation to get the ek
     * @param ek is the energy of every site, should be like: |spin_up|--|impurity|---|spin_dw|
     * @param nPart is the number of particles,
     * @param imp_size is the size of the central sites that will not be rotated later
     */
    static Fb_mps_spin<T> from_slater(arma::Mat<T> const& rot,arma::vec const& ek, int nPart, int imp_size)
    {
        Fb_mps_spin<T> fb;
        fb.sites=itensor::Fermion(ek.size(), {"ConserveNf",true});
        fb.cc=arma::Mat<T>(ek.size(), ek.size(), arma::fill::zeros);
        auto state = itensor::InitState(fb.sites,"0");
        arma::uvec iek=arma::stable_sort_index(ek);
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            fb.cc(k,k)=1;
        }
        fb.psi=itensor::MPS(state);
        fb.rot=rot;
        fb.imp_size=imp_size;
        std::tie(fb.p1,fb.p2)=fb.interval_impurity_full();
        return fb;
    }

    int length() const { return sites.length(); }

    /// return interval [a,b) of the slater part
    std::pair<int,int> interval_slater(Spin s) const { if (s==up) return {0,p1}; else return {p2,length()}; }

    /// return interval [a,b) of the active part
    std::pair<int,int> interval_active_full() const { return {p1,p2}; }

    /// return interval [a,b) of the active part
    std::pair<int,int> interval_active(Spin s) const { if (s==up) return {p1,length()/2}; else return {length()/2,p2}; }

    /// return interval [a,b) of the impurity part
    std::pair<int,int> interval_impurity_full() const { return {(length()-imp_size)/2, (length()+imp_size)/2}; }

    /// return interval [a,b) of the impurity part
    std::pair<int,int> interval_impurity(Spin s) const
    {
        int L=length(), d=imp_size;
        if (s==up) return {(L-d)/2, L/2};
        else       return {L/2, (L+d)/2};
    }

    /// return interval [a,b) that can be rotated
    std::pair<int,int> interval_rotating(Spin s) const
    {
        auto [a0,b0]=interval_impurity(s);
        auto [a1,b1]=interval_active(s);
        if (s==up) return {a1,a0};
        else       return {b0,b1};
    }

    /// convert to complex values. There is an specialization for `double` below.
    Fb_mps_spin<cmpx> to_complex() const { return *this; }

    // void extract_representative(arma::Mat<T>& K, int nRef) { extract_representative(K,nRef,p2-p1); }

    /// extract representative orbitals of the sites with ni=nRef where nRef can be 0 or 1,
    /// @param K is the matrix to extract the subspace
    /// @param use_active whether to use the active sites instead of the impurity sites
    void extract_representative(arma::Mat<T>& K, int nRef, bool use_active)
    {
        for(auto spin : {up,dw}) {

            // 1. find the orbitals with the occupation nref
            arma::uvec pos0; {
                auto [a,b]=interval_slater(spin);
                if (a>=b) continue; // no Slater
                arma::vec ni_bath=occupations_ni().rows(a,b-1);
                arma::vec delta_n_bath=arma::abs(ni_bath-nRef);
                pos0=arma::find(delta_n_bath<0.5).eval()+a ;
                if (pos0.empty()) continue;
            }

            // 2. find the Givens rotations
            int nSv; // number of singular values
            arma::Mat<T> rot1; // rotation of the slater
            {
                auto [a,b]=use_active ? interval_active(spin):
                                        interval_impurity(spin);
                auto k12 = K.rows(a,b-1).eval().cols(pos0).eval();
                arma::vec s;
                arma::Mat<T> U, V;
                svd_econ(U,s,V, k12);
                nSv=arma::find(s>tol*s[0]).eval().size();
                arma::Mat<T> V_slater=V.head_cols(nSv);
                auto givens= spin==dw ? GivensRotForRot_left(V_slater):
                                        GivensRotForRot_right(V_slater);
                GivensDaggerInPlace(givens);
                rot1=matrot_from_Givens(givens, k12.n_cols)/*.st()*/;
            }

            // 3. update K, rot and cc
            K.cols(pos0)=K.cols(pos0).eval()*rot1;
            K.rows(pos0)=rot1.t()*K.rows(pos0).eval();
            rot.cols(pos0)=rot.cols(pos0)*rot1;
            // no need to update cc

            // 4. move the nSv representative orbitals to the closest-to-impurity site of the Slater
            for(auto i=0; i<nSv; i++) {
                auto [a,b]=interval_active_full();
                int i1,i2;
                if (spin==dw) { i1=b; i2=pos0[i]; }
                else          { i1=pos0[pos0.size()-1-i]; i2=a-1; };
                SlaterWaveFunctionSwap (i1,i2);
                K.swap_cols(i1,i2);
                K.swap_rows(i1,i2);
                rot.swap_cols(i1,i2);
                cc.swap_cols(i1,i2);
                cc.swap_rows(i1,i2);  // TODO <---------- bug in armadillo
                if (spin==dw) p2++; else p1--;
            }

        } // for spin
    }

    /*
    bool extract_representative(int sv_index,arma::Mat<T>& K, int nRef, int nRows)
    {
        // 1. find the orbitals with the occupation nref
        auto ni_bath=arma::vec( arma::real( cc.diag().eval().rows(nActive, cc.n_rows-1) ) );
        arma::vec delta_n_bath=arma::abs(ni_bath-nRef);
        arma::uvec pos0=arma::find(delta_n_bath<0.5).eval()+nActive ;
        if (pos0.empty()) { std::cout<<"warning: no Slater?\n"; return false; }

        // 2. find the Givens rotations for them
        auto k12 = K.head_rows(nRows).eval().cols(pos0).eval();
        arma::vec s;
        arma::Mat<T> U, V;
        svd_spin(U,s,V, k12);
        int nSv=arma::find(s>tol*s[0]).eval().size();
        std::vector<GivensRot<T>> givens;
        if (true || nSv==1) {
            if (sv_index>=nSv) return false;
            givens=GivensRotForRot_left(V.cols(sv_index,sv_index).eval());
            GivensDaggerInPlace(givens);
        }
        else {
            std::cout<<"\nnSv>1\n";
            std::terminate();
            if (sv_index+1>=nSv) return false;
            givens=GivensRotForRot_left(V.cols(sv_index,sv_index+1).eval());
            GivensDaggerInPlace(givens);
        }

        // arma::Mat<T> rot1=matrot_from_Givens(givens, k12.n_cols);
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
        for(auto i=0; i<1; i++) {
            SlaterWaveFunctionSwap (nActive,pos0.at(i));
            K.swap_cols(nActive,pos0.at(i));
            K.swap_rows(nActive,pos0.at(i));
            rot.swap_cols(nActive,pos0.at(i));
            cc.swap_cols(nActive,pos0.at(i));
            cc.swap_rows(nActive,pos0.at(i));
            nActive++;
        }
        return true;
    }
    */

    /// TODO: update using extract_representative()
    void extract_representative_final(arma::Mat<T>& K) //TODO
    {
        for(auto spin:{up,dw}) {
            // 1. find the interval for the transformation
            auto [a,b]=interval_rotating(spin);
            if (a==b) continue;

            // 2. find the Givens rotations
            std::vector<GivensRot<T>> givens;
            {
                auto k12=K.submat(0,a,a-1,b-1);
                arma::vec s;
                arma::Mat<T> U,V;
                svd_econ(U,s,V,k12);
                int nSv=arma::find(s>tol*s[0]).eval().size();  // it should be nSv==nChannel
                arma::Mat<T> V_eff=V.head_cols(nSv);
                if (spin==dw) givens=GivensRotForRot_left(V_eff);
                else          givens=GivensRotForRot_right(V_eff);
                GivensDaggerInPlace(givens);
            }

            // 3. update K, rot and cc
            {
                arma::Mat<T> rot1=matrot_from_Givens(givens, b-a)/*.st()*/;
                K.cols(a,b-1)=K.cols(a,b-1).eval()*rot1;
                K.rows(a,b-1)=rot1.t()*K.rows(a,b-1).eval();
                rot.cols(a,b-1)=rot.cols(a,b-1)*rot1;
                cc.cols(a,b-1)=cc.cols(a,b-1).eval()*rot1.st().t();
                cc.rows(a,b-1)=rot1.st()*cc.rows(a,b-1).eval();
                // do not update nActive
            }

            // 4. update the mps
            {
                for(auto& g:givens) g.b+=a;
                auto gates=Fermionic::NOGates(sites, GivensTranspose(givens));
                gateTEvol(gates,1,1,psi,{"Cutoff",tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
            }
        }
    }

    /// update the cc in the active sector using the psi
    void update_cc()
    {
        for (auto spin : {up,dw}) {
            auto [a,b]=interval_active(spin);
            if constexpr (std::is_same<T,double>::value) {
                auto ccz=correlationMatrix(psi, sites,"Cdag","C",itensor::range1(a+1,b));
                for(auto i=0u; i<ccz.size(); i++)
                    for(auto j=0u; j<ccz[i].size(); j++)
                        cc(a+i,a+j)=ccz.at(i).at(j);
            }
            else {
                auto ccz=correlationMatrixC(psi, sites,"Cdag","C",itensor::range1(a+1,b));
                for(auto i=0u; i<ccz.size(); i++)
                    for(auto j=0u; j<ccz[i].size(); j++)
                        cc(a+i,a+j)=ccz.at(i).at(j);
            }
        }
    }

    /// Diagonalize the `cc` submatrix for the active orbitals that can be rotated.
    /// Rotate the variables accordingly
    /// @return the rotation Q applied: ci=Qij*dj (where ci are the old orbitals)
    arma::Mat<T> rotateToNaturalOrbitals()
    {
        auto [a_full,b_full]=interval_active_full();
        arma::Mat<T> rot_update(length(), length(), arma::fill::eye);
        for(auto spin:{up,dw}) {
            auto [a,b]=interval_rotating(spin);
            if (a==b) continue;

            // 1. find the Givens that diagonalize cc
            std::vector<GivensRot<T>> givens;//=GivensRotForCC_right(cc1,natOrbDepth);
            {
                auto cc1 = arma::Mat<T>( cc.submat(a,a,b-1, b-1).eval() );
                arma::vec eval;
                arma::Mat<T> evec;
                eig_sym(eval,evec,cc1);
                arma::vec activity=eval;
                for(auto &x : activity) x=std::min(x,1-x);
                arma::uvec iek=arma::stable_sort_index(activity);
                arma::Mat<T> rotation=evec.cols(iek);
                if (spin==dw) givens=GivensRotForRot_right(rotation);
                else          givens=GivensRotForRot_left(rotation);
            }
            if (givens.empty()) continue;

            // 2. apply the corresponding quantum gates
            for(auto& g:givens) g.b+=a; // absolute positions
            auto gates=Fermionic::NOGates(sites,givens);
            for(auto& g:givens) g.b-=a;
            itensor::gateTEvol(gates,1,1,psi,{"Cutoff",tol/*,"MaxDim",512*/,"Quiet",true, "Normalize",false,"ShowPercent",false});

            // 3. rotate rot, cc
            auto rot1=matrot_from_Givens(givens,b-a);
            rot.cols(a,b-1)=rot.cols(a,b-1).eval()*rot1.st();
            cc .cols(a,b-1)=cc .cols(a,b-1).eval()*rot1.t();
            cc .rows(a,b-1)=rot1*cc.rows(a,b-1).eval();
            rot_update.submat(a,a,b-1,b-1) = rot1.st();

            // 4. find the new active orbitals
            arma::vec ni_bath = occupations_ni().rows(a,b-1);
            int nActive_old=b-a;
            int nActive_new= arma::find(ni_bath>tol && ni_bath<1-tol).eval().size();
            int delta=nActive_new - nActive_old;
            if (spin==dw) p2+=delta;
            else          p1-=delta;
        } // for spin
        return rot_update.submat(a_full,a_full,b_full-1,b_full-1);
    }

    /// Energy of the Slater part. K is the kinetic energy matrix
    double SlaterEnergy(arma::Mat<T> const& K) const
    {
        double energy=0;
        for(auto spin:{up,dw}){
            auto [a,b]=interval_slater(spin);
            for(auto i=a; i<b; i++)
                energy += std::real(cc(i,i)*K(i,i));
        }
        return energy;
    }

    arma::vec occupations_ni() const { return arma::vec( arma::real(cc.diag()) );}

    arma::vec occupations_ni2() const
    {
        arma::vec ni(cc.n_rows);
        auto niv=itensor::expectC(psi,sites,"N");
        for(auto i=0u; i<ni.size(); i++)
            ni[i]=niv[i].real();
        return ni;
    }

    void print_bond_dims(std::string_view msg="") const
    {
        arma::cout<<msg<<arma::endl;
        arma::cout<<"active: "<<p2-p1<<arma::endl;
        for(auto i=0; i+1<psi.length(); i++)
            arma::cout<<itensor::leftLinkIndex(psi,i+1).dim()<<" ";
        arma::cout<<arma::endl;
    }

    /// compute all the correlator <ci^ cj> where i and j are original sites (i.e. before the rotation).
    arma::Mat<T> correlator_all() const
    {
        arma::Mat<T> Qinv=rot.st()/*.t()*/;
        return Qinv.t() * cc * Qinv;
    }

    /// compute the correlator <ci^ cj> where i and j are original sites (i.e. before the rotation).
    T correlator(int i, int j) const
    {
        arma::Mat<T> Qinv=rot.st()/*.t()*/;
        arma::Col<T> ccQinv=cc*Qinv.col(j);
        return arma::cdot(Qinv.col(i), ccQinv);
    }

    /// compute the correlator <ci^ cj> for all i, where i and j are original sites (i.e. before the rotation).
    arma::Col<T> correlator_all_i(int j) const
    {
        arma::Mat<T> Qinv=rot.st()/*.t()*/;
        arma::Col<T> ccQinv=cc*Qinv.col(j);
        return Qinv.t() * ccQinv;
    }

    /// compute the correlator <ci^ cj> for all j, where i and j are original sites (i.e. before the rotation).
    arma::Col<T> correlator_all_j(int i) const
    {
        arma::Mat<T> Qinv=rot.st()/*.t()*/;
        arma::Col<T> Qinv_t_cc=Qinv.col(i).t()*cc;
        return Qinv_t_cc*Qinv;
    }

private:

    /// Swap to sites inside the Slater part
    void SlaterWaveFunctionSwap(int i,int j)
    {
        if (i==j) return;
        auto [a,b]=interval_active_full();
        if (i>=a && i<b) throw std::invalid_argument("SlaterSwap for active orbital i");
        if (j>=a && j<b) throw std::invalid_argument("SlaterSwap for active orbital j");
        T ni=cc(i,i), nj=cc(j,j);
        if (std::abs(ni-nj)<0.5) throw std::invalid_argument("SlaterSwap for equal occupations");

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
inline Fb_mps_spin<cmpx> Fb_mps_spin<double>::to_complex() const
{
    Fb_mps_spin<cmpx> fb;
    fb.sites = sites;
    fb.psi = psi * cmpx(1,0);
    fb.rot = rot * cmpx(1,0);
    fb.cc = cc * cmpx(1,0);
    fb.imp_size = imp_size;
    fb.p1 = p1;
    fb.p2 = p2;
    fb.natOrbDepth = natOrbDepth;
    fb.tol = tol;
    return fb;
}


#endif // FB_MPS_H


