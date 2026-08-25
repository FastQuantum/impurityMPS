#ifndef FBR_FB_MPS_H
#define FBR_FB_MPS_H

#include "givens_rotation.h"
#include "itensor_utils.h"
#include "orbital_update.h"

#include <armadillo>
#include <itensor/all.h>

namespace fbr {

/// This class stores a few body state.
template<class T>
struct Fb_mps
{
    itensor::Fermion sites;     ///< the sites of the network from ITensor
    itensor::MPS psi;           ///< the mps state
    arma::Mat<T> rot;           ///< the actual rotation frame
    arma::Mat<T> cc;            ///< the correlation matrix or one-particle density matrix
    int nActive;                ///< the number of active orbitals (the rest nActive...sites.length() is considered Slater)
    int imp_size=0;             ///< the number of non-rotating impurity orbitals, located at [0,imp_size)
    bool spin=false;            ///< whether the sites can be splitted in even/odd for cc
    double tol=1e-10;           ///< the tolerance used for both applying the gates and defining active orbitals.
    int nSv=-1;                 ///< fixed rank of impurity–bath coupling (set by dynamics solver). -1 = recompute dynamically.


    /**
     * @brief Construct a Fb_mps as a Slater state.
     * @param rot is the rotation to get the ek
     * @param ek is the energy of every site,
     * @param nPart is the number of particles,
     * @param nActive is the (for now artifitially imposed) number of active orbitals.
     *        the part that will not rotated later
     */
    static Fb_mps<T> from_slater(arma::Mat<T> const& rot,arma::vec const& ek, int nPart, int nActive, bool spin)
    {
        Fb_mps<T> fb;
        fb.sites=itensor::Fermion(ek.size(), {"ConserveNf",true});
        fb.cc=arma::Mat<T>(ek.size(), ek.size(), arma::fill::zeros);
        auto state = itensor::InitState(fb.sites,"0");
        arma::uvec iek=my_sort_index(ek,spin);
        for(int j = 0; j < nPart; j++) {
            int k=iek[j];
            state.set(k+1,"1");
            fb.cc(k,k)=1;
        }
        fb.psi=itensor::MPS(state);
        fb.rot=rot;
        fb.nActive=nActive;
        fb.imp_size=nActive;   // initially the active window is exactly the non-rotating impurity
        fb.spin=spin;
        return fb;
    }

    OrbitalUpdate<T> planRepresentative(arma::Mat<T> const& K,int nRef) const
    {
        return planRepresentative(K,nRef,imp_size);
    }

    OrbitalUpdate<T> planRepresentative(arma::Mat<T> const& K,int nRef,int nRows) const
    {
        OrbitalUpdate<T> update(0,nActive);
        if (nActive>=(int)cc.n_rows) return update;

        arma::vec ni=arma::real(cc.diag()).eval().rows(nActive,cc.n_rows-1);
        arma::uvec positions=arma::find(arma::abs(ni-nRef)<0.5).eval()+nActive;
        if (positions.empty()) return update;

        auto k12=K.head_rows(nRows).eval().cols(positions).eval();
        arma::vec singular_values;
        arma::Mat<T> U,V;
        my_svd(U,singular_values,V,k12,spin);
        if (singular_values.empty()) return update;

        int nSv=(this->nSv>=0)
                  ? std::min<int>(this->nSv,(int)V.n_cols)
                  : (int)arma::find(singular_values>tol*singular_values[0]).eval().size();
        if (nSv<=0) return update;
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);
        update.append(positions,givens);
        for (int i=0; i<nSv; ++i)
            update.gates.emplace_back(nActive+i,(int)positions[i]);
        update.active={0,nActive+nSv};
        return update;
    }

    OrbitalUpdate<T> planActiveRepresentative(arma::Mat<T> const& K) const
    {
        return planActiveRepresentative(K,imp_size,nActive);
    }

    OrbitalUpdate<T> planActiveRepresentative(arma::Mat<T> const& K,
                                               int start,int end) const
    {
        OrbitalUpdate<T> update(0,nActive);
        if (start>=end) return update;

        arma::Mat<T> k12=K.submat(0,start,start-1,end-1);
        arma::vec singular_values;
        arma::Mat<T> U,V;
        my_svd(U,singular_values,V,k12,spin);
        if (singular_values.empty()) return update;

        int nSv=(this->nSv>=0)
                  ? std::min<int>(this->nSv,(int)V.n_cols)
                  : (int)arma::find(singular_values>tol*singular_values[0]).eval().size();
        if (nSv<=0) return update;
        auto givens=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(givens);
        update.append(arma::regspace<arma::uvec>(start,end-1),givens);
        return update;
    }

    OrbitalUpdate<T> planNaturalOrbitals(arma::Mat<T> const& cc_source) const
    {
        return planNaturalOrbitals(cc_source,imp_size);
    }

    OrbitalUpdate<T> planNaturalOrbitals(arma::Mat<T> const& cc_source,int start) const
    {
        OrbitalUpdate<T> update(0,nActive);
        if (start>=nActive) return update;

        auto cc_block=cc_source.submat(start,start,nActive-1,nActive-1).eval();
        arma::vec occupations;
        arma::Mat<T> orbitals;
        my_eig_sym(occupations,orbitals,cc_block,spin);
        arma::vec activity=occupations;
        for (auto& x : activity) x=std::min(x,1-x);
        arma::uvec order=my_sort_index(activity,spin);
        auto givens=GivensRotForRot_right(orbitals.cols(order).eval());
        for (auto& g : givens) g.b+=start;

        update.append(arma::regspace<arma::uvec>(0,nActive-1),
                      GivensTranspose(givens));

        arma::Mat<T> rotated_cc=cc_source;
        for (auto const& gate : update.gates)
            gate.applyAsCorrelator(rotated_cc);
        arma::vec ni=arma::real(rotated_cc.diag()).eval().rows(start,rotated_cc.n_rows-1);
        arma::uvec active=arma::find(ni>tol && ni<1-tol).eval();
        update.active={0,active.empty() ? start+1 : (int)active.back()+1+start};
        return update;
    }

    void applyUpdate(OrbitalUpdate<T> const& update)
    {
        std::vector<GivensRot<T>> circuit;
        auto flushCircuit=[&]() {
            auto gates=NOGates(sites,circuit);
            if (!gates.empty())
                itensor::gateTEvol(gates,1,1,psi,
                                   {"Cutoff",tol,"Quiet",true,"Normalize",false,"ShowPercent",false});
            circuit.clear();
        };

        for (auto const& gate : update.gates) {
            if (gate.swap) {
                flushCircuit();
                SlaterWaveFunctionSwap(gate.a,gate.b);
            }
            else {
                bool a_active=gate.a<nActive;
                bool b_active=gate.b<nActive;
                if (a_active!=b_active)
                    throw std::logic_error("orbital rotation crosses the active boundary");
                if (a_active) {
                    if (gate.b!=gate.a+1)
                        throw std::logic_error("active orbital rotation is not nearest-neighbor");
                    circuit.push_back(gate.givens(gate.a).transpose());
                }
            }
            gate.applyAsFrame(rot);
            gate.applyAsCorrelator(cc);
        }
        flushCircuit();
        nActive=update.active.second;
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

    /// Apply the single-site operator op to the (real-space) site i of the mps, where op
    /// is one of {"C", "Cdag", "N"} as in itensor. Only valid on the non-rotating impurity
    /// orbitals [0,imp_size): there site i maps exactly onto a single MPS site, so the
    /// operator is a genuine single-site operator.
    void applyLocalOp(std::string op, int i)
    {
        using namespace std;
        static const set<string> op_all={"C", "Cdag", "N"};
        if (op_all.count(op)==0)
            throw invalid_argument("Fb::applyLocalOp: op is not in my list. See itensor op for Fermion");

        arma::Mat<T> Qinv=rot.st();
        int i0=(int)arma::abs(Qinv.col(i)).index_max();
        if (i0 >= imp_size)
            throw invalid_argument("Fb::applyLocalOp: site i is not a non-rotating impurity site");

        // ITensor MPS sites/operators are 1-based.
        psi.position(i0+1);
        auto G=sites.op(op,i0+1);
        auto newA = G*psi(i0+1);
        newA.noPrime();
        psi.set(i0+1,newA);
        update_cc();
    }

    /// Energy of the Slater part. K is the kinetic energy matrix
    double SlaterEnergy(arma::Mat<T> const& K) const
    {
        double energy=0;
        for(auto i=nActive; i<cc.n_rows; i++)
            energy += std::real(cc(i,i)*K(i,i));
        return energy;
    }

    arma::vec occupations_ni2() const
    {
        arma::vec ni(cc.n_rows);
        auto niv=itensor::expectC(psi,sites,"N");
        for(auto i=0u; i<ni.size(); i++)
            ni[i]=niv[i].real();
        return ni;
    }

    /// compute all the correlator <ci^ cj> where i and j are original sites (i.e. before the rotation).
    /// Convention (impurity_param.h): c_i = sum_a rot[i,a] d_a, so
    ///   <c_i^dag c_j> = sum_{a,b} conj(rot[i,a]) cc[a,b] rot[j,b] = (Qinv^dag cc Qinv)[i,j]
    /// with Qinv = rot.st() (so that Qinv.col(i) holds rot.row(i) as a column).
    arma::Mat<T> correlator_all() const
    {
        arma::Mat<T> Qinv=rot.st();
        return Qinv.t() * cc * Qinv;
    }

    /// compute the correlator <ci^ cj> where i and j are original sites (i.e. before the rotation).
    T correlator(int i, int j) const
    {
        arma::Mat<T> Qinv=rot.st();
        arma::Col<T> ccQinv=cc*Qinv.col(j);
        return arma::cdot(Qinv.col(i), ccQinv);
    }

    /// compute the correlator <ci^ cj> for all i, where i and j are original sites (i.e. before the rotation).
    arma::Col<T> correlator_all_i(int j) const
    {
        arma::Mat<T> Qinv=rot.st();
        arma::Col<T> ccQinv=cc*Qinv.col(j);
        return Qinv.t() * ccQinv;
    }

    /// compute the correlator <ci^ cj> for all j, where i and j are original sites (i.e. before the rotation).
    arma::Col<T> correlator_all_j(int i) const
    {
        arma::Mat<T> Qinv=rot.st();
        arma::Row<T> Qinv_t_cc=Qinv.col(i).t()*cc;
        return (Qinv_t_cc*Qinv).st();
    }

private:

    /// Swap two sites inside the Slater part via a hopping MPO c†_i c_j - c†_j c_i.
    /// AutoMPO supplies the Jordan-Wigner string between non-adjacent orbitals.
    void SlaterWaveFunctionSwap(int i,int j)
    {
        if (i==j) return;
        if (i<nActive) throw std::invalid_argument("SlaterSwap for active orbital i");
        if (j<nActive) throw std::invalid_argument("SlaterSwap for active orbital j");
        T ni=cc(i,i), nj=cc(j,j);
        if (std::abs(ni-nj)<0.5)
            throw std::invalid_argument("SlaterSwap for equal occupations");

        itensor::AutoMPO ampo(sites);
        ampo+=1.0,"Cdag",i+1,"C",j+1;
        ampo+=-1.0,"Cdag",j+1,"C",i+1;
        auto H=itensor::toMPO(ampo);
        psi=itensor::applyMPO(H,psi,{"Cutoff",tol,"Normalize",false});
        psi.noPrime();
    }

};

} // namespace fbr

#endif // FBR_FB_MPS_H
