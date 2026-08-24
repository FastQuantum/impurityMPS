#ifndef FBR_FB_MPS_SPIN_H
#define FBR_FB_MPS_SPIN_H

#include "givens_rotation.h"
#include "itensor_utils.h"
#include "orbital_update.h"

#include <armadillo>
#include <itensor/all.h>

namespace fbr {

enum Spin{up, dw};

/// This class stores a few body state with spin up/down on the left/right of the active orbitals.
/// |spin_up|--|impurity|---|spin_dw|
template<class T>
struct Fb_mps_spin
{
    itensor::Fermion sites;     ///< the sites of the network from ITensor
    itensor::MPS psi;           ///< the mps state
    arma::Mat<T> rot;           ///< the actual rotation frame
    arma::Mat<T> cc;            ///< the correlation matrix or one-particle density matrix
    int imp_size;               ///< the impurity size. These orbitals go to the center and they will not be rotated
    int p1, p2;                 ///< the active orbitals are in [p1,p2)
    int natOrbDepth=-1;         ///< the depth of the circuit used to extract the natural orbitals (-1 means the to use an exact circuit)
    double tol=1e-10;           ///< the tolerance used for both applying the gates and defining active orbitals.
    int nSv=-1;                 ///< fixed rank of impurity–bath coupling (set by dynamics solver). -1 = recompute dynamically.

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
        arma::uvec iek=arma::sort_index(ek);
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

    /// return interval [a,b) of the impurity part
    std::pair<int,int> interval_bath(Spin s) const
    {
        auto [a,b] = interval_impurity(s);
        if (s==up) return {0,a};
        else       return {b,length()};
    }

    /// return interval [a,b) that can be rotated
    std::pair<int,int> interval_rotating(Spin s) const
    {
        auto [a0,b0]=interval_impurity(s);
        auto [a1,b1]=interval_active(s);
        if (s==up) return {a1,a0};
        else       return {b0,b1};
    }

    /// ensure reflection symmetry of rows and columns
    static void ensure_reflection_mat(arma::Mat<T> &K)
    {
        int L=K.n_rows;
        auto irev=arma::regspace<arma::uvec>(L/2-1,0);
        K.submat(irev,irev)=K.submat(L/2,L/2,L-1,L-1);
    }

    /// ensure reflection symmetry of columns: col(i)=col(L-1-i)
    static void ensure_reflection_col(arma::Mat<T> &R)
    {
        int L=R.n_rows;
        auto irev=arma::regspace<arma::uvec>(L/2-1,0);
        R.cols(irev)=R.cols(L/2,L-1);
    }

    /// convert to complex values. There is an specialization for `double` below.
    Fb_mps_spin<cmpx> to_complex() const { return *this; }

    OrbitalUpdate<T> planRepresentative(arma::Mat<T> const& K,int nRef) const
    {
        return planRepresentativeFrom(K,nRef,interval_impurity(dw));
    }

    /// As above, but drawing the coupling subspace from the active window
    /// (used by the ground-state solver) instead of the impurity block.
    OrbitalUpdate<T> planRepresentative(arma::Mat<T> const& K,int nRef,bool use_active) const
    {
        return planRepresentativeFrom(K,nRef,
                   use_active ? interval_active(dw) : interval_impurity(dw));
    }

    OrbitalUpdate<T> planActiveRepresentative(arma::Mat<T> const& K) const
    {
        OrbitalUpdate<T> update(p1,p2);
        auto [a_imp,b_imp]=interval_impurity(dw);
        auto [a,b]=interval_rotating(dw);
        if (a>=b) return update;

        arma::Mat<T> k12=K.rows(a_imp,b_imp-1).eval().cols(a,b-1);
        arma::vec singular_values;
        arma::Mat<T> U,V;
        arma::svd_econ(U,singular_values,V,k12);
        if (singular_values.empty()) return update;

        int nSv=(this->nSv>=0)
                  ? std::min<int>(this->nSv,(int)V.n_cols)
                  : (int)arma::find(singular_values>tol*singular_values[0]).eval().size();
        if (nSv<=0) return update;

        auto gates_dw=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(gates_dw);
        auto gates_up=GivensReflect(gates_dw,b-a);
        auto [a_up,b_up]=interval_rotating(up);
        update.append(arma::regspace<arma::uvec>(a,b-1),gates_dw);
        update.append(arma::regspace<arma::uvec>(a_up,b_up-1),gates_up);
        return update;
    }

    OrbitalUpdate<T> planNaturalOrbitals(arma::Mat<T> const& cc_source) const
    {
        OrbitalUpdate<T> update(p1,p2);
        auto [a,b]=interval_rotating(dw);
        if (a>=b) return update;

        auto cc_block=cc_source.submat(a,a,b-1,b-1).eval();
        arma::vec occupations;
        arma::Mat<T> orbitals;
        arma::eig_sym(occupations,orbitals,cc_block);
        arma::vec activity=occupations;
        for (auto& x : activity) x=std::min(x,1-x);
        arma::uvec order=arma::stable_sort_index(activity);

        auto gates_dw=GivensRotForRot_right(orbitals.cols(order).eval());
        auto gates_up=GivensReflect(gates_dw,b-a);
        auto [a_up,b_up]=interval_rotating(up);
        update.append(arma::regspace<arma::uvec>(a,b-1),
                      GivensTranspose(gates_dw));
        update.append(arma::regspace<arma::uvec>(a_up,b_up-1),
                      GivensTranspose(gates_up));

        arma::Mat<T> rotated_cc=cc_source;
        for (auto const& gate : update.gates)
            gate.applyAsCorrelator(rotated_cc);
        ensure_reflection_mat(rotated_cc);

        arma::vec ni=arma::real(rotated_cc.diag()).eval().rows(a,b-1);
        arma::uvec active=arma::find(ni>tol && ni<1-tol).eval();
        int active_end=active.empty() ? a+2 : a+(int)active.back()+1;
        int delta=active_end-p2;
        update.active={p1-delta,active_end};
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
                bool a_active=gate.a>=p1 && gate.a<p2;
                bool b_active=gate.b>=p1 && gate.b<p2;
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
        ensure_reflection_mat(cc);
        p1=update.active.first;
        p2=update.active.second;
    }

    // Compatibility wrappers for the previous mutating API.
    void extract_representative(arma::Mat<T>& K,int nRef,bool use_active)
    {
        auto source=use_active ? interval_active(dw) : interval_impurity(dw);
        auto update=planRepresentativeFrom(K,nRef,source);
        update.applyAsBasis(K);
        ensure_reflection_mat(K);
        applyUpdate(update);
    }

    void extract_representative_final(arma::Mat<T>& K)
    {
        auto update=planActiveRepresentative(K);
        update.applyAsBasis(K);
        ensure_reflection_mat(K);
        applyUpdate(update);
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

    /// Diagonalize the `cc` submatrix for the active orbitals that can be rotated,
    /// then rotate psi, rot and cc accordingly.
    void rotateToNaturalOrbitals()
    {
        applyUpdate(planNaturalOrbitals(cc));
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

    /// Map a real-space site i to the MPS orbital index a (0-based) that carries it.
    /// Convention (see correlator_all): c_i = sum_a rot[i,a] d_a, so the orbital is
    /// argmax_a |rot[i,a]|, i.e. the largest entry of *row* i of rot. For an impurity
    /// site (whose orbital is never rotated) row i is a unit vector and the mapping is exact.
    int frame_site(int i) const
    {
        arma::vec w = arma::abs(rot.row(i)).t();   // |rot[i,a]| over a (real, length L)
        return (int) w.index_max();
    }

    /// Apply the single-site operator op to the (real-space) site i of the mps, where op
    /// is one of {"C", "Cdag", "N"} as in itensor. Only valid on the non-rotating impurity
    /// orbitals (interval_impurity_full): there site i maps exactly onto a single MPS site,
    /// so the operator is a genuine single-site operator.
    void applyLocalOp(std::string op, int i)
    {
        using namespace std;
        static const set<string> op_all={"C", "Cdag", "N"};
        if (op_all.count(op)==0)
            throw invalid_argument("Fb_mps_spin::applyLocalOp: op is not in my list. See itensor op for Fermion");

        int i0 = frame_site(i);   // 0-based MPS orbital
        auto [a_imp,b_imp] = interval_impurity_full();
        if (i0 < a_imp || i0 >= b_imp)
            throw invalid_argument("Fb_mps_spin::applyLocalOp: site i is not a non-rotating impurity site");

        // ITensor MPS sites/operators are 1-based.
        psi.position(i0+1);
        auto G=sites.op(op,i0+1);
        auto newA=G*psi(i0+1);
        newA.noPrime();
        psi.set(i0+1,newA);
        update_cc();
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

    OrbitalUpdate<T> planRepresentativeFrom(arma::Mat<T> const& K,int nRef,
                                             std::pair<int,int> source) const
    {
        OrbitalUpdate<T> update(p1,p2);
        auto [a_slater,b_slater]=interval_slater(dw);
        if (a_slater>=b_slater) return update;

        arma::vec ni=occupations_ni().rows(a_slater,b_slater-1);
        arma::uvec positions=arma::find(arma::abs(ni-nRef)<0.5).eval()+a_slater;
        if (positions.empty()) return update;

        auto [a_source,b_source]=source;
        auto k12=K.rows(a_source,b_source-1).eval().cols(positions).eval();
        arma::vec singular_values;
        arma::Mat<T> U,V;
        arma::svd_econ(U,singular_values,V,k12);
        if (singular_values.empty()) return update;

        int nSv=(this->nSv>=0)
                  ? std::min<int>(this->nSv,(int)V.n_cols)
                  : (int)arma::find(singular_values>tol*singular_values[0]).eval().size();
        if (nSv<=0) return update;

        auto gates_dw=GivensRotForRot_left(V.head_cols(nSv).eval());
        GivensDaggerInPlace(gates_dw);
        auto gates_up=GivensReflect(gates_dw,(int)positions.size());
        arma::uvec positions_up=length()-1-arma::reverse(positions);
        update.append(positions,gates_dw);
        update.append(positions_up,gates_up);

        int active_begin=p1;
        int active_end=p2;
        for (int i=0; i<nSv; ++i) {
            update.gates.emplace_back(length()-1-(int)positions[i],active_begin-1);
            active_begin--;
        }
        for (int i=0; i<nSv; ++i) {
            update.gates.emplace_back(active_end,(int)positions[i]);
            active_end++;
        }
        update.active={active_begin,active_end};
        return update;
    }

    /// Swap two sites inside the Slater part via a hopping MPO c†_i c_j + h.c.
    /// AutoMPO handles the Jordan-Wigner string; the global fermionic sign is irrelevant.
    void SlaterWaveFunctionSwap(int i,int j)
    {
        if (i==j) return;
        auto [a,b]=interval_active_full();
        if (i>=a && i<b) throw std::invalid_argument("SlaterSwap for active orbital i");
        if (j>=a && j<b) throw std::invalid_argument("SlaterSwap for active orbital j");
        T ni=cc(i,i), nj=cc(j,j);
        if (std::abs(ni-nj)<0.5) throw std::invalid_argument("SlaterSwap for equal occupations");

        itensor::AutoMPO ampo(sites);
        ampo += 1.0,"Cdag",i+1,"C",j+1;
        ampo += -1.0,"Cdag",j+1,"C",i+1;
        auto H = itensor::toMPO(ampo);
        psi = itensor::applyMPO(H,psi,{"Cutoff",tol,"Normalize",false});
        psi.noPrime();
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
    fb.nSv = nSv;
    return fb;
}


} // namespace fbr

#endif // FBR_FB_MPS_SPIN_H
