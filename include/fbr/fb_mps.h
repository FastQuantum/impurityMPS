#ifndef FBR_FB_MPS_H
#define FBR_FB_MPS_H

#include "givens_rotation.h"
#include "itensor_utils.h"
#include "orbital_update.h"

#include <armadillo>
#include <itensor/all.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <vector>

namespace fbr {

enum Spin{up, dw};

/// Arrangement of the active orbital window.
enum Layout {
    leading,        ///< |imp|active|slater|, a single sector (spinless)
    spin_symmetric, ///< centered window, spin up is the reflection of spin down
    spin_block      ///< centered window, the two spin blocks are independent
};

/// This class stores a few body state: a small active window of orbitals kept
/// in an MPS (with entanglement), while the rest is a Slater determinant.
///
/// The chain always reads
///     |slater_up|active_up|imp_up|imp_dw|active_dw|slater_dw|
/// with the non-rotating impurity orbitals around the center `mid()`. The
/// `leading` layout is the degenerate case mid()==0: the up sector is empty and
/// the chain is simply |imp|active|slater|. That is why every interval below is
/// meaningful for the three layouts, the up ones being empty for `leading`.
template<class T>
struct Fb_mps
{
    itensor::Fermion sites;     ///< the sites of the network from ITensor
    itensor::MPS psi;           ///< the mps state
    arma::Mat<T> rot;           ///< the actual rotation frame
    arma::Mat<T> cc;            ///< the correlation matrix or one-particle density matrix
    int imp_size=0;             ///< the number of non-rotating impurity orbitals, at the center
    int p1=0, p2=0;             ///< the active orbitals are in [p1,p2)
    Layout layout=leading;      ///< the arrangement of the active window
    bool spin=false;            ///< (`leading` only) the sites can be splitted in even/odd for cc
    double tol=1e-10;           ///< the tolerance used for both applying the gates and defining active orbitals.
    int nSv=-1;                 ///< fixed rank of impurity–bath coupling (set by dynamics solver). -1 = recompute dynamically.

    /**
     * @brief Construct a Fb_mps as a Slater state.
     * @param rot is the rotation to get the ek
     * @param ek is the energy of every site, ordered as the chosen layout
     * @param nPart is the number of particles,
     * @param imp_size is the number of central sites that will not be rotated later
     * @param layout is the arrangement of the active window
     * @param spin (`leading` only) whether even/odd sites are independent spin sectors
     */
    static Fb_mps<T> from_slater(arma::Mat<T> const& rot, arma::vec const& ek,
                                 int nPart, int imp_size, Layout layout, bool spin=false)
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
        fb.imp_size=imp_size;
        fb.layout=layout;
        fb.spin=spin;
        // initially the active window is exactly the non-rotating impurity
        std::tie(fb.p1,fb.p2)=fb.interval_impurity_full();
        return fb;
    }

    int length() const { return sites.length(); }

    /// the number of active orbitals
    int nActive() const { return p2-p1; }

    /// the center of the chain: the impurity sits there, the two spin sectors
    /// grow away from it. The `leading` layout has no up sector, so its center
    /// is the left edge.
    int mid() const { return layout==leading ? 0 : length()/2; }

    /// return interval [a,b) of the impurity part
    std::pair<int,int> interval_impurity_full() const
    {
        int m=mid();
        return layout==leading ? std::pair<int,int>{m, m+imp_size}
                               : std::pair<int,int>{m-imp_size/2, m+imp_size/2};
    }

    /// return interval [a,b) of the impurity part
    std::pair<int,int> interval_impurity(Spin s) const
    {
        auto [a,b]=interval_impurity_full();
        int m=mid();
        if (s==up) return {a,m};
        else       return {m,b};
    }

    /// return interval [a,b) of the slater part
    std::pair<int,int> interval_slater(Spin s) const { if (s==up) return {0,p1}; else return {p2,length()}; }

    /// return interval [a,b) of the active part
    std::pair<int,int> interval_active_full() const { return {p1,p2}; }

    /// return interval [a,b) of the active part
    std::pair<int,int> interval_active(Spin s) const { if (s==up) return {p1,mid()}; else return {mid(),p2}; }

    /// return interval [a,b) of the bath part
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

    /// impose on K the symmetry of this layout, if any
    void ensure_symmetry(arma::Mat<T> &K) const
    {
        if (layout==spin_symmetric) ensure_reflection_mat(K);
    }

    /// convert to complex values. There is an specialization for `double` below.
    Fb_mps<cmpx> to_complex() const { return *this; }

    OrbitalUpdate<T> planRepresentative(arma::Mat<T> const& K,int nRef,
                                        bool use_active=false) const
    {
        OrbitalUpdate<T> update(p1,p2);
        auto plan_dw=representativeSector(K,nRef,use_active,dw);
        SectorPlan plan_up;
        if (layout==spin_symmetric)   plan_up=reflectSector(plan_dw);
        else if (layout==spin_block)  plan_up=representativeSector(K,nRef,use_active,up);

        update.append(plan_up.positions,plan_up.givens);
        update.append(plan_dw.positions,plan_dw.givens);

        // bring the representatives into the window, growing it on both sides
        int active_begin=p1, active_end=p2;
        for (int i=0; i<plan_up.count; ++i)
            update.gates.emplace_back(--active_begin,
                                      (int)plan_up.positions[plan_up.positions.size()-1-i]);
        for (int i=0; i<plan_dw.count; ++i)
            update.gates.emplace_back(active_end++,(int)plan_dw.positions[i]);
        update.active={active_begin,active_end};
        return update;
    }

    OrbitalUpdate<T> planActiveRepresentative(arma::Mat<T> const& K) const
    {
        OrbitalUpdate<T> update(p1,p2);
        auto sector=[&](Spin s) {
            auto [a,b]=interval_rotating(s);
            auto [a_imp,b_imp]=interval_impurity(s);
            if (a>=b || a_imp>=b_imp) return;

            arma::Mat<T> k12=K.rows(a_imp,b_imp-1).eval().cols(a,b-1);
            arma::vec singular_values;
            arma::Mat<T> U,V;
            my_svd(U,singular_values,V,k12,spin);
            if (singular_values.empty()) return;

            int nSv=rank(singular_values,(int)V.n_cols);
            if (nSv<=0) return;

            auto givens=(s==dw) ? GivensRotForRot_left(V.head_cols(nSv).eval())
                                : GivensRotForRot_right(V.head_cols(nSv).eval());
            GivensDaggerInPlace(givens);
            update.append(arma::regspace<arma::uvec>(a,b-1),givens);
            if (s==dw && layout==spin_symmetric) {
                auto [a_up,b_up]=interval_rotating(up);
                update.append(arma::regspace<arma::uvec>(a_up,b_up-1),
                              GivensReflect(givens,b-a));
            }
        };
        if (layout==spin_block) sector(up);
        sector(dw);
        return update;
    }

    OrbitalUpdate<T> planNaturalOrbitals(arma::Mat<T> const& cc_source) const
    {
        OrbitalUpdate<T> update(p1,p2);
        auto [a_dw,b_dw]=interval_rotating(dw);
        if (layout!=spin_block && a_dw>=b_dw) return update;   // nothing to rotate

        auto sector=[&](Spin s) {
            auto [a,b]=interval_rotating(s);
            if (a>=b) return;
            bool flipped = (s==up && layout==spin_block);

            arma::Mat<T> block=cc_source.submat(a,a,b-1,b-1).eval();
            if (flipped) block=arma::fliplr(arma::flipud(block).eval()).eval();
            arma::vec occupations;
            arma::Mat<T> orbitals;
            my_eig_sym(occupations,orbitals,block,spin);
            arma::vec activity=occupations;
            for (auto& x : activity) x=std::min(x,1-x);
            arma::Mat<T> rotation=orbitals.cols(sortActivity(activity));
            if (flipped) rotation=arma::flipud(rotation).eval();

            auto givens=flipped ? GivensRotForRot_left(rotation)
                                : GivensRotForRot_right(rotation);
            update.append(arma::regspace<arma::uvec>(a,b-1),GivensTranspose(givens));
            if (s==dw && layout==spin_symmetric) {
                auto [a_up,b_up]=interval_rotating(up);
                update.append(arma::regspace<arma::uvec>(a_up,b_up-1),
                              GivensTranspose(GivensReflect(givens,b-a)));
            }
        };
        if (layout==spin_block) sector(up);
        sector(dw);

        arma::Mat<T> rotated_cc=cc_source;
        for (auto const& gate : update.gates)
            gate.applyAsCorrelator(rotated_cc);
        ensure_symmetry(rotated_cc);

        // the window keeps the orbitals that are neither empty nor full
        if (layout==spin_block) {
            auto [a_up,b_up]=interval_rotating(up);
            if (a_up<b_up) {
                arma::uvec act=activeOrbitals(rotated_cc,a_up,b_up);
                update.active.first=act.empty() ? std::max(b_up-2,a_up)
                                                : a_up+(int)act.front();
            }
            if (a_dw<b_dw) {
                arma::uvec act=activeOrbitals(rotated_cc,a_dw,b_dw);
                update.active.second=act.empty() ? std::min(a_dw+2,b_dw)
                                                 : a_dw+(int)act.back()+1;
            }
        }
        else if (layout==spin_symmetric) {
            arma::uvec act=activeOrbitals(rotated_cc,a_dw,b_dw);
            int active_end=act.empty() ? a_dw+2 : a_dw+(int)act.back()+1;
            update.active={p1-(active_end-p2),active_end};   // grow symmetrically
        }
        else {  // leading: the window grows only to the right
            arma::uvec act=activeOrbitals(rotated_cc,a_dw,rotated_cc.n_rows);
            update.active={p1, act.empty() ? a_dw+1 : a_dw+(int)act.back()+1};
        }
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
        ensure_symmetry(cc);
        p1=update.active.first;
        p2=update.active.second;
    }

    /// update the cc in the active sector using the psi
    void update_cc()
    {
        for (auto s : {up,dw}) {
            auto [a,b]=interval_active(s);
            if (a>=b) continue;
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

    /// Energy of the Slater part. K is the kinetic energy matrix
    double SlaterEnergy(arma::Mat<T> const& K) const
    {
        double energy=0;
        for(auto s : {up,dw}) {
            auto [a,b]=interval_slater(s);
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

    /// Apply the single-site operator op to the (real-space) site i of the mps, where op
    /// is one of {"C", "Cdag", "N"} as in itensor. Only valid on the non-rotating impurity
    /// orbitals (interval_impurity_full): there site i maps exactly onto a single MPS site,
    /// so the operator is a genuine single-site operator.
    void applyLocalOp(std::string op, int i)
    {
        using namespace std;
        static const set<string> op_all={"C", "Cdag", "N"};
        if (op_all.count(op)==0)
            throw invalid_argument("Fb_mps::applyLocalOp: op is not in my list. See itensor op for Fermion");

        arma::Mat<T> Qinv=rot.st();
        int i0=(int)arma::abs(Qinv.col(i)).index_max();
        auto [a_imp,b_imp] = interval_impurity_full();
        if (i0 < a_imp || i0 >= b_imp)
            throw invalid_argument("Fb_mps::applyLocalOp: site i is not a non-rotating impurity site");

        // C and Cdag are odd operators: the sites to the left of i0 carry the
        // Jordan-Wigner string. F is diagonal and unitary, so applying it does
        // not move the orthogonality center. (N is even and needs no string.)
        // ITensor MPS sites/operators are 1-based.
        if (op!="N")
            for (int k=1; k<=i0; k++) {
                auto FA=sites.op("F",k)*psi(k);
                FA.noPrime();
                psi.set(k,FA);
            }

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

    /// The representatives of one spin sector: the Slater orbitals to rotate,
    /// the rotation itself, and how many of them enter the active window.
    struct SectorPlan {
        arma::uvec positions;
        std::vector<GivensRot<T>> givens;
        int count=0;
    };

    /// The rank kept from a set of singular values: either the fixed nSv or
    /// every value above the relative tolerance.
    int rank(arma::vec const& singular_values, int nCols) const
    {
        if (nSv>=0) return std::min<int>(nSv,nCols);
        return (int)arma::find(singular_values>tol*singular_values[0]).eval().size();
    }

    /// Order of the natural orbitals by activity min(n,1-n). The centered
    /// layouts need a stable order to keep up and dw mirror images.
    arma::uvec sortActivity(arma::vec const& activity) const
    {
        return layout==leading ? my_sort_index(activity,spin)
                               : arma::stable_sort_index(activity);
    }

    /// Positions (relative to a) of the orbitals in [a,b) that are neither empty nor full.
    arma::uvec activeOrbitals(arma::Mat<T> const& cc_source, int a, int b) const
    {
        arma::vec ni=arma::real(cc_source.diag()).eval().rows(a,b-1);
        return arma::find(ni>tol && ni<1-tol).eval();
    }

    /// Rotate the Slater orbitals of one spin sector so that their coupling to
    /// the impurity (or to the whole active window) concentrates on `count` of them.
    SectorPlan representativeSector(arma::Mat<T> const& K,int nRef,
                                    bool use_active,Spin s) const
    {
        SectorPlan plan;
        auto [a_slater,b_slater]=interval_slater(s);
        if (a_slater>=b_slater) return plan;

        arma::vec ni=occupations_ni().rows(a_slater,b_slater-1);
        plan.positions=arma::find(arma::abs(ni-nRef)<0.5).eval()+a_slater;
        if (plan.positions.empty()) return plan;

        auto [a_source,b_source]=use_active ? interval_active(s) : interval_impurity(s);
        if (a_source>=b_source) return plan;

        auto k12=K.rows(a_source,b_source-1).eval().cols(plan.positions).eval();
        arma::vec singular_values;
        arma::Mat<T> U,V;
        my_svd(U,singular_values,V,k12,spin);
        if (singular_values.empty()) return plan;

        plan.count=rank(singular_values,(int)V.n_cols);
        if (plan.count<=0) { plan.count=0; return plan; }

        plan.givens=(s==dw) ? GivensRotForRot_left(V.head_cols(plan.count).eval())
                            : GivensRotForRot_right(V.head_cols(plan.count).eval());
        GivensDaggerInPlace(plan.givens);
        return plan;
    }

    /// The up sector of a symmetric layout is the mirror image of the dw one.
    SectorPlan reflectSector(SectorPlan const& plan_dw) const
    {
        SectorPlan plan;
        if (plan_dw.positions.empty()) return plan;
        plan.positions=length()-1-arma::reverse(plan_dw.positions);
        plan.givens=GivensReflect(plan_dw.givens,(int)plan_dw.positions.size());
        plan.count=plan_dw.count;
        return plan;
    }

    /// Swap two sites inside the Slater part via a hopping MPO c†_i c_j - c†_j c_i.
    /// AutoMPO supplies the Jordan-Wigner string between non-adjacent orbitals.
    void SlaterWaveFunctionSwap(int i,int j)
    {
        if (i==j) return;
        auto [a,b]=interval_active_full();
        if (i>=a && i<b) throw std::invalid_argument("SlaterSwap for active orbital i");
        if (j>=a && j<b) throw std::invalid_argument("SlaterSwap for active orbital j");
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

template<>
inline Fb_mps<cmpx> Fb_mps<double>::to_complex() const
{
    Fb_mps<cmpx> fb;
    fb.sites = sites;
    fb.psi = psi * cmpx(1,0);
    fb.rot = rot * cmpx(1,0);
    fb.cc = cc * cmpx(1,0);
    fb.imp_size = imp_size;
    fb.p1 = p1;
    fb.p2 = p2;
    fb.layout = layout;
    fb.spin = spin;
    fb.tol = tol;
    fb.nSv = nSv;
    return fb;
}

} // namespace fbr

#endif // FBR_FB_MPS_H
