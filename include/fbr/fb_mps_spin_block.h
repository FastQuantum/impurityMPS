#ifndef FBR_FB_MPS_SPIN_BLOCK_H
#define FBR_FB_MPS_SPIN_BLOCK_H

#include "fb_mps_spin.h"   // for Spin enum and helpers

#include <algorithm>

namespace fbr {

/// Block version of Fb_mps_spin: the up/dw blocks are treated as two independent
/// fermionic problems sharing a single MPS chain.  Layout:
///   |spin_up Slater | spin_up active | imp_up | imp_dw | spin_dw active | spin_dw Slater|
/// The rotation matrix `rot` is maintained block-diagonal in spin
/// (rot.submat(0,0,L/2-1,L/2-1) acts on up; rot.submat(L/2,L/2,L-1,L-1) on dw).
/// p1 (left edge of up active window) and p2 (right edge of dw active window)
/// evolve independently — no symmetry constraint between them.
/// No spin-flip symmetry of H is assumed; no reflection enforcement.
template<class T>
struct Fb_mps_spin_block
{
    itensor::Fermion sites;
    itensor::MPS psi;
    arma::Mat<T> rot;
    arma::Mat<T> cc;
    int imp_size;
    int p1, p2;
    int natOrbDepth = -1;
    double tol = 1e-10;
    /// Fixed number of singular values kept by both extract_representative methods,
    /// identically for spin up and spin down.  Set ONCE by the dynamics solver
    /// constructor via SVD of the initial impurity–bath coupling block.
    /// Sentinel -1 = recompute dynamically each call (kept for non-dynamics use).
    int nSv = -1;

    /// Construct a Fb_mps_spin_block as a Slater state.  Same convention as Fb_mps_spin::from_slater.
    static Fb_mps_spin_block<T> from_slater(arma::Mat<T> const& rot, arma::vec const& ek, int nPart, int imp_size)
    {
        Fb_mps_spin_block<T> fb;
        fb.sites = itensor::Fermion(ek.size(), {"ConserveNf", true});
        fb.cc = arma::Mat<T>(ek.size(), ek.size(), arma::fill::zeros);
        auto state = itensor::InitState(fb.sites, "0");
        arma::uvec iek = arma::sort_index(ek);
        for (int j = 0; j < nPart; j++) {
            int k = iek[j];
            state.set(k+1, "1");
            fb.cc(k, k) = 1;
        }
        fb.psi = itensor::MPS(state);
        fb.rot = rot;
        fb.imp_size = imp_size;
        std::tie(fb.p1, fb.p2) = fb.interval_impurity_full();
        return fb;
    }

    int length() const { return sites.length(); }

    std::pair<int,int> interval_slater(Spin s) const { if (s==up) return {0,p1}; else return {p2,length()}; }
    std::pair<int,int> interval_active_full() const { return {p1,p2}; }
    std::pair<int,int> interval_active(Spin s) const { if (s==up) return {p1,length()/2}; else return {length()/2,p2}; }
    std::pair<int,int> interval_impurity_full() const { return {(length()-imp_size)/2, (length()+imp_size)/2}; }
    std::pair<int,int> interval_impurity(Spin s) const
    {
        int L=length(), d=imp_size;
        if (s==up) return {(L-d)/2, L/2};
        else       return {L/2, (L+d)/2};
    }
    std::pair<int,int> interval_bath(Spin s) const
    {
        auto [a,b] = interval_impurity(s);
        if (s==up) return {0,a};
        else       return {b,length()};
    }
    std::pair<int,int> interval_rotating(Spin s) const
    {
        auto [a0,b0]=interval_impurity(s);
        auto [a1,b1]=interval_active(s);
        if (s==up) return {a1,a0};
        else       return {b0,b1};
    }

    Fb_mps_spin_block<cmpx> to_complex() const { return *this; }

    OrbitalUpdate<T> planRepresentative(arma::Mat<T> const& K,int nRef) const
    {
        return planRepresentativeFrom(K,nRef,interval_impurity(up),interval_impurity(dw));
    }

    OrbitalUpdate<T> planActiveRepresentative(arma::Mat<T> const& K) const
    {
        OrbitalUpdate<T> update(p1,p2);
        for (Spin spin : {up,dw}) {
            auto [a,b]=interval_rotating(spin);
            if (a>=b) continue;
            auto [a_imp,b_imp]=interval_impurity(spin);
            arma::Mat<T> k12=K.rows(a_imp,b_imp-1).eval().cols(a,b-1);
            arma::vec singular_values;
            arma::Mat<T> U,V;
            arma::svd_econ(U,singular_values,V,k12);
            if (singular_values.empty()) continue;
            int n=(nSv>=0)
                    ? std::min<int>(nSv,(int)V.n_cols)
                    : (int)arma::find(singular_values>tol*singular_values[0]).eval().size();
            if (n<=0) continue;
            auto givens=(spin==dw)
                          ? GivensRotForRot_left(V.head_cols(n).eval())
                          : GivensRotForRot_right(V.head_cols(n).eval());
            GivensDaggerInPlace(givens);
            update.append(arma::regspace<arma::uvec>(a,b-1),givens);
        }
        return update;
    }

    OrbitalUpdate<T> planNaturalOrbitals(arma::Mat<T> const& cc_source) const
    {
        OrbitalUpdate<T> update(p1,p2);
        for (Spin spin : {up,dw}) {
            auto [a,b]=interval_rotating(spin);
            if (a>=b) continue;
            arma::Mat<T> block=cc_source.submat(a,a,b-1,b-1).eval();
            if (spin==up)
                block=arma::fliplr(arma::flipud(block).eval()).eval();
            arma::vec occupations;
            arma::Mat<T> orbitals;
            arma::eig_sym(occupations,orbitals,block);
            arma::vec activity=occupations;
            for (auto& x : activity) x=std::min(x,1-x);
            arma::uvec order=arma::stable_sort_index(activity);
            arma::Mat<T> rotation=orbitals.cols(order);
            if (spin==up) rotation=arma::flipud(rotation).eval();
            auto givens=(spin==dw) ? GivensRotForRot_right(rotation)
                                    : GivensRotForRot_left(rotation);
            update.append(arma::regspace<arma::uvec>(a,b-1),
                          GivensTranspose(givens));
        }

        arma::Mat<T> rotated_cc=cc_source;
        for (auto const& gate : update.gates)
            gate.applyAsCorrelator(rotated_cc);

        auto [a_up,b_up]=interval_rotating(up);
        if (a_up<b_up) {
            arma::vec ni=arma::real(rotated_cc.diag()).eval().rows(a_up,b_up-1);
            arma::uvec active=arma::find(ni>tol && ni<1-tol).eval();
            update.active.first=active.empty() ? std::max(b_up-2,a_up)
                                                : a_up+(int)active.front();
        }
        auto [a_dw,b_dw]=interval_rotating(dw);
        if (a_dw<b_dw) {
            arma::vec ni=arma::real(rotated_cc.diag()).eval().rows(a_dw,b_dw-1);
            arma::uvec active=arma::find(ni>tol && ni<1-tol).eval();
            update.active.second=active.empty() ? std::min(a_dw+2,b_dw)
                                                 : a_dw+(int)active.back()+1;
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
        p1=update.active.first;
        p2=update.active.second;
    }

    // Compatibility wrappers for the previous mutating API.
    void extract_representative(arma::Mat<T>& K,int nRef,bool use_active)
    {
        auto up_source=use_active ? interval_active(up) : interval_impurity(up);
        auto dw_source=use_active ? interval_active(dw) : interval_impurity(dw);
        auto update=planRepresentativeFrom(K,nRef,up_source,dw_source);
        update.applyAsBasis(K);
        applyUpdate(update);
    }

    void extract_representative_final(arma::Mat<T>& K)
    {
        auto update=planActiveRepresentative(K);
        update.applyAsBasis(K);
        applyUpdate(update);
    }

    void update_cc()
    {
        for (auto spin : {up, dw}) {
            auto [a, b] = interval_active(spin);
            if constexpr (std::is_same<T,double>::value) {
                auto ccz = correlationMatrix(psi, sites, "Cdag", "C", itensor::range1(a+1, b));
                for (auto i = 0u; i < ccz.size(); i++)
                    for (auto j = 0u; j < ccz[i].size(); j++)
                        cc(a+i, a+j) = ccz.at(i).at(j);
            } else {
                auto ccz = correlationMatrixC(psi, sites, "Cdag", "C", itensor::range1(a+1, b));
                for (auto i = 0u; i < ccz.size(); i++)
                    for (auto j = 0u; j < ccz[i].size(); j++)
                        cc(a+i, a+j) = ccz.at(i).at(j);
            }
        }
    }

    /// Compatibility wrapper returning the rotation on the old active interval.
    arma::Mat<T> rotateToNaturalOrbitals()
    {
        auto [a,b]=interval_active_full();
        auto update=planNaturalOrbitals(cc);
        arma::Mat<T> full(length(),length(),arma::fill::eye);
        for (auto const& gate : update.gates)
            gate.applyAsFrame(full);
        auto result=full.submat(a,a,b-1,b-1).eval();
        applyUpdate(update);
        return result;
    }

    double SlaterEnergy(arma::Mat<T> const& K) const
    {
        double energy = 0;
        for (auto spin : {up, dw}) {
            auto [a, b] = interval_slater(spin);
            for (auto i = a; i < b; i++)
                energy += std::real(cc(i,i) * K(i,i));
        }
        return energy;
    }

    arma::vec occupations_ni() const { return arma::vec(arma::real(cc.diag())); }

    arma::vec occupations_ni2() const
    {
        arma::vec ni(cc.n_rows);
        auto niv = itensor::expectC(psi, sites, "N");
        for (auto i = 0u; i < ni.size(); i++) ni[i] = niv[i].real();
        return ni;
    }

    void print_bond_dims(std::string_view msg="") const
    {
        arma::cout << msg << arma::endl;
        arma::cout << "active: " << p2 - p1 << arma::endl;
        for (auto i = 0; i+1 < psi.length(); i++)
            arma::cout << itensor::leftLinkIndex(psi, i+1).dim() << " ";
        arma::cout << arma::endl;
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
            throw invalid_argument("Fb_mps_spin_block::applyLocalOp: op is not in my list. See itensor op for Fermion");

        int i0 = frame_site(i);   // 0-based MPS orbital
        auto [a_imp,b_imp] = interval_impurity_full();
        if (i0 < a_imp || i0 >= b_imp)
            throw invalid_argument("Fb_mps_spin_block::applyLocalOp: site i is not a non-rotating impurity site");

        // ITensor MPS sites/operators are 1-based.
        psi.position(i0+1);
        auto G=sites.op(op,i0+1);
        auto newA=G*psi(i0+1);
        newA.noPrime();
        psi.set(i0+1,newA);
        update_cc();
    }

    /// Real-space correlator <c_i^dag c_j>.
    /// Convention (impurity_param.h): c_i = sum_a rot[i,a] d_a, so
    ///   <c_i^dag c_j> = (Qinv^dag cc Qinv)[i,j]  with Qinv = rot.st().
    arma::Mat<T> correlator_all() const
    {
        arma::Mat<T> Qinv = rot.st();
        return Qinv.t() * cc * Qinv;
    }

    T correlator(int i, int j) const
    {
        arma::Mat<T> Qinv = rot.st();
        arma::Col<T> ccQinv = cc * Qinv.col(j);
        return arma::cdot(Qinv.col(i), ccQinv);
    }

    arma::Col<T> correlator_all_i(int j) const
    {
        arma::Mat<T> Qinv = rot.st();
        arma::Col<T> ccQinv = cc * Qinv.col(j);
        return Qinv.t() * ccQinv;
    }

    arma::Col<T> correlator_all_j(int i) const
    {
        arma::Mat<T> Qinv = rot.st();
        arma::Row<T> Qinv_t_cc = Qinv.col(i).t() * cc;
        return (Qinv_t_cc * Qinv).st();
    }

private:
    OrbitalUpdate<T> planRepresentativeFrom(arma::Mat<T> const& K,int nRef,
                                             std::pair<int,int> up_source,
                                             std::pair<int,int> dw_source) const
    {
        struct PerSpin {
            arma::uvec positions;
            std::vector<GivensRot<T>> givens;
            int count=0;
        };

        auto compute=[&](Spin spin,std::pair<int,int> source) {
            PerSpin result;
            auto [a_slater,b_slater]=interval_slater(spin);
            if (a_slater>=b_slater) return result;
            arma::vec ni=occupations_ni().rows(a_slater,b_slater-1);
            result.positions=arma::find(arma::abs(ni-nRef)<0.5).eval()+a_slater;
            if (result.positions.empty()) return result;

            auto [a_source,b_source]=source;
            arma::Mat<T> k12=K.rows(a_source,b_source-1).eval()
                               .cols(result.positions).eval();
            arma::vec singular_values;
            arma::Mat<T> U,V;
            arma::svd_econ(U,singular_values,V,k12);
            if (singular_values.empty()) return result;
            result.count=(nSv>=0)
                           ? std::min<int>(nSv,(int)V.n_cols)
                           : (int)arma::find(singular_values>tol*singular_values[0]).eval().size();
            if (result.count<=0) return result;
            result.givens=(spin==dw)
                            ? GivensRotForRot_left(V.head_cols(result.count).eval())
                            : GivensRotForRot_right(V.head_cols(result.count).eval());
            GivensDaggerInPlace(result.givens);
            return result;
        };

        auto up_data=compute(up,up_source);
        auto dw_data=compute(dw,dw_source);
        OrbitalUpdate<T> update(p1,p2);
        update.append(up_data.positions,up_data.givens);
        update.append(dw_data.positions,dw_data.givens);

        int active_begin=p1;
        int active_end=p2;
        for (int i=0; i<up_data.count; ++i) {
            int source=(int)up_data.positions[up_data.positions.size()-1-i];
            update.gates.emplace_back(active_begin-1,source);
            active_begin--;
        }
        for (int i=0; i<dw_data.count; ++i) {
            update.gates.emplace_back(active_end,(int)dw_data.positions[i]);
            active_end++;
        }
        update.active={active_begin,active_end};
        return update;
    }

    void SlaterWaveFunctionSwap(int i, int j)
    {
        if (i == j) return;
        auto [a, b] = interval_active_full();
        if (i >= a && i < b) throw std::invalid_argument("SlaterSwap for active orbital i");
        if (j >= a && j < b) throw std::invalid_argument("SlaterSwap for active orbital j");
        T ni = cc(i,i), nj = cc(j,j);
        if (std::abs(ni - nj) < 0.5) throw std::invalid_argument("SlaterSwap for equal occupations");

        itensor::AutoMPO ampo(sites);
        ampo += 1.0, "Cdag", i+1, "C", j+1;
        ampo += -1.0, "Cdag", j+1, "C", i+1;
        auto H = itensor::toMPO(ampo);
        psi = itensor::applyMPO(H, psi, {"Cutoff", tol, "Normalize", false});
        psi.noPrime();
    }
};

template<>
inline Fb_mps_spin_block<cmpx> Fb_mps_spin_block<double>::to_complex() const
{
    Fb_mps_spin_block<cmpx> fb;
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

#endif // FBR_FB_MPS_SPIN_BLOCK_H
