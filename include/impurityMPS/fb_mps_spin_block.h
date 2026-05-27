#ifndef FB_MPS_SPIN_BLOCK_H
#define FB_MPS_SPIN_BLOCK_H

#include "fb_mps_spin.h"   // for Spin enum and helpers

#include <algorithm>

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

    /// Promote bath orbitals with occupation ~nRef into the active window, processing
    /// each spin block independently.  p1 and p2 are updated independently per spin.
    void extract_representative(arma::Mat<T>& K, int nRef, bool use_active)
    {
        struct PerSpin {
            arma::uvec pos0;
            std::vector<GivensRot<T>> givens;
            arma::Mat<T> rot_block;
            int nSv_use = 0;
        };

        arma::vec ni = occupations_ni();

        auto compute = [&](Spin spin) -> PerSpin {
            PerSpin r;
            auto [a_s, b_s] = interval_slater(spin);
            if (a_s >= b_s) return r;
            arma::vec ni_slater = ni.rows(a_s, b_s-1);
            arma::vec delta = arma::abs(ni_slater - nRef);
            r.pos0 = arma::find(delta < 0.5).eval() + a_s;
            if (r.pos0.empty()) return r;

            auto [a, b] = use_active ? interval_active(spin) : interval_impurity(spin);
            arma::Mat<T> k12 = K.rows(a, b-1).eval().cols(r.pos0).eval();
            arma::vec s;
            arma::Mat<T> U, V;
            arma::svd_econ(U, s, V, k12);
            r.nSv_use = (this->nSv >= 0)
                          ? std::min<int>(this->nSv, (int)V.n_cols)
                          : (int)arma::find(s > tol*s[0]).eval().size();
            if (r.nSv_use <= 0) return r;
            arma::Mat<T> V_eff = V.head_cols(r.nSv_use);
            r.givens = (spin == dw) ? GivensRotForRot_left(V_eff)
                                    : GivensRotForRot_right(V_eff);
            GivensDaggerInPlace(r.givens);
            r.rot_block = matrot_from_Givens(r.givens, k12.n_cols);
            return r;
        };

        PerSpin du = compute(up);
        PerSpin dd = compute(dw);

        if (du.pos0.empty() && dd.pos0.empty()) return;

        // 3. apply per-spin rotation to K, rot (cc is unchanged because pos0 lie in slater,
        //    where cc is diagonal == nRef on those entries)
        auto apply_rot = [&](PerSpin const& d) {
            if (d.pos0.empty() || d.givens.empty()) return;
            arma::Mat<T> rot1(length(), length(), arma::fill::eye);
            rot1(d.pos0, d.pos0) = d.rot_block;
            K = rot1.t() * K * rot1;
            rot = rot * rot1;
        };
        apply_rot(du);
        apply_rot(dd);

        // 4. promote each spin's orbitals independently — p1 and p2 evolve separately.
        auto promote = [&](Spin spin, PerSpin const& d) {
            int n = std::min<int>(d.nSv_use, (int)d.pos0.size());
            for (int i = 0; i < n; i++) {
                auto [a, b] = interval_active_full();
                int i1, i2;
                if (spin == dw) {
                    i1 = b;
                    i2 = d.pos0[i];
                } else {
                    i1 = a - 1;
                    i2 = d.pos0[d.pos0.size() - 1 - i];
                }
                if (i1 != i2) {
                    SlaterWaveFunctionSwap(i1, i2);
                    K.swap_cols(i1, i2);
                    K.swap_rows(i1, i2);
                    rot.swap_cols(i1, i2);
                    cc.swap_cols(i1, i2);
                    cc.swap_rows(i1, i2);
                }
                if (spin == dw) p2++; else p1--;
            }
        };
        promote(up, du);
        promote(dw, dd);
    }

    /// Rotate each spin's rotating interval to concentrate the impurity-bath
    /// coupling at the leading positions next to the impurity.
    void extract_representative_final(arma::Mat<T>& K)
    {
        struct PerSpin {
            std::vector<GivensRot<T>> givens;
            int a = 0, b = 0;
        };

        auto compute = [&](Spin spin) -> PerSpin {
            PerSpin r;
            std::tie(r.a, r.b) = interval_rotating(spin);
            if (r.a >= r.b) return r;
            auto [a_imp, b_imp] = interval_impurity(spin);
            arma::Mat<T> k12 = K.rows(a_imp, b_imp-1).eval().cols(r.a, r.b-1);
            arma::vec s;
            arma::Mat<T> U, V;
            arma::svd_econ(U, s, V, k12);
            int nSv_use = (this->nSv >= 0)
                            ? std::min<int>(this->nSv, (int)V.n_cols)
                            : (int)arma::find(s > tol*s[0]).eval().size();
            if (nSv_use <= 0) return r;
            arma::Mat<T> V_eff = V.head_cols(nSv_use);
            r.givens = (spin == dw) ? GivensRotForRot_left(V_eff)
                                    : GivensRotForRot_right(V_eff);
            GivensDaggerInPlace(r.givens);
            return r;
        };

        PerSpin u_data = compute(up);
        PerSpin d_data = compute(dw);

        auto apply_KCCRot = [&](PerSpin const& d) {
            if (d.a >= d.b || d.givens.empty()) return;
            arma::Mat<T> rot1 = matrot_from_Givens(d.givens, d.b - d.a);
            K.cols(d.a, d.b-1) = K.cols(d.a, d.b-1).eval() * rot1;
            K.rows(d.a, d.b-1) = rot1.t() * K.rows(d.a, d.b-1).eval();
            cc.cols(d.a, d.b-1) = cc.cols(d.a, d.b-1).eval() * rot1.st().t();
            cc.rows(d.a, d.b-1) = rot1.st() * cc.rows(d.a, d.b-1).eval();
            rot.cols(d.a, d.b-1) = rot.cols(d.a, d.b-1).eval() * rot1;
        };
        apply_KCCRot(u_data);
        apply_KCCRot(d_data);

        // Apply gates on the MPS as ONE combined sweep so the orthogonality
        // center traverses the chain once rather than twice.
        {
            std::vector<GivensRot<T>> gQ_all;
            for (auto const& d : {u_data, d_data}) {
                if (d.givens.empty()) continue;
                auto gQ = d.givens;
                for (auto& g : gQ) g.b += d.a;
                gQ_all.insert(gQ_all.end(), gQ.begin(), gQ.end());
            }
            if (!gQ_all.empty()) {
                auto gates = Fermionic::NOGates(sites, GivensTranspose(gQ_all));
                itensor::gateTEvol(gates, 1, 1, psi,
                                   {"Cutoff", tol, "Quiet", true, "Normalize", false, "ShowPercent", false});
            }
        }
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

    /// Diagonalize cc on each spin's rotating interval; keep p1, p2 symmetric.
    /// Returns the rotation applied to the active_full block.
    arma::Mat<T> rotateToNaturalOrbitals()
    {
        auto [a_full, b_full] = interval_active_full();
        arma::Mat<T> rot_update(length(), length(), arma::fill::eye);

        struct PerSpin {
            std::vector<GivensRot<T>> givens;
            int a = 0, b = 0;
        };

        auto diag = [&](Spin spin) -> PerSpin {
            PerSpin r;
            std::tie(r.a, r.b) = interval_rotating(spin);
            if (r.a >= r.b) return r;
            arma::Mat<T> cc1(cc.submat(r.a, r.a, r.b-1, r.b-1).eval());
            // For up: reflect the matrix before eig_sym so LAPACK sees the same layout
            // as for dw under a reflection-symmetric input.  We reflect the resulting
            // eigenvectors back afterwards.
            if (spin == up) cc1 = arma::fliplr(arma::flipud(cc1).eval()).eval();
            arma::vec eval;
            arma::Mat<T> evec;
            eig_sym(eval, evec, cc1);
            arma::vec activity = eval;
            for (auto& x : activity) x = std::min(x, 1-x);
            arma::uvec iek = arma::stable_sort_index(activity);
            arma::Mat<T> rotation = evec.cols(iek);
            if (spin == up) rotation = arma::flipud(rotation).eval();
            r.givens = (spin == dw) ? GivensRotForRot_right(rotation)
                                    : GivensRotForRot_left(rotation);
            return r;
        };

        PerSpin u_data = diag(up);
        PerSpin d_data = diag(dw);

        // Apply rotation to rot, cc per spin
        auto apply_KCCRot = [&](PerSpin const& d) {
            if (d.givens.empty()) return;
            auto rot1 = matrot_from_Givens(d.givens, d.b - d.a);
            rot_update.submat(d.a, d.a, d.b-1, d.b-1) = rot1.st();
            cc.cols(d.a, d.b-1) = cc.cols(d.a, d.b-1).eval() * rot1.t();
            cc.rows(d.a, d.b-1) = rot1 * cc.rows(d.a, d.b-1).eval();
        };
        apply_KCCRot(u_data);
        apply_KCCRot(d_data);
        rot = rot * rot_update;

        // Apply gates on the MPS as ONE combined sweep so the orthogonality
        // center traverses the chain once rather than twice.
        {
            std::vector<GivensRot<T>> gQ_all;
            for (auto const& d : {u_data, d_data}) {
                if (d.givens.empty()) continue;
                auto gQ = d.givens;
                for (auto& g : gQ) g.b += d.a;
                gQ_all.insert(gQ_all.end(), gQ.begin(), gQ.end());
            }
            if (!gQ_all.empty()) {
                auto gates = Fermionic::NOGates(sites, gQ_all);
                itensor::gateTEvol(gates, 1, 1, psi,
                                   {"Cutoff", tol, "Quiet", true, "Normalize", false, "ShowPercent", false});
            }
        }

        // Determine new (p1, p2) per spin, independently — no symmetry constraint.
        arma::vec ni = occupations_ni();

        if (u_data.a < u_data.b) {
            int a_up = u_data.a, b_up = u_data.b;
            arma::vec ni_up = ni.rows(a_up, b_up-1);
            arma::uvec pos_active = arma::find(ni_up > tol && ni_up < 1-tol).eval();
            p1 = pos_active.empty() ? std::max(b_up - 2, a_up)
                                    : (a_up + (int)pos_active.front());
        }
        if (d_data.a < d_data.b) {
            int a_dw = d_data.a, b_dw = d_data.b;
            arma::vec ni_dw = ni.rows(a_dw, b_dw-1);
            arma::uvec pos_active = arma::find(ni_dw > tol && ni_dw < 1-tol).eval();
            p2 = pos_active.empty() ? std::min(a_dw + 2, b_dw)
                                    : (a_dw + (int)pos_active.back() + 1);
        }

        return rot_update.submat(a_full, a_full, b_full-1, b_full-1);
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

#endif // FB_MPS_SPIN_BLOCK_H
