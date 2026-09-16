#ifndef FBR_ITENSOR_UTILS_H
#define FBR_ITENSOR_UTILS_H

#include "givens_rotation.h"
#include <itensor/all.h>

namespace fbr {

struct DmrgParam {
    int max_bond_dim=512;
    int n_iter_diag=4;
    double noise=1e-8;
};

/// Control parameters for a TDVP time step: bond-dimension limits, local Krylov evolution, and addBasis local basis expansion.
struct TdvpParam {
    // --- pure TDVP parameters ---
    int max_bond_dim=1024;  ///< Maximum MPS bond dimension during the TDVP sweep.
    double noise=0;         ///< ITensor sweep noise term.
    int n_iter_diag=16;      ///< Krylov iterations used to apply exp(-i * Heff * dt) locally.
    double err_goal=1e-7;   ///< TDVP local evolution error goal.
    // --- addBasis (global subspace expansion) parameters ---
    // Defaults tuned on the star-geometry SIAM benchmark (test/ref/star_dyn_tune.cpp):
    // this set tracks the chain baseline as tightly as the old overkill
    // (n_krylov=15, err_goal=1e-8, epsilon_M=1e-7, epsilon_K=1e-8) at ~4x less cost.
    // n_krylov is the cheap knob (15->2 is free); err_goal and the two epsilon cutoffs
    // are sensitive (~1 order of loosening is the safe limit). FBR callers set
    // epsilon_M=0 to skip the expansion entirely, so n_krylov/epsilon_K are inert there.
    double epsilon_M=3e-7;   ///< addBasis density-matrix cutoff; set to 0 to skip basis expansion.
    int n_krylov=2;          ///< Krylov order of the addBasis global subspace expansion
    double epsilon_K=3e-8;   ///< add basis cutoff for each Krylov-vector
};

inline arma::cx_mat get_cc(itensor::Fermion const& sites, itensor::MPS const& psi)
{
    arma::cx_mat cc(sites.length(), sites.length());
    auto ccz=correlationMatrixC(psi, sites,"Cdag","C");
    for(auto i=0u; i<ccz.size(); i++)
        for(auto j=0u; j<ccz[i].size(); j++)
            cc(i,j)=ccz.at(i).at(j);
    return cc;
}

inline arma::vec get_ni(itensor::Fermion const& sites, itensor::MPS const& psi)
{
    arma::vec ni(sites.length());
    auto niz=expectC(psi, sites,"N");
    for(auto i=0u; i<sites.length(); i++)
        ni[i]=std::real(niz[i]);
    return ni;
}

/// Renyi-1/2 entanglement entropy across the bond between sites b and b+1
/// (1-based b). With Schmidt values s_i (sum s_i^2 = 1, so the reduced density
/// matrix eigenvalues are p_i = s_i^2), S_{1/2} = 2 ln(sum_i sqrt(p_i)) = 2 ln(sum_i s_i).
/// A copy of psi is taken so the caller's gauge is untouched.
inline double renyi_half_at(itensor::MPS psi, int b)
{
    psi.position(b);
    auto site_b = itensor::siteIndex(psi, b);
    auto [U, S, V] = (b > 1)
        ? itensor::svd(psi(b), {itensor::leftLinkIndex(psi, b), site_b})
        : itensor::svd(psi(b), {site_b});
    auto si = itensor::commonIndex(U, S);
    double sum_sqrt = 0;
    for (int n = 1; n <= itensor::dim(si); n++) sum_sqrt += itensor::elt(S, n, n);
    return 2.0 * std::log(sum_sqrt);
}

/// Maximum link dimension and the Renyi-1/2 entropy at that (bottleneck) bond.
/// One SVD per call: cheap next to a TDVP step, and the max-bond entropy is the
/// proxy for whether the truncation is discarding real weight or just noise.
inline std::pair<int,double> max_bond_and_renyi_half(itensor::MPS const& psi)
{
    int L = itensor::length(psi);
    int bstar = 1, mmax = 1;
    for (int i = 1; i < L; i++) {
        int m = itensor::dim(itensor::linkIndex(psi, i));
        if (m >= mmax) { mmax = m; bstar = i; }
    }
    return {mmax, renyi_half_at(psi, bstar)};
}

/// Renyi-1/2 entropy at EVERY bond, in one left-to-right sweep: returns
/// {sum over bonds, max over bonds}. The sum is extensive and smooth in time,
/// unlike the single max-bond value which jumps when the bottleneck bond moves.
/// A copy of psi is swept, so the caller's gauge is untouched. Bonds with link
/// dimension 1 contribute exactly 0 and are skipped.
inline std::pair<double,double> renyi_half_profile(itensor::MPS psi)
{
    int L = itensor::length(psi);
    if (L < 2) return {0.0, 0.0};
    psi.position(1);
    psi.normalize();
    double total = 0, mx = 0;
    for (int b = 1; b < L; b++) {
        auto site_b = itensor::siteIndex(psi, b);
        auto [U, S, V] = (b > 1)
            ? itensor::svd(psi(b), {itensor::leftLinkIndex(psi, b), site_b}, {"Cutoff", 0.0})
            : itensor::svd(psi(b), {site_b}, {"Cutoff", 0.0});
        auto si = itensor::commonIndex(U, S);
        double ss = 0;
        for (int n = 1; n <= itensor::dim(si); n++) ss += itensor::elt(S, n, n);
        double s_half = ss > 0 ? 2.0 * std::log(ss) : 0.0;
        total += s_half;
        if (s_half > mx) mx = s_half;
        psi.set(b, U);                     // move the orthogonality center to b+1
        psi.set(b + 1, S * V * psi(b + 1));
    }
    return {total, mx};
}

/// The ITensor two-site gates that apply a circuit of Givens rotations to an MPS,
/// used to rotate the active window into its natural orbitals.
template<class T>
std::vector<itensor::BondGate> gates_from_givens(itensor::Fermion const& sites, std::vector<GivensRot<T>> const& gs)
{
    using itensor::BondGate;
    std::vector<itensor::BondGate> gates;
    for(const GivensRot<T>& g : gs)
    {
        int b=g.b+1;
        auto rot=g.matrix().t().eval();

        auto s1 = itensor::dag(sites(b));
        auto s2 = itensor::dag(sites(b+1));
        auto s1p = prime(sites(b));
        auto s2p = prime(sites(b+1));
        itensor::ITensor hterm(s1,s2,s1p,s2p);
        hterm.set(s1(1),s2(1),s1p(1),s2p(1), 1);
        hterm.set(s1(2),s2(2),s1p(2),s2p(2), 1);
        hterm.set(s1(2),s2(1),s1p(2),s2p(1), rot(0,0));
        hterm.set(s1(2),s2(1),s1p(1),s2p(2), rot(0,1));
        hterm.set(s1(1),s2(2),s1p(2),s2p(1), rot(1,0));
        hterm.set(s1(1),s2(2),s1p(1),s2p(2), rot(1,1));

        if (hterm) {
            auto bg=BondGate(sites,b,b+1,hterm);
            gates.push_back(bg);
        }
    }
    return gates;
}

} // namespace fbr

#endif // FBR_ITENSOR_UTILS_H
