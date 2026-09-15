// Full-MPS (no active window) helpers for the bond-dimension studies in app/:
// the whole L-site MPS is kept, in whatever single-particle basis K is written
// (the real-space chain, or the star of ImpurityParam::to_star()). This is the
// brute-force cost the FBR's active window removes.
//
// Include from ONE translation unit per executable (the TDVP headers define
// addBasis non-inline).

#ifndef APP_FULL_MPS_H
#define APP_FULL_MPS_H

#include <itensor/all.h>
#include <fbr/itensor_utils.h>
#include <fbr/fbr_dyn.h>   // brings tdvp.h and basisextension.h (no include guard of its own)
#include <armadillo>

#include <iostream>
#include <iomanip>
#include <vector>

namespace full_mps {

using cmpx = std::complex<double>;

/// H = sum_ij K(i,j) c_i^dag c_j + sum_ij Umat(i,j) n_i n_j, in the basis of K.
inline itensor::MPO hamiltonian(itensor::Fermion const& sites, arma::mat const& K,
                                arma::mat const& Umat)
{
    int L = K.n_rows;
    itensor::AutoMPO h(sites);
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++) {
            if (std::abs(Umat(i, j)) > 1e-15) h += Umat(i, j), "N", i + 1, "N", j + 1;
            if (std::abs(K(i, j)) > 1e-12)    h += K(i, j), "Cdag", i + 1, "C", j + 1;
        }
    return itensor::toMPO(h);
}

/// Product state filling the n orbitals of lowest ek (the occupation basis of K).
inline itensor::MPS product_state(itensor::Fermion const& sites, arma::vec const& ek, int n)
{
    auto state = itensor::InitState(sites, "0");
    arma::uvec iek = arma::stable_sort_index(ek);
    for (int j = 0; j < n; j++) state.set((int)iek[j] + 1, "1");
    return itensor::MPS(state);
}

/// Ground state by DMRG. In the star the impurity couples to every mode (long
/// range) and plain DMRG traps easily in a wrong-occupation minimum: a strong,
/// slowly decaying noise and a bond dimension grown over many sweeps shake it
/// loose (the recipe of test/ref/star_green_siam.cpp). Seed psi with the
/// impurity occupied.
inline double ground_state(itensor::MPS& psi, itensor::MPO const& mpo, int nsweep = 60,
                           double cutoff = 1e-12)
{
    std::vector<double> noise = {1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-8, 0};
    double e = 0;
    for (int i = 0; i < nsweep; i++) {
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = std::min(64 + 32 * i, 1024);
        sweeps.cutoff() = cutoff;
        sweeps.niter()  = 4;
        sweeps.noise()  = i < (int)noise.size() ? noise[i] : 0.0;
        e = itensor::dmrg(psi, mpo, sweeps, {"Quiet", true, "Silent", true});
        if (i % 10 == 9)
            std::cout << "#   dmrg sweep " << i + 1 << " m=" << itensor::maxLinkDim(psi)
                      << " E=" << std::setprecision(12) << e << std::endl;
    }
    return e;
}

/// The fermionic c_site^dag (0-based site), Jordan-Wigner string included.
inline void apply_cdag(itensor::Fermion const& sites, itensor::MPS& psi, int site)
{
    for (int k = 1; k <= site; k++) {
        auto A = sites.op("F", k) * psi(k);
        A.noPrime();
        psi.set(k, A);
    }
    psi.position(site + 1);
    auto A = sites.op("Cdag", site + 1) * psi(site + 1);
    A.noPrime();
    psi.set(site + 1, A);
}

/// <n_site> (0-based), a local operator in any basis where the site is not rotated.
inline double density(itensor::Fermion const& sites, itensor::MPS psi, int site)
{
    psi.position(site + 1);
    auto ket = psi(site + 1);
    return std::real(itensor::eltC(itensor::dag(itensor::prime(ket, "Site"))
                                   * sites.op("N", site + 1) * ket));
}

/// One two-site TDVP step, truncation at `cutoff`. `expand`: global subspace
/// expansion with the tuned star set (TdvpParam defaults) -- needed in the star,
/// where the impurity couples to every mode; a nearest-neighbour chain grows its
/// bonds through the two-site update alone (measured in test/ref/chain_green_irlm.cpp).
inline void tdvp_step(itensor::MPS& psi, itensor::MPO const& mpo, double dt, double cutoff,
                      int max_bond_dim, bool expand = true)
{
    fbr::TdvpParam args{};
    auto sweeps = itensor::Sweeps(1);
    sweeps.maxdim() = max_bond_dim;
    sweeps.cutoff() = cutoff;
    sweeps.niter()  = args.n_iter_diag;
    sweeps.noise()  = args.noise;

    std::vector<double> epsilon_K(args.n_krylov, args.epsilon_K);
    if (expand)
    itensor::addBasis(psi, mpo, epsilon_K,
                      {"Cutoff", args.epsilon_M, "Method", "DensityMatrix",
                       "KrylovOrd", args.n_krylov, "DoNormalize", true,
                       "Quiet", true, "Silent", true});

    itensor::tdvp(psi, mpo, -cmpx(0, 1) * dt, sweeps,
                  {"Truncate", true, "DoNormalize", true, "Quiet", true,
                   "Silent", true, "NumCenter", 2, "ErrGoal", args.err_goal});
}

} // namespace full_mps

#endif // APP_FULL_MPS_H
