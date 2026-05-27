#ifndef IMPURITY_DYN_SPIN_BLOCK_H
#define IMPURITY_DYN_SPIN_BLOCK_H

#include "fermionic.h"
#include "impurity_param_spin.h"
#include "impurity_spin_init.h"
#include "fb_mps_spin_block.h"

#include "tdvp.h"
#include "basisextension.h"

/// Block version of Impurity_dyn_spin: spin up/down are treated as two independent
/// fermionic blocks of a shared MPS chain.  No spin-flip symmetry of H is assumed.
struct Impurity_dyn_spin_block {
    ImpurityParamSpin param;
    double dt;
    arma::cx_mat Kbath;
    arma::cx_mat Kip0;
    arma::uvec imp_pos;
    arma::uvec bath_pos;
    arma::cx_mat rotS;

    Fb_mps_spin_block<cmpx> fb;
    arma::cx_mat K;
    double energy = -1000;
    int nIter = 0;

    explicit Impurity_dyn_spin_block(ImpuritySpin const& imp,
                                     Fb_mps_spin_block<cmpx> const& fb_,
                                     double dt_=0.1)
        : param(imp.param), dt(dt_), fb { fb_ }
    {
        int L = param.length();
        using namespace arma;

        imp_pos  = arma::conv_to<arma::uvec>::from(param.impPos);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L, param.impPos));

        mat Kstar = param.Kmat;
        cx_mat Kbath_full(L, L, arma::fill::zeros);
        Kbath_full(bath_pos, bath_pos) = Kstar(bath_pos, bath_pos) * cmpx(1, 0);
        cx_mat commutator = Kstar*Kbath_full - Kbath_full*Kstar;
        Kip0 = Kstar - Kbath_full - cmpx(0, 0.5*dt) * commutator;

        rotS = fb.rot;
        arma::cx_mat K0 = param.Kmat * cmpx(1, 0);
        this->Kbath = K0.submat(bath_pos, bath_pos);

        // Fix nSv = rank of the impurity–bath coupling block at construction
        // (per spin, take max). Same value used for every extract_representative*
        // call thereafter, for both spin up and spin down.
        {
            int nSv_max = 0;
            for (Spin spin : {up, dw}) {
                auto [a_imp, b_imp] = fb.interval_impurity(spin);
                auto [a_sla, b_sla] = fb.interval_slater(spin);
                if (a_imp < b_imp && a_sla < b_sla) {
                    arma::cx_mat k12 = param.Kmat.submat(a_imp, a_sla, b_imp-1, b_sla-1) * cmpx(1,0);
                    arma::vec s; arma::cx_mat U, V;
                    arma::svd_econ(U, s, V, k12);
                    int n = (s.is_empty() || s[0] == 0) ? 0
                              : (int)arma::find(s > fb.tol*s[0]).eval().size();
                    nSv_max = std::max(nSv_max, n);
                }
            }
            fb.nSv = nSv_max;
        }
    }

    /// Convenience constructor from ImpuritySpinInit (arbitrary-ordering input).
    /// The Fb_mps_spin produced by ImpuritySpinInit is copied field-by-field into a
    /// Fb_mps_spin_block (same chain layout, block-diagonal rot already).
    explicit Impurity_dyn_spin_block(ImpuritySpinInit const& init, double dt_=0.1)
        : Impurity_dyn_spin_block(init.model, fromSpinFb(init.fb), dt_) {}

    void iterate(TdvpParam args={})
    {
        int L = fb.length();
        arma::cx_mat exp_ih(L, L, arma::fill::eye);
        exp_ih.submat(bath_pos, bath_pos) = expIH<cmpx>(Kbath * nIter * dt);

        arma::cx_mat rot = exp_ih * rotS.t() * fb.rot;
        K = rot.t() * Kip0 * rot;
        nIter++;

        extract_representative(0);
        extract_representative(1);
        extract_representative_final();
        doTdvp(args);
        rotateToNaturalOrbitals();
    }

    void extract_representative(int nRef) { fb.extract_representative(K, nRef, /*use_active=*/false); }
    void extract_representative_final()   { fb.extract_representative_final(K); }

    void doTdvp(TdvpParam args={})
    {
        auto [a, b] = fb.interval_active_full();
        auto mpo = fullHamiltonian(a, b);

        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;

        if (args.epsilonM != 0) {
            std::vector<double> epsilonK(args.nKrylov, 1E-8);
            itensor::addBasis(fb.psi, mpo, epsilonK,
                              {"Cutoff", args.epsilonM,
                               "Method", "DensityMatrix",
                               "KrylovOrd", args.nKrylov,
                               "DoNormalize", true,
                               "Quiet", true,
                               "Silent", true});
        }

        energy = itensor::tdvp(fb.psi, mpo, -imag_1*dt, sweeps,
                               {"Truncate", true,
                                "DoNormalize", true,
                                "Quiet", true,
                                "Silent", true,
                                "NumCenter", 2,
                                "ErrGoal", args.err_goal});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    void rotateToNaturalOrbitals()
    {
        auto [a, b] = fb.interval_active_full();
        auto rot1 = fb.rotateToNaturalOrbitals();
        K.cols(a, b-1) = K.cols(a, b-1).eval() * rot1;
        K.rows(a, b-1) = rot1.t() * K.rows(a, b-1).eval();
    }

    /// Effective MPS->real rotation in the Schrödinger picture.
    /// The MPS lives in the interaction picture of H_bath, so fb.rot tracks only the
    /// natural-orbital basis change. To recover real-space Schrödinger-picture
    /// correlators, we dress the bath block with exp(-i Kbath * t).
    arma::cx_mat effective_rot() const
    {
        int L = fb.length();
        arma::cx_mat exp_ih(L, L, arma::fill::eye);
        if (nIter > 0)
            exp_ih.submat(bath_pos, bath_pos) = expIH<cmpx>(Kbath * (static_cast<double>(nIter) * dt));
        return rotS * exp_ih * rotS.t() * fb.rot;
    }

    /// Schrödinger-picture real-space <c_i^dag c_j> matrix.
    arma::cx_mat correlator_all() const
    {
        arma::cx_mat Q = effective_rot();
        return arma::conj(Q) * fb.cc * Q.st();
    }

    /// Schrödinger-picture real-space <c_i^dag c_j>.
    cmpx correlator(int i, int j) const
    {
        arma::cx_mat Q = effective_rot();
        arma::cx_vec ccQj = fb.cc * Q.row(j).st();
        return arma::cdot(Q.row(i).st(), ccQj);
    }

    /// Row of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed j, all i.
    arma::cx_vec correlator_all_i(int j) const
    {
        arma::cx_mat Q = effective_rot();
        arma::cx_vec ccQj = fb.cc * Q.row(j).st();
        return arma::conj(Q) * ccQj;
    }

    /// Column of the Schrödinger-picture correlator: <c_i^dag c_j> for fixed i, all j.
    arma::cx_vec correlator_all_j(int i) const
    {
        arma::cx_mat Q = effective_rot();
        arma::cx_rowvec v = arma::conj(Q.row(i)) * fb.cc;
        return Q * v.st();
    }

    itensor::MPO fullHamiltonian(int a, int b) const
    {
        itensor::AutoMPO h(fb.sites);
        int L = param.length();
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (std::abs(param.Umat(i,j)) > 1e-15)
                    h += param.Umat(i,j), "N", i+1, "N", j+1;
        for (auto i = a; i < b; i++)
            for (auto j = a; j < b; j++)
                if (std::abs(K(i,j)) > fb.tol)
                    h += K(i,j), "Cdag", i+1, "C", j+1;
        return itensor::toMPO(h);
    }

private:
    static Fb_mps_spin_block<cmpx> fromSpinFb(Fb_mps_spin<cmpx> const& src)
    {
        Fb_mps_spin_block<cmpx> dst;
        dst.sites       = src.sites;
        dst.psi         = src.psi;
        dst.rot         = src.rot;
        dst.cc          = src.cc;
        dst.imp_size    = src.imp_size;
        dst.p1          = src.p1;
        dst.p2          = src.p2;
        dst.natOrbDepth = src.natOrbDepth;
        dst.tol         = src.tol;
        // dst.nSv is left at its default sentinel (-1); the dynamics constructor
        // overwrites it with the SVD-based rank of the impurity–bath block.
        return dst;
    }
};

#endif // IMPURITY_DYN_SPIN_BLOCK_H
