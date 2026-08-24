#ifndef FBR_DYN_SPIN_H
#define FBR_DYN_SPIN_H

#include "graph.h"
#include "itensor_utils.h"
#include "impurity_param_spin.h"
#include "fb_mps_spin.h"

#include "tdvp.h"
#include "basisextension.h"

namespace fbr {

struct Fbr_dyn_spin {
    ImpurityParamSpin param;
    double dt;
    arma::cx_mat Kbath;
    arma::cx_mat Kip0;
    arma::uvec imp_pos;
    arma::uvec bath_pos;
    arma::cx_mat rotS;

    /// these quantities are updated during the iterations
    Fb_mps_spin<cmpx> fb;        ///< the current few body MPS
    arma::cx_mat K;         ///< the current Hamiltonian
    double energy=-1000;        // TODO remove energy (or compute it)
    int nIter=0;

    explicit Fbr_dyn_spin(ImpuritySpin const& imp, Fb_mps_spin<cmpx> const& fb_, double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , fb { fb_ }
    {
        int L=param.length();

        {
            using namespace arma;

            imp_pos  = arma::conv_to<arma::uvec>::from(param.impPos);
            bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L, param.impPos));

            mat Kstar=param.Kmat;

            // Star geometry: the bath-bath block of Kstar is diagonal. Let d be
            // that diagonal embedded in an L-vector (zero on impurity sites).
            // With D=diag(d), the commutator entries are
            //   (Kstar*D - D*Kstar)(i,j) = Kstar(i,j)*(d(j)-d(i)),
            // so column/row scaling gives it in O(L^2) instead of the two dense
            // O(L^3) products Kstar*Kbath_full and Kbath_full*Kstar.
            vec d(L, fill::zeros);
            d(bath_pos) = vec(Kstar.diag())(bath_pos);
            mat c1 = Kstar; c1.each_row() %= d.t();   // Kstar*D
            mat c2 = Kstar; c2.each_col() %= d;       // D*Kstar
            mat commutator = c1 - c2;

            Kip0 = Kstar * cmpx(1,0);
            Kip0(bath_pos,bath_pos).zeros();          // Kstar - Kbath_full (arrow)
            Kip0 -= cmpx(0,0.5*dt) * commutator;

            rotS = fb.rot;

            this->Kbath = Kstar.submat(bath_pos,bath_pos) * cmpx(1,0);
        }

        // Fix nSv = rank of the impurity–bath coupling block at construction.
        // Same value used for every extract_representative* call thereafter.
        {
            auto [a_imp, b_imp] = fb.interval_impurity(dw);
            auto [a_sla, b_sla] = fb.interval_slater(dw);
            if (a_imp < b_imp && a_sla < b_sla) {
                arma::cx_mat k12 = param.Kmat.submat(a_imp, a_sla, b_imp-1, b_sla-1) * cmpx(1,0);
                arma::vec s; arma::cx_mat U, V;
                arma::svd_econ(U, s, V, k12);
                fb.nSv = (s.is_empty() || s[0] == 0) ? 0
                          : (int)arma::find(s > fb.tol*s[0]).eval().size();
            }
        }
    }

    void iterate(TdvpParam args={})
    {
        K = buildK();   // interaction-picture Hamiltonian, O(L^2)
        nIter++;

        applyPlan(fb.planRepresentative(K,0));
        applyPlan(fb.planRepresentative(K,1));
        applyPlan(fb.planActiveRepresentative(K));
        doTdvp(args);
        applyPlan(fb.planNaturalOrbitals(fb.cc));
    }

    /// Diagonal of the interaction-picture phase exp(-i H_bath * n*dt).
    /// In star geometry H_bath is diagonal, so this is computed in O(L_bath),
    /// avoiding the O(L^3) dense matrix exponential.
    arma::cx_vec ipPhase(int n) const
    {
        int L = fb.length();
        arma::cx_vec d(L, arma::fill::ones);
        if (n > 0)
            d(bath_pos) = arma::exp(-imag_1 * Kbath.diag() * (static_cast<double>(n)*dt));
        return d;
    }

    /// Interaction-picture Hamiltonian K = rot^dag * Kip0 * rot, with
    /// rot = diag(ipPhase) * rotS^dag * fb.rot.
    /// O(L^2): Kip0 is Hermitian and its bath-bath block is exactly zero (a "cross"
    /// matrix), so only the nImp impurity rows of rot are ever needed. Writing
    /// Kip0 = e_I M + M^dag e_I^dag - e_I D e_I^dag (I = impurity indices,
    /// M = Kip0.rows(I), D = Kip0(I,I)) gives the rank-2*nImp update
    ///   K = A^dag B + B^dag A - A^dag D A,   A = rot.rows(I),  B = M*rot.
    arma::cx_mat buildK() const
    {
        arma::cx_vec d = ipPhase(nIter);
        // A = rot.rows(imp_pos); exp_ih is identity on impurity rows, so it drops out.
        arma::cx_mat A = rotS.cols(imp_pos).t() * fb.rot;   // nImp x L
        // B = M * rot, evaluated left-to-right to keep every factor nImp x L.
        arma::cx_mat B = Kip0.rows(imp_pos);                // M (nImp x L)
        B.each_row() %= d.st();                             // M * diag(exp_ih)
        B = B * rotS.t();                                   // nImp x L
        B = B * fb.rot;                                     // nImp x L
        arma::cx_mat D = Kip0.submat(imp_pos, imp_pos);     // nImp x nImp
        return A.t()*B + B.t()*A - A.t()*(D*A);
    }

    /// Reference O(L^3) full conjugation. Numerically identical to buildK();
    /// kept only for validation tests.
    arma::cx_mat buildK_reference() const
    {
        int L = fb.length();
        arma::cx_mat exp_ih(L, L, arma::fill::eye);
        if (nIter > 0)
            exp_ih.submat(bath_pos,bath_pos) = expIH<cmpx>(Kbath * (static_cast<double>(nIter)*dt));
        arma::cx_mat rot = exp_ih * rotS.t() * fb.rot;
        return rot.t() * Kip0 * rot;
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ applyPlan(fb.planRepresentative(K,nRef)); }

    /// extract representative orbitals within the active sector
    void extract_representative_final() { applyPlan(fb.planActiveRepresentative(K)); }

    void applyPlan(OrbitalUpdate<cmpx> const& update)
    {
        update.applyAsBasis(K);
        Fb_mps_spin<cmpx>::ensure_reflection_mat(K);
        fb.applyUpdate(update);
    }

    void doTdvp(TdvpParam args={})
    {
        auto [a,b]=fb.interval_active_full();
        auto mpo=fullHamiltonian(a,b);

        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;

        if (args.epsilonM != 0)
        {
            std::vector<double> epsilonK(args.nKrylov,args.epsilonK);
            itensor::addBasis(fb.psi,mpo,epsilonK,
                              {"Cutoff", args.epsilonM,
                               "Method", "DensityMatrix",
                               "KrylovOrd", args.nKrylov,
                               "DoNormalize", true,
                               "Quiet", true,
                               "Silent", true});
        }

        energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,
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
        applyPlan(fb.planNaturalOrbitals(fb.cc));
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

    /// return the mpo of the Hamiltoninan given by himp and the kinetic energy kin
    itensor::MPO fullHamiltonian(int a,int b) const
    {
        itensor::AutoMPO h(fb.sites);
        int L = param.length();
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (std::abs(param.Umat(i,j)) > 1e-15)
                    h += param.Umat(i,j), "N", i+1, "N", j+1;

        for(auto i=a; i<b; i++)
            for(auto j=a; j<b; j++)
                if (std::abs(K(i,j))>fb.tol)
                    h += K(i,j),"Cdag",i+1,"C",j+1;

        return itensor::toMPO(h);
    }
};

} // namespace fbr

#endif // FBR_DYN_SPIN_H
