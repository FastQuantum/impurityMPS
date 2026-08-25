#ifndef FBR_DYN_H
#define FBR_DYN_H

#include "graph.h"
#include "itensor_utils.h"
#include "impurity_param.h"
#include "fb_mps.h"

#include "tdvp.h"
#include "basisextension.h"

namespace fbr {

struct Fbr_dyn {
    ImpurityParam param;
    double dt;
    arma::cx_mat exp_ih;
    arma::cx_mat Kip0;
    int nChannel;            ///< the number of channels that connect the impurity with the bath
    arma::uvec imp_pos;      ///< original impurity positions
    arma::uvec bath_pos;     ///< orignal bath positions

    /// these quantities are updated during the iterations
    Fb_mps<cmpx> fb;        ///< the current few body MPS
    arma::cx_mat K;         ///< the current Hamiltonian
    double energy=-1000;

    explicit Fbr_dyn(Impurity const& imp, Fb_mps<cmpx> const& fb_, double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , fb { fb_ }
    {
        int L=param.length();
        arma::cx_mat K0 = fb.rot * param.Kmat * fb.rot.t(); // original Hamiltonian matrix
        imp_pos = arma::conv_to<arma::uvec>::from(param.impPos);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L,param.impPos));

        exp_ih = arma::cx_mat(L,L, arma::fill::eye);
        exp_ih.submat(bath_pos,bath_pos)=expIH<cmpx>(K0.submat(bath_pos,bath_pos) * dt);
        arma::vec s = arma::svd(param.Kmat.submat(imp_pos,bath_pos));
        nChannel = arma::find(s>fb.tol*s[0]).eval().size();

        // interaction picture
        arma::cx_mat K1 = K0.submat(imp_pos,bath_pos) *
                          K0.submat(bath_pos,bath_pos) * arma::cx_double(0,-0.5*dt); //the commutator
        Kip0=K0;
        Kip0.submat(bath_pos,bath_pos).fill(0.0);
        Kip0.submat(imp_pos,bath_pos)+=K1;
        Kip0.submat(bath_pos,imp_pos)+=K1.t();

        K=Kip0;
        K.rows(imp_pos)=K.rows(imp_pos).eval()*fb.rot;
        K.cols(imp_pos)=fb.rot.t()*K.cols(imp_pos).eval();
    }

    void iterate(TdvpParam args={})
    {
        // rotate from scratch
        K=Kip0;
        K.rows(imp_pos)=K.rows(imp_pos).eval()*fb.rot;
        K.cols(imp_pos)=fb.rot.t()*K.cols(imp_pos).eval();
        fb.rot=this->exp_ih*fb.rot;   // update of the interaction picture

        applyPlan(fb.planRepresentative(K,0));
        applyPlan(fb.planRepresentative(K,1));
        applyPlan(fb.planActiveRepresentative(K));
        doTdvp(args);
        applyPlan(fb.planNaturalOrbitals(fb.cc));
    }

    void applyPlan(OrbitalUpdate<cmpx> const& update)
    {
        update.applyAsBasis(K);
        fb.applyUpdate(update);
    }

    void doTdvp(TdvpParam args={})
    {
        int localL=param.nImp()+nChannel;
        auto mpo=fullHamiltonian( K.submat(0, 0, localL-1, localL-1) ); //TODO: fix this
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;

        if (args.epsilonM != 0)
        {
            std::vector<double> epsilonK(args.nKrylov,args.epsilonK);  // Global subspace expansion
            itensor::addBasis(fb.psi,mpo,epsilonK,
                              {"Cutoff", args.epsilonM,
                               "Method", "DensityMatrix",
                               "KrylovOrd", args.nKrylov,
                               "DoNormalize", true,
                               "Quiet", true,
                               "Silent", true});
        }

        energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,          // TDVP sweep
                               {"MaxSite",localL,
                                "Truncate", true,
                                "DoNormalize", true,
                                "Quiet", true,
                                "Silent", true,
                                "NumCenter", 2,
                                "ErrGoal", args.err_goal});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    /// Schrödinger-picture real-space <c_i^dag c_j> matrix.
    /// In this spinless dynamics, fb.rot already absorbs the accumulated bath phase
    /// (fb.rot = exp_ih * fb.rot every step), so fb.correlator_all() is directly the
    /// Schrödinger-picture correlator.
    arma::cx_mat correlator_all() const { return fb.correlator_all(); }
    cmpx correlator(int i, int j) const { return fb.correlator(i, j); }
    arma::cx_vec correlator_all_i(int j) const { return fb.correlator_all_i(j); }
    arma::cx_vec correlator_all_j(int i) const { return fb.correlator_all_j(i); }

    /// return the mpo of the Hamiltoninan given by himp and the kinetic energy kin
    itensor::MPO fullHamiltonian(arma::cx_mat const& kin) const
    {
        itensor::AutoMPO h(fb.sites);
        for(auto i=0; i<param.nImp(); i++)
            for(auto j=0; j<param.nImp(); j++)
                if (std::abs(param.Umat(i,j))>1e-15)
                    h += param.Umat(i,j), "N", i+1, "N", j+1;
        for(auto i=0; i<kin.n_rows; i++)
            for(auto j=0; j<kin.n_cols; j++)
                if (std::abs(kin(i,j))>fb.tol)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }
};

} // namespace fbr

#endif // FBR_DYN_H
