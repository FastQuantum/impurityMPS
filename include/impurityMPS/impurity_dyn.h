#include "fermionic.h"
#include "impurity_param.h"
#include "fb_mps.h"

#include "tdvp.h"
#include "basisextension.h"

struct Impurity_dyn {
    ImpurityParam param;
    double dt;
    arma::cx_mat exp_ih;

    /// these quantities are updated during the iterations
    Fb_mps<cmpx> fb;        ///< the current few body MPS
    arma::cx_mat K;         ///< the current Hamiltonian
    arma::cx_mat Kip;       ///< the current Hamiltonian in the interaction picture of the bath
    double energy=-1000;

    explicit Impurity_dyn(Impurity const& imp, Fb_mps<cmpx> const& fb_, double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , fb { fb_ }
        , K { param.Kmat, arma::zeros(arma::size(param.Kmat)) } // convert to complex
    {
        int nImp=param.nImp();
        exp_ih = arma::cx_mat(size(K), arma::fill::eye);
        exp_ih.submat(nImp,nImp, K.n_rows-1,K.n_rows-1)=expIH<cmpx>(K.submat(nImp,nImp, K.n_rows-1,K.n_rows-1) * dt);
    }

    void iterate(TdvpParam args={})
    {
        rotateIntPicture();
        extract_representative(0);
        extract_representative(1);
        doTdvp(args);
        rotateToNaturalOrbitals();
    }

    void rotateIntPicture()
    {
        int L=K.n_cols;
        int nImp=param.nImp();

        const auto& K0=K;
        arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, L-1) *
                K0.submat(nImp,nImp,L-1, L-1) * arma::cx_double(0,-0.5*dt); //the commutator
        Kip=K0;
        Kip.submat(nImp, nImp, L-1, L-1).fill(0.0);
        Kip.submat(0, nImp, nImp-1, L-1)+=K1;
        Kip.submat(nImp, 0, L-1, nImp-1)+=K1.t();

        // rot.t()*Kip*rot
        Kip.rows(0,nImp-1)=Kip.rows(0,nImp-1).eval()*fb.rot;
        Kip.cols(0,nImp-1)=fb.rot.t()*Kip.cols(0,nImp-1).eval();

        fb.rot=exp_ih*fb.rot;
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(Kip,nRef); }

    void doTdvp(TdvpParam args={})
    {
        auto mpo=fullHamiltonian( Kip.submat(0,0,fb.nActive-1,fb.nActive-1) );
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;

        if (args.epsilonM != 0)
        {
            std::vector<double> epsilonK(args.nKrylov,1E-8);  // Global subspace expansion
            itensor::addBasis(fb.psi,mpo,epsilonK,
                              {"Cutoff", args.epsilonM,
                               "Method", "DensityMatrix",
                               "KrylovOrd", args.nKrylov,
                               "DoNormalize", true,
                               "Quiet", true,
                               "Silent", true});
        }

        energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,          // TDVP sweep
                               {"Truncate", true,
                                "DoNormalize", false,
                                "Quiet", true,
                                "Silent", true,
                                "NumCenter", 2,
                                "ErrGoal", args.err_goal});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    void rotateToNaturalOrbitals()
    {
        int nA=fb.nActive; // it will change
        auto rot1=fb.rotateToNaturalOrbitals(param.nImp());
        K.cols(0,nA-1)=K.cols(0,nA-1).eval()*rot1;
        K.rows(0,nA-1)=rot1.t()*K.rows(0,nA-1).eval();
    }

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
