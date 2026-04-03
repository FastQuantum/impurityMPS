#include "fermionic.h"
#include "impurity_param_spin.h"
#include "fb_mps_spin.h"

#include "tdvp.h"
#include "basisextension.h"

struct Impurity_dyn {
    ImpurityParamSpin param;
    double dt;
    arma::cx_mat exp_ih;
    arma::cx_mat Kip0;
    int nChannel;            ///< the number of channels that connect the impurity with the bath

    /// these quantities are updated during the iterations
    Fb_mps_spin<cmpx> fb;        ///< the current few body MPS
    arma::cx_mat K;         ///< the current Hamiltonian
    // arma::cx_mat Kip;       ///< the current Hamiltonian in the interaction picture of the bath
    double energy=-1000;        // TODO remove energy (or compute it)

    explicit Impurity_dyn(ImpuritySpin const& imp, Fb_mps_spin<cmpx> const& fb_, double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , fb { fb_ }
    {
        int nImp=param.nImp();
        int L=param.length();
        arma::cx_mat K0 = fb.rot * param.Kmat * fb.rot.t(); // original Hamiltonian matrix
        exp_ih = arma::cx_mat(L,L, arma::fill::eye);
        exp_ih.submat(nImp,nImp, L-1,L-1)=expIH<cmpx>(K0.submat(nImp,nImp, L-1,L-1) * dt);
        arma::vec s = arma::svd(param.Kmat.submat(0, nImp, nImp-1, L-1));
        nChannel = arma::find(s>fb.tol*s[0]).eval().size();

        // interaction picture
        arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, L-1) *
                          K0.submat(nImp,nImp,L-1, L-1) * arma::cx_double(0,-0.5*dt); //the commutator
        Kip0=K0;
        Kip0.submat(nImp, nImp, L-1, L-1).fill(0.0);
        Kip0.submat(0, nImp, nImp-1, L-1)+=K1;
        Kip0.submat(nImp, 0, L-1, nImp-1)+=K1.t();        
    }

    void iterate(TdvpParam args={})
    {
        // rotate from scratch
        int nImp=param.nImp();
        K=Kip0;
        K.rows(0,nImp-1)=K.rows(0,nImp-1).eval()*fb.rot;
        K.cols(0,nImp-1)=fb.rot.t()*K.cols(0,nImp-1).eval();
        fb.rot=this->exp_ih*fb.rot;   // update of the interaction picture

        extract_representative(0);
        extract_representative(1);
        extract_representative_final();
        // evolve();
        doTdvp(args);
        rotateToNaturalOrbitals();
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef, /*use_active=*/false); }

    /// extract representative orbitals within the active sector
    void extract_representative_final() { fb.extract_representative_final(K); }

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
                               {"MaxSite",localL,
                                "Truncate", true,
                                "DoNormalize", true,
                                "Quiet", true,
                                "Silent", true,
                                "NumCenter", 2,
                                "ErrGoal", args.err_goal});
        //fb.psi.orthogonalize({"Cutoff",fb.tol});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    void rotateToNaturalOrbitals()
    {
        auto [a,b]=fb.interval_active_full(); // the interval will change
        auto rot1=fb.rotateToNaturalOrbitals();
        K.cols(a,b-1)=K.cols(a,b-1).eval()*rot1;
        K.rows(a,b-1)=rot1.t()*K.rows(a,b-1).eval();
    }

    /// return the mpo of the Hamiltoninan given by himp and the kinetic energy kin
    itensor::MPO fullHamiltonian(int a,int b) const
    {
        itensor::AutoMPO h(fb.sites);
        auto impPos=param.impPos();
        for(auto i=0; i<param.nImp(); i++)
            for(auto j=0; j<param.nImp(); j++) {
                int ii=impPos[i];
                int jj=impPos[j];
                if (std::abs(param.Umat(i,j))>1e-15)
                    h += param.Umat(i,j), "N", ii+1, "N", jj+1;
            }

        for(auto i=a; i<b; i++)
            for(auto j=a; j<b; j++)
                if (std::abs(K(i,j))>fb.tol)
                    h += K(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }
};
