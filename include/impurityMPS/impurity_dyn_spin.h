#include "fermionic.h"
#include "impurity_param_spin.h"
#include "impurity_spin_init.h"
#include "fb_mps_spin.h"

#include "tdvp.h"
#include "basisextension.h"

struct Impurity_dyn_spin {
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

    explicit Impurity_dyn_spin(ImpuritySpin const& imp, Fb_mps_spin<cmpx> const& fb_, double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , fb { fb_ }
    {
        int L=param.length();

        {
            using namespace arma;

            auto pos_i = param.impPos();
            imp_pos  = arma::conv_to<arma::uvec>::from(pos_i);
            bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L, pos_i));

            mat Kstar=param.Kmat;

            cx_mat Kbath_full(L,L,arma::fill::zeros);
            Kbath_full(bath_pos,bath_pos) = Kstar(bath_pos,bath_pos) * cmpx(1,0);
            cx_mat commutator = Kstar*Kbath_full - Kbath_full*Kstar;
            Kip0 = Kstar - Kbath_full - cmpx(0,0.5*dt)*commutator;

            rotS = fb.rot;

            arma::cx_mat K0 = param.Kmat * cmpx(1,0);
            this->Kbath=K0.submat(bath_pos,bath_pos);
        }
    }

    /// Convenience constructor from ImpuritySpinInit (arbitrary-ordering input)
    explicit Impurity_dyn_spin(ImpuritySpinInit const& init, double dt_=0.1)
        : Impurity_dyn_spin(init.model, init.fb, dt_) {}

    void iterate(TdvpParam args={})
    {
        arma::cx_mat exp_ih;
        {  // interaction picture update
            int L=fb.length();
            exp_ih=arma::cx_mat(L,L,arma::fill::eye);
            exp_ih.submat(bath_pos,bath_pos)=expIH<cmpx>(Kbath * nIter * dt);

            arma::cx_mat rot = exp_ih * rotS.t() * fb.rot;
            K = rot.t() * Kip0 * rot;
            nIter++;
        }


        extract_representative(0);
        extract_representative(1);
        extract_representative_final();
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
            std::vector<double> epsilonK(args.nKrylov,1E-8);
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
                if (std::abs(param.Umat(i,j))>1e-15) {
                    h += param.Umat(i,j), "N", ii+1, "N", jj+1;
                }
            }

        for(auto i=a; i<b; i++)
            for(auto j=a; j<b; j++)
                if (std::abs(K(i,j))>fb.tol)
                    h += K(i,j),"Cdag",i+1,"C",j+1;

        return itensor::toMPO(h);
    }
};
