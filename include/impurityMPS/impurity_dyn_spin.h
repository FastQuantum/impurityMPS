#include "fermionic.h"
#include "impurity_param_spin.h"
#include "fb_mps_spin.h"

#include "tdvp.h"
#include "basisextension.h"

struct Impurity_dyn_spin {
    ImpurityParamSpin param;
    double dt;
    arma::cx_mat exp_ih;
    arma::cx_mat Kip0;    


    /// these quantities are updated during the iterations
    Fb_mps_spin<cmpx> fb;        ///< the current few body MPS
    arma::cx_mat K;         ///< the current Hamiltonian
    double energy=-1000;        // TODO remove energy (or compute it)

    explicit Impurity_dyn_spin(ImpuritySpin const& imp, Fb_mps_spin<cmpx> const& fb_, double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , fb { fb_ }
    {
        //int nImp=param.nImp();
        int L=param.length();
        arma::uvec imp_pos;
        arma::uvec bath_pos;
        arma::cx_mat K0 = fb.rot * param.Kmat * fb.rot.t(); // original Hamiltonian matrix
        auto pos_i = param.impPos0();
        imp_pos=arma::conv_to<arma::uvec>::from(pos_i);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L,pos_i));
        exp_ih = arma::cx_mat(L,L, arma::fill::eye);
        exp_ih.submat(bath_pos,bath_pos)=expIH<cmpx>(K0.submat(bath_pos,bath_pos) * dt);

        // interaction picture
        arma::cx_mat K1 = K0.submat(imp_pos,bath_pos) *
                          K0.submat(bath_pos,bath_pos) * arma::cx_double(0,-0.5*dt); //the commutator
        Kip0=K0;
        Kip0.submat(bath_pos,bath_pos).fill(0.0);
        Kip0.submat(imp_pos,bath_pos)+=K1;
        Kip0.submat(bath_pos,imp_pos)+=K1.t();

        arma::real(K0*1).eval().clean(1e-11).print("Kip0");
        // arma::real(Kip0*1).eval().clean(1e-11).print("Kip0 after IP");
        arma::real(param.Kmat*1).eval().clean(1e-11).print("param.Kmat");
        arma::real(fb.rot.t()*K0*fb.rot).eval().clean(1e-11).print("Kip0 before IP rotated at constructor (expected = param.Kmat)");
        // arma::real(fb.rot.t()*Kip0*fb.rot).eval().clean(1e-11).print("Kip0 after IP rotated at constructor (expected impurity untouch)");
    }

    void iterate(TdvpParam args={})
    {
        arma::real(fb.rot.t()*Kip0*fb.rot).eval().clean(1e-11).print("Kip0 before repr0 rotated (expected impurity untouch)");

        // rotate from scratch
        K=Kip0;
        // K.rows(imp_pos)=K.rows(imp_pos).eval()*fb.rot;
        // K.cols(imp_pos)=fb.rot.t()*K.cols(imp_pos).eval();
        K=fb.rot.t()*Kip0*fb.rot;
        fb.rot=exp_ih*fb.rot;   // update of the interaction picture

        arma::real(K*1.0).eval().clean(1e-11).print("K before repr 0");
        fb.occupations_ni2().as_row().eval().clean(1e-10).print("ni");

        extract_representative(0);

        arma::real(K*1).eval().clean(1e-11).print("K before repr 1");
        fb.occupations_ni2().as_row().eval().clean(1e-10).print("ni");

        extract_representative(1);
        // extract_representative_final();

        arma::real(K*1).eval().clean(1e-11).print("K before tdvp");
        fb.occupations_ni2().as_row().eval().clean(1e-10).print("ni");

        doTdvp(args);

        arma::real(K*1).eval().clean(1e-11).print("K before nat orb");
        fb.occupations_ni2().as_row().eval().clean(1e-10).print("ni");
        fb.occupations_ni().as_row().eval().clean(1e-10).print("ni from cc");


        rotateToNaturalOrbitals();

        arma::real(K*1).eval().clean(1e-11).print("K after nat orb");
        fb.occupations_ni2().as_row().eval().clean(1e-10).print("ni");

    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef, /*use_active=*/false); }

    /// extract representative orbitals within the active sector
    void extract_representative_final() { fb.extract_representative_final(K); }

    void doTdvp(TdvpParam args={})
    {
        auto [a,b]=fb.interval_active_full();
        auto mpo=fullHamiltonian(a,b);

        // mpo.orthogonalize(); // TODO: what is this ???

        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;

        if (args.epsilonM != 0)
        {
            std::vector<double> epsilonK(args.nKrylov,1E-8);  // Global subspace expansion
            itensor::addBasis(fb.psi,mpo,epsilonK,
                              {/*"Minb",a+1,"MaxSite",b,*/
                               "Cutoff", args.epsilonM,
                               "Method", "DensityMatrix",
                               "KrylovOrd", args.nKrylov,
                               "DoNormalize", true,
                               "Quiet", true,
                               "Silent", true});
        }

        energy = itensor::tdvp(fb.psi,mpo, -imag_1*dt, sweeps,          // TDVP sweep
                               {/*"Minb",a+1,"MaxSite",b,*/
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
                if (std::abs(param.Umat(i,j))>1e-15) {
                    h += param.Umat(i,j), "N", ii+1, "N", jj+1;
                    std::cout<<"ii,jj="<<ii<<" "<<jj<<"\n";
                }
            }


        for(auto i=a; i<b; i++)
            for(auto j=a; j<b; j++)
                if (std::abs(K(i,j))>fb.tol)
                    h += K(i,j),"Cdag",i+1,"C",j+1;

        arma::real(K*1.0).eval().clean(1e-9).eval().print("Keff");
        fb.occupations_ni2().as_row().eval().print("ni");

        return itensor::toMPO(h);
    }
};
