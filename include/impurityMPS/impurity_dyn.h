#include "fermionic.h"
#include "impurity_param.h"
#include "fb_mps.h"

struct Impurity_dyn {
    ImpurityParam param;
    double dt=0.1;

    /// these quantities are updated during the iterations
    Fb_mps<cmpx> fb;
    arma::cx_mat K;
    double energy=-1000;

    explicit Impurity_dyn(Impurity const& imp, Fb_mps<cmpx> const& fb_, double dt_)
        : param(imp.param)
        , dt(dt_)
        , fb { fb_ }
        , K { param.Kmat, arma::zeros(arma::size(param.Kmat)) }
    {}

    void iterate(DmrgParam args={})
    {
        extract_representative(0);
        extract_representative(1);
        doTdvp(args);
        rotateToNaturalOrbitals();
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef); }

    void doTdvp(DmrgParam args={})
    {
        auto mpo=fullHamiltonian( K.submat(0,0,fb.nActive-1,fb.nActive-1) );
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"MaxSite",fb.nActive,"Quiet", true, "Silent", true});
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
