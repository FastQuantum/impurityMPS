#ifndef IMPURITY_GS_H
#define IMPURITY_GS_H

#include "fermionic.h"
#include "impurity_param.h"
#include "fb_mps.h"

struct Impurity_gs {
    ImpurityParam param;

    /// these quantities are updated during the iterations
    Fb_mps<double> fb;
    arma::mat K;
    double energy=-1000;

    Impurity_gs(Impurity const& imp, Fb_mps<double> const& fb_)
        : param(imp.param)
        , fb { fb_ }
        , K(param.Kmat)
    {}

    void iterate(DmrgParam args={})
    {
        itensor::cpu_time t0;
        extract_representative(0);
        extract_representative(1);
        // fb.print_bond_dims("after repr");
        std::cout<<"representatives "<<t0.sincemark().wall; t0.mark();
        doDmrg(args);
        // fb.print_bond_dims("after dmrg");
        std::cout<<" dmrg "<<t0.sincemark().wall; t0.mark();
        rotateToNaturalOrbitals();
        // fb.print_bond_dims("after nat orb");
        std::cout<<" NatOrb "<<t0.sincemark().wall<<"\n";

        // for(auto i=0;i<10;i++) doDmrg(args);
    }

    void iterate2(DmrgParam args={})
    {
        itensor::cpu_time t0;
        for(int i=0; ; i+=1) {
            bool r0=extract_representative2(i,0);
            bool r1=extract_representative2(i,1);
            if (!r0 && !r1) break;
            doDmrg(args);
            // std::cout<<i<<" "<< itensor::maxLinkDim(fb.psi)<<" "<<fb.nActive<<" "<<t0.sincemark().wall; t0.mark();
            rotateToNaturalOrbitals();
            // std::cout<<" "<<t0.sincemark().wall<<"\n"; std::cout.flush(); t0.mark();
        }
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef,fb.nActive); }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    bool extract_representative2(int sv_index,int nRef){ return fb.extract_representative(sv_index,K,nRef,fb.nActive); }

    void doDmrg(DmrgParam args={})
    {
        auto mpo=fullHamiltonian( K.submat(0,0,fb.nActive-1,fb.nActive-1) );
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"MaxSite",fb.nActive,"Quiet", true, "Silent", true});
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
    itensor::MPO fullHamiltonian(arma::mat const& kin) const
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


#endif // IMPURITY_GS_H
