#ifndef IMPURITY_GS_SPIN_H
#define IMPURITY_GS_SPIN_H

#include "fermionic.h"
#include "impurity_param_spin.h"
#include "fb_mps_spin.h"

struct Impurity_gs_spin {
    ImpurityParamSpin param;

    /// these quantities are updated during the iterations
    Fb_mps_spin<double> fb;
    arma::mat K;
    double energy=-1000;

    Impurity_gs_spin(ImpuritySpin const& imp, Fb_mps_spin<double> const& fb_)
        : param(imp.param)
        , fb { fb_ }
        , K(param.Kmat)
    {}

    void iterate(DmrgParam args={})
    {
        itensor::cpu_time t0;
        std::cout<<fb.p2-fb.p1<<" nActive\n";
        fb.occupations_ni2().as_row().eval().print("before repr");
        fb.occupations_ni().as_row().eval().print("before repr cc");
        extract_representative(0);
        extract_representative(1);
        std::cout<<fb.p2-fb.p1<<" nActive afte repr\n";
        // std::cout<<"representatives "<<t0.sincemark().wall; t0.mark();

        fb.occupations_ni2().as_row().eval().print("before dmrg");
        fb.occupations_ni().as_row().eval().print("before repr cc");
        doDmrg(args);
        // fb.print_bond_dims("after dmrg");
        // std::cout<<" dmrg "<<t0.sincemark().wall; t0.mark();
        fb.occupations_ni2().as_row().eval().print("before nat orb");
        fb.occupations_ni().as_row().eval().print("before repr cc");
        rotateToNaturalOrbitals();
        // fb.print_bond_dims("after nat orb");
        // std::cout<<" NatOrb "<<t0.sincemark().wall<<"\n";

        // for(auto i=0;i<10;i++) doDmrg(args);
    }

    // void iterate2(DmrgParam args={})
    // {
    //     itensor::cpu_time t0;
    //     for(int i=0; ; i+=1) {
    //         bool r0=extract_representative2(i,0);
    //         bool r1=extract_representative2(i,1);
    //         if (!r0 && !r1) break;
    //         doDmrg(args);
    //         // std::cout<<i<<" "<< itensor::maxLinkDim(fb.psi)<<" "<<fb.nActive<<" "<<t0.sincemark().wall; t0.mark();
    //         rotateToNaturalOrbitals();
    //         // std::cout<<" "<<t0.sincemark().wall<<"\n"; std::cout.flush(); t0.mark();
    //     }
    // }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef,true); }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    // bool extract_representative2(int sv_index,int nRef){ return fb.extract_representative(sv_index,K,nRef,fb.nActive); }

    void doDmrg(DmrgParam args={})
    {
        auto [a,b]=fb.interval_active_full();
        auto mpo=fullHamiltonian( K.submat(a,a,b-1,b-1), a);
        auto sweeps = itensor::Sweeps(1);
        sweeps.maxdim() = args.max_bond_dim;
        sweeps.cutoff() = fb.tol;
        sweeps.niter() = args.nIter_diag;
        sweeps.noise() = args.noise;
        fb.psi.position(a+1);
        energy=itensor::dmrg(fb.psi,mpo,sweeps, {"Minb",a+1,"MaxSite",b,"Quiet", true, "Silent", true});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    void rotateToNaturalOrbitals()
    {
        auto [a,b]=fb.interval_active_full(); // it will change
        auto rot1=fb.rotateToNaturalOrbitals();
        K.cols(a,b-1)=K.cols(a,b-1).eval()*rot1;
        K.rows(a,b-1)=rot1.t()*K.rows(a,b-1).eval();
    }

    /// return the mpo of the Hamiltoninan given by himp and the kinetic energy kin
    itensor::MPO fullHamiltonian(arma::mat const& kin, int a) const
    {
        kin.print("kin at fullH");
        itensor::AutoMPO h(fb.sites);
        for(auto i=0; i<param.nImp(); i++)
            for(auto j=0; j<param.nImp(); j++) {
                int ii=param.impPos[i];
                int jj=param.impPos[j];
                if (std::abs(param.Umat(i,j))>1e-15)
                    h += param.Umat(i,j), "N", ii+1, "N", jj+1;
            }
        for(auto i=0; i<kin.n_rows; i++)
            for(auto j=0; j<kin.n_cols; j++)
                if (std::abs(kin(i,j))>fb.tol)
                    h += kin(i,j),"Cdag",a+i+1,"C",a+j+1;
        return itensor::toMPO(h);
    }
};


#endif // IMPURITY_GS_SPIN_H
