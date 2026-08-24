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
    // arma::cx_mat Kip;       ///< the current Hamiltonian in the interaction picture of the bath
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

        if (fb.nActive+2*nChannel < param.length()) {
            double nref=fb.cc(fb.nActive, fb.nActive).real();
            int occ= nref+0.5;
            applyPlan(fb.planRepresentative(K,occ));
            applyPlan(fb.planRepresentative(K,1-occ));
        }
        else fb.nActive=param.length();

        applyPlan(fb.planActiveRepresentative(K));

        // evolve();
        doTdvp(args);

        applyPlan(fb.planNaturalOrbitals(fb.cc));
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef) { applyPlan(fb.planRepresentative(K,nRef)); }

    /// extract representative orbitals within the active sector
    void extract_representative_final() { applyPlan(fb.planActiveRepresentative(K)); }

    void applyPlan(OrbitalUpdate<cmpx> const& update)
    {
        update.applyAsBasis(K);
        fb.applyUpdate(update);
    }

    template<class T>
    auto TrotterGatesExp(arma::Mat<T> const& Kip,int nTB,double dt) const
    {
        using namespace itensor;
        using namespace arma;

        mat22 Id(fill::eye),
                N={{0,0},{0,1}},
                C={{0,1},{0,0}},
                Cdag=C.t();

        auto to_itgate=[&](int i,cx_mat44 const& rot) {
            int b=i+1;
            auto s1 = itensor::dag(fb.sites(b));
            auto s2 = itensor::dag(fb.sites(b+1));
            auto s1p = prime(fb.sites(b));
            auto s2p = prime(fb.sites(b+1));
            itensor::ITensor hterm(s1,s2,s1p,s2p);
            hterm.set(s1(1),s2(1),s1p(1),s2p(1), rot(0,0));
            hterm.set(s1(2),s2(2),s1p(2),s2p(2), rot(3,3));
            hterm.set(s1(2),s2(1),s1p(2),s2p(1), rot(1,1));
            hterm.set(s1(2),s2(1),s1p(1),s2p(2), rot(1,2));
            hterm.set(s1(1),s2(2),s1p(2),s2p(1), rot(2,1));
            hterm.set(s1(1),s2(2),s1p(1),s2p(2), rot(2,2));
            return BondGate(fb.sites,b,b+1,hterm);
        };

        auto mykron=[](mat22 const& A,mat22 const& B) { return mat44 {kron(B,A).st()}; };

        auto gates = std::vector<BondGate>();

        auto U=param.Umat(0,1);
        //Create the gates exp(-i*tstep/2*hterm)
        for(int i=0; i<nTB-1; ++i)
        {
            cx_mat44 hloc = Kip(i,i+1)*mykron(Cdag,C);
            hloc += Kip(i+1,i)*mykron(C,Cdag);
            hloc += Kip(i,i)*mykron(N,Id);
            if (i==nTB-2) hloc += Kip(i+1,i+1)*mykron(Id,N);
            if (i==0) hloc += T(U)*mykron(N,N);

            cx_mat44 rot=expIH<T>(hloc * (0.5*dt));
            gates.push_back(to_itgate(i,rot));
        }
        //Create the gates exp(-i*tstep/2*hterm) in reverse
        for(int i = nTB-2; i>=0; --i)
        {
            cx_mat44 hloc = Kip(i,i+1)*mykron(Cdag,C);
            hloc += Kip(i+1,i)*mykron(C,Cdag);
            hloc += Kip(i,i)*mykron(N,Id);
            if (i==nTB-2) hloc += Kip(i+1,i+1)*mykron(Id,N);
            if (i==0) hloc += T(U)*mykron(N,N);

            cx_mat44 rot=expIH<T>(hloc * (0.5*dt));
            gates.push_back(to_itgate(i,rot));
        }
        return gates;
    }

    void evolve()
    {
        auto gates=TrotterGatesExp(K,3,dt);
        gateTEvol(gates,1,1,fb.psi,{"Cutoff=",fb.tol,"Quiet=",true, "Normalize",false,"ShowPercent",false});
        fb.update_cc();
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
        //fb.psi.orthogonalize({"Cutoff",fb.tol});
        energy += fb.SlaterEnergy(K);
        fb.update_cc();
    }

    void rotateToNaturalOrbitals()
    {
        applyPlan(fb.planNaturalOrbitals(fb.cc));
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
