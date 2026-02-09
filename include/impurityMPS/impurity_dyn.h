#include "fermionic.h"
#include "impurity_param.h"
#include "fb_mps.h"

#include "tdvp.h"
#include "basisextension.h"

struct Impurity_dyn {
    ImpurityParam param;
    double dt;
    arma::cx_mat exp_ih;
    arma::cx_mat Kip0;
    int nChannel;            ///< the number of channels that connect the impurity with the bath

    /// these quantities are updated during the iterations
    Fb_mps<cmpx> fb;        ///< the current few body MPS
    arma::cx_mat K;         ///< the current Hamiltonian
    // arma::cx_mat Kip;       ///< the current Hamiltonian in the interaction picture of the bath
    double energy=-1000;

    explicit Impurity_dyn(Impurity const& imp, Fb_mps<cmpx> const& fb_, double dt_=0.1)
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
        // rot.t()*K*rot
        int nImp=param.nImp();
        K=Kip0;
        K.rows(0,nImp-1)=K.rows(0,nImp-1).eval()*fb.rot;
        K.cols(0,nImp-1)=fb.rot.t()*K.cols(0,nImp-1).eval();
        fb.rot=this->exp_ih*fb.rot;   // update of the interaction picture

        arma::abs(K).eval().clean(1e-15).print("K rotated");
        arma::imag(K).eval().clean(1e-15).print("K.i");
        arma::real(fb.rot).eval().clean(1e-15).print("rot ip");
        arma::imag(fb.rot).eval().clean(1e-15).print("rot.i");

        double nref=fb.cc(fb.nActive, fb.nActive).real();
        int occ= nref+0.5;
        extract_representative(occ);
        extract_representative(1-occ);

        arma::abs(K).eval().clean(1e-15).print("K f0 f1");
        arma::imag(K).eval().clean(1e-15).print("K.i");
        arma::real(fb.rot).eval().clean(1e-15).print("rot f0 f1");
        arma::imag(fb.rot).eval().clean(1e-15).print("rot.i");

        extract_representative_final();

        arma::real(K).eval().clean(1e-15).print("K f2");
        arma::imag(K).eval().clean(1e-15).print("K.i");
        arma::real(fb.rot).eval().clean(1e-15).print("rot f2");
        arma::imag(fb.rot).eval().clean(1e-15).print("rot.i");

        arma::vec ni=fb.occupations_ni2(); //arma::real(cc.diag());
        ni.print("ni before tdvp");

        evolve();
        // doTdvp(args);

        ni=fb.occupations_ni2(); //arma::real(cc.diag());
        ni.print("ni before NOrb");

        rotateToNaturalOrbitals();

        arma::real(fb.rot).eval().clean(1e-15).print("rot after NOrb");
        arma::imag(fb.rot).eval().clean(1e-15).print("rot.i");

        ni=fb.occupations_ni2(); //arma::real(cc.diag());
        ni.print("ni after NOrb");
    }

    /// extract representative orbital of the sites with ni=nRef where nRef can be 0 or 1
    void extract_representative(int nRef){ fb.extract_representative(K,nRef,param.nImp()); }

    /// extract representative orbitals within the active sector
    void extract_representative_final() { fb.extract_representative_final(K, param.nImp(), fb.nActive); }

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
