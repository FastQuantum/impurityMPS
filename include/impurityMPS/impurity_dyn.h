#include "fermionic.h"
#include "impurity_param.h"
#include "fb_mps.h"

#include "tdvp.h"
#include "basisextension.h"

struct Impurity_dyn {
    ImpurityParam param;
    double dt;
    arma::cx_mat exp_ih;
    int nChannel;            ///< the number of channels that connect the impurity with the bath

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
        arma::vec s = arma::svd(K.submat(0, nImp, nImp-1, K.n_cols-1));
        nChannel = arma::find(s>fb.tol*s[0]).eval().size();
    }

    void iterate(TdvpParam args={})
    {
        rotateIntPicture();
        extract_representative(0);
        extract_representative(1);
        extract_representative_final();
        evolve(); //doTdvp(args);
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

    void extract_representative_final()
    {
        int L=param.length();
        int nImp=param.nImp();
        int p0=fb.nActive-1;                 // the position before Slater starts
        int p1=std::min(L-1,p0+2*nChannel);  // the position of the last representative
        auto k12=Kip.submat(0,nImp,nImp-1,p1);
        arma::vec s;
        arma::Mat<cmpx> U, V;
        svd_econ(U,s,V,k12);
        int nSv=arma::find(s>fb.tol*s[0]).eval().size();  // it should be nSv==nChannel
        //std::cout<<"nSV="<<nSv<<std::endl;
        auto givens=GivensRotForRot_left(arma::conj(V.head_cols(nSv)).eval());
        for(auto& g:givens) g.b+=nImp;
        arma::cx_mat rot1=matrot_from_Givens(givens, k12.n_cols+nImp).st();
        Kip.cols(0,p1)=Kip.cols(0,p1).eval()*rot1;
        Kip.rows(0,p1)=rot1.t()*Kip.rows(0,p1).eval();
        fb.rot.cols(0,p1)=fb.rot.cols(0,p1)*rot1;

        auto gates=Fermionic::NOGates(fb.sites,givens);
        gateTEvol(gates,1,1,fb.psi,{"Cutoff",fb.tol,"Quiet",true, "Normalize",false,"ShowPercent",false});
        fb.update_cc();
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
        auto gates=TrotterGatesExp(Kip,3,dt);
        gateTEvol(gates,1,1,fb.psi,{"Cutoff=",fb.tol,"Quiet=",true, "Normalize",false,"ShowPercent",false});
    }

    void doTdvp(TdvpParam args={})
    {
        int localL=fb.nActive; //param.nImp()+nChannel;
        auto mpo=fullHamiltonian( Kip.submat(0, 0, localL-1, localL-1) ); //TODO: fix this
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
                               {"MaxSite",fb.nActive,
                                "Truncate", true,
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
