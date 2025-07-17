#ifndef IRLM_H
#define IRLM_H


#include "fermionic.h"
#include <tuple>

struct IRLM {
    int L=20;
    double t=0.5;
    double V=0.15;
    double U=-0.5;
    double ed=0;
    bool connected=true;

    auto matrices() const
    {
        // Kinetic energy TB Hamiltonian
        arma::mat K(L,L,arma::fill::zeros);
        for(auto i=1+!connected; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=t;
        K(0,1)=K(1,0)=V;
        K(0,0)=ed;

        // U ni nj
        arma::mat Umat(L,L,arma::fill::zeros);
        Umat(0,1)=U;
        K(1,1)=-U/2;
        K(0,0)+=-U/2;
        return std::make_pair(K,Umat);
    }

    auto rotStar() const
    {
        auto [K, _]=matrices();
        // Diagonalize the bath
        arma::vec ek;
        arma::mat R;
        arma::eig_sym( ek, R, K.submat(2,2,L-1,L-1) );
        arma::uvec iek=arma::sort_index( arma::abs(ek) );
        arma::mat Rfull(L,L,arma::fill::eye);
        arma::mat Rr=R.cols(iek);
        Rfull.submat(2,2,L-1,L-1)=Rr;
        return Rfull;
    }

    /// the Hamiltonian setting the kinetic(inactive,inactive)=0
    HamSys HamRestricted(arma::mat const& rot={}, bool rotateOnlyKin=false, arma::uvec iInactive={}) const
    {
        arma::mat K,Umat;
        std::tie(K,Umat)=matrices();
        Fermionic sys=[&]() {
            if (rot.empty()) return Fermionic(K,Umat);
            if (rotateOnlyKin) return Fermionic(rot.t()*K*rot, Umat);
            return Fermionic(K,Umat,rot);
        }();
        sys.Kmat.submat(iInactive,iInactive).fill(0);
        auto hsys=sys.Ham();
        hsys.hamEnrich=hsys.ham;
        return hsys;
    }

    HamSys Ham(arma::mat const& rot={}, bool rotateOnlyKin=false, arma::uvec iInactive={}) const
    {
        arma::mat K,Umat;
        std::tie(K,Umat)=matrices();
        Fermionic sys=[&]() {
            if (rot.empty()) return Fermionic(K,Umat);
            if (rotateOnlyKin) return Fermionic(rot.t()*K*rot, Umat);
            return Fermionic(K,Umat,rot);
        }();
        HamSys hsys=sys.Ham();
        hsys.hamEnrich=hsys.ham;
        if (iInactive.empty()) return hsys;
        sys.Kmat.submat(iInactive,iInactive).fill(0);
        hsys.hamEnrich=sys.Ham().ham;
        return hsys;
    }

    HamSysV HamV(arma::mat const& rot={}, bool rotateOnlyKin=false, arma::uvec iInactive={}) const
    {
        arma::mat K,Umat;
        std::tie(K,Umat)=matrices();
        Fermionic sys=[&]() {
            if (rot.empty()) return Fermionic(K,Umat);
            if (rotateOnlyKin) return Fermionic(rot.t()*K*rot, Umat);
            return Fermionic(K,Umat,rot);
        }();
        HamSysV hsys=sys.HamV();
        if (iInactive.empty()) return hsys;  // normal path
        sys.Kmat.submat(iInactive,iInactive).fill(0);
        hsys.hamEnrich=sys.Ham().ham;
        return hsys;
    }

    HamSys HamStar() const { return Ham(rotStar(), true); }

    HamSys HamNO(arma::mat const& cc, bool fromStar=true) const { auto R=Fermionic::rotNO(cc); if (fromStar) R=R*rotStar(); return Ham(R); }

};

struct IRLM_ip {
    IRLM irlm;
    itensor::Fermion sites;
    itensor::AutoMPO hImp;
    arma::mat K;
    arma::cx_mat exp_ih;

    explicit IRLM_ip(itensor::Fermion sites_, const IRLM& irlm_, double dt)
        : irlm(irlm_)
        , sites(sites_)
        , hImp (sites)
    {
        arma::mat Umat;
        std::tie(K,Umat)=irlm.matrices();

        std::vector<int> impPos={0,1};
        for(auto i:impPos)
            for(auto j:impPos)
                if (std::abs(Umat(i,j))>1e-15)
                    hImp += Umat(i,j),"N", i+1,"N", j+1;

        using namespace arma;
        int nImp=impPos.size();
        exp_ih=arma::cx_mat(size(K), fill::eye);
        exp_ih.submat(nImp,nImp, K.n_rows-1,K.n_rows-1)=expIH<double>(K.submat(nImp,nImp, K.n_rows-1,K.n_rows-1) * dt);
    }

    template<class T>
    HamSys Ham(arma::Mat<T> const& rot) const
    {
        arma::cx_mat Kin=rot.t()*K*rot;
        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kin(i,j))>1e-15)
                h += Kin(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);
        return {sites,mpo,mpo};
    }

    template<class T>
    HamSys HamIP(arma::Mat<T> const& rot, int nImp, double dt) const
    {
        if (nImp==rot.n_rows) return Ham(rot);
        arma::Mat<T> K0=rot.t()*K*rot;
        arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, rot.n_rows-1) *
                           K0.submat(nImp,nImp,rot.n_rows-1, rot.n_rows-1) * arma::cx_double(0,-0.5*dt);
        arma::cx_mat Kip=K0 * arma::cx_double(1,0);
        Kip.submat(nImp, nImp, rot.n_rows-1, rot.n_rows-1).fill(0.0);
        Kip.submat(0,0,nImp-1,nImp-1)=K0.submat(0,0,nImp-1,nImp-1) * arma::cx_double(1,0);
        Kip.submat(0, nImp, nImp-1, rot.n_rows-1)+=K1;
        Kip.submat(nImp, 0, rot.n_rows-1, nImp-1)+=K1.t();

        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kip(i,j))>1e-15)
                h += Kip(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);
        return {sites,mpo,mpo};
    }

    /// return exp(-i H2 dt), i.e. the rotation generated after the interaction picture evolution
    template<class T>
    arma::cx_mat rotIP(arma::Mat<T> const& rot, int nImp, double dt) const
    {
        using namespace arma;
        arma::Mat<T> Kin=rot.t()*K*rot;
        arma::cx_mat rotK(size(rot), fill::eye);
        rotK.submat(nImp,nImp, rot.n_rows-1,rot.n_rows-1)=expIH<T>(Kin.submat(nImp,nImp, rot.n_rows-1,rot.n_rows-1) * dt );
        return rotK;
    }

    /// return the HamSys and the list of Givens rotations.
    template<class T>
    std::tuple<HamSys, std::vector<GivensRot<T>>, arma::Mat<T> > HamIP_f(arma::Mat<T> const& rot, int nImp, double dt, bool extractf, double tolSv=1e-12) const
    {
        if (nImp==rot.n_rows) return {Ham(rot),{},{}};
        arma::Mat<T> K0=rot.t()*K*rot;
        arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, rot.n_rows-1) *
                           K0.submat(nImp,nImp,rot.n_rows-1, rot.n_rows-1) * arma::cx_double(0,-0.5*dt); //the commutator
        arma::cx_mat Kip=K0 * arma::cx_double(1,0);
        Kip.submat(nImp, nImp, rot.n_rows-1, rot.n_rows-1).fill(0.0);
        Kip.submat(0,0,nImp-1,nImp-1)=K0.submat(0,0,nImp-1,nImp-1) * arma::cx_double(1,0);
        Kip.submat(0, nImp, nImp-1, rot.n_rows-1)+=K1;
        Kip.submat(nImp, 0, rot.n_rows-1, nImp-1)+=K1.t();

        std::vector<GivensRot<T>> givens;   // TODO: Wannierize the subspace
        if (extractf){// the circuit to extract f orbitals
            auto k12=Kip.submat(0,nImp,nImp-1,rot.n_rows-1);
            arma::vec s;
            arma::Mat<T> U, V;
            svd_econ(U,s,V,arma::conj(k12));
            int nSv=arma::find(s>tolSv*s[0]).eval().size();
            //std::cout<<"nSV="<<nSv<<std::endl;
            givens=GivensRotForRot_left(V.head_cols(nSv).eval());
            for(auto& g:givens) g.b+=nImp;
            arma::Mat<T> rot1=matrot_from_Givens(givens, rot.n_cols).st();
            Kip=(rot1.t()*Kip*rot1).eval();
        }

        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kip(i,j))>1e-15)
                h += Kip(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);
        HamSys ham{sites,mpo,mpo};
        return make_tuple(ham, givens, Kip);
    }

    template<class T>
    struct HamIPOut
    {
        HamSys ham;
        std::vector<GivensRot<T>> givens;
        arma::cx_mat rot;
        arma::cx_mat Kip;
        int from=0;  /// the sites to swap in the Slater part
        int to=0;
        // the fermionic sign det(rot) also affects the state, but it always cancels out (?)
    };

    /// return the HamSys and the list of Givens rotations.
    template<class T>
    HamIPOut<T> HamIP_f3(arma::Mat<T> const& rot, int nImp, arma::vec ni, double dt, bool extractf, double tolSv=1e-9) const
    {
        itensor::cpu_time t0;
        if (nImp==rot.n_rows) return {Ham(rot),{},{}};
        HamIPOut<T> out;

        int L=rot.n_cols;
        arma::cx_mat  Kip; // interaction picture
        {
            const auto& K0=K;
            arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, rot.n_rows-1) *
                               K0.submat(nImp,nImp,rot.n_rows-1, rot.n_rows-1) * arma::cx_double(0,-0.5*dt); //the commutator
            Kip=K0 * arma::cx_double(1,0);
            Kip.submat(nImp, nImp, rot.n_rows-1, rot.n_rows-1).fill(0.0);
            Kip.submat(0,0,nImp-1,nImp-1)=K0.submat(0,0,nImp-1,nImp-1) * arma::cx_double(1,0);
            Kip.submat(0, nImp, nImp-1, rot.n_rows-1)+=K1;
            Kip.submat(nImp, 0, rot.n_rows-1, nImp-1)+=K1.t();

            // rot.t()*K*rot
            Kip.rows(0,nImp-1)=Kip.rows(0,nImp-1).eval()*rot;
            Kip.cols(0,nImp-1)=rot.t()*Kip.cols(0,nImp-1).eval();
        }

        // std::cout<<"Kip:"<<t0.sincemark()<<std::endl;
        t0.mark();

        out.rot=this->exp_ih*rot; //this->rotIP(rot,nImp,dt);
        // std::cout<<"rotIP:"<<t0.sincemark()<<std::endl;
        t0.mark();

        int p0=L-1; {// the position before Slater starts
            for(; p0>=nImp; p0--)
                if (ni[p0]>tolSv && ni[p0]<1-tolSv) break;
        }
        out.from=out.to=p0+2;
        if (p0+2<L) // in fact there is Slater
        {
            { // 1) find the first site where the occupation is not the one at p0+1
                auto isEmpty=[](double x){ return x<0.5; };  // helper function
                out.to = std::find_if(ni.begin()+p0+2, ni.end(),
                                      [&,n0=ni[p0+1]](double x) {return isEmpty(x)!=isEmpty(n0); })
                         - ni.begin();
            }

            // arma::vec(ni).clean(1e-9).raw_print("ni=");

            if (out.from != out.to && out.to!=L){ // 2) swap sites in the Hamiltonian
                std::swap(ni[out.from], ni[out.to]);
                // arma::cx_mat rot1(L, L, arma::fill::eye);
                // rot1.swap_cols(out.from, out.to);
                // Kip=(rot1.t()*Kip*rot1).eval();
                // out.rot=out.rot*rot1;
                Kip.swap_cols(out.from,out.to);
                Kip.swap_rows(out.from,out.to);
                out.rot.swap_cols(out.from,out.to);
            }
            // 3) extract f orbital of the sites with ni=0 and ni=1 independently
            arma::vec nSlater=ni.rows(p0+1,L-1).eval();
            arma::uvec posImp(nImp);
            for(auto i=0; i<nImp; i++) posImp[i]=i;
            // std::cout<<out.from<<" --> "<< out.to<<std::endl; std::cout.flush();
            { // ni==0
                arma::uvec pos0=arma::find(nSlater<0.5)+p0+1;
                if (!pos0.empty()) {
                    // pos0.print("empty");
                    auto k12=Kip.submat(posImp,pos0).eval();
                    arma::vec s;
                    arma::Mat<T> U, V;
                    svd(U,s,V,k12);
                    // arma::cx_mat rot1(L, L, arma::fill::eye);
                    // rot1.cols(pos0)=(rot1.cols(pos0).eval()*V).eval();
                    // Kip=(rot1.t()*Kip*rot1).eval();
                    // out.rot=out.rot*rot1;
                    // Kip.cols(pos0)=Kip.cols(pos0).eval()*V;
                    // Kip.rows(pos0)=V.t()*Kip.rows(pos0).eval();

                    Kip.submat(posImp,pos0).fill(0);
                    Kip.submat(0,pos0[0],nImp-1,pos0[0])=U*s.cols(0,0);
                    Kip.submat(pos0,posImp)=Kip.submat(posImp,pos0).t();
                    out.rot.cols(pos0)=out.rot.cols(pos0)*V;
                    //arma::abs(Kip).eval().clean(1e-6).print("kip empty");
                }
            }
            { // ni==1  TODO: this should add a fermionic sign to the state --> det(V)
                arma::uvec pos0=arma::find(nSlater>0.5)+p0+1;
                if (!pos0.empty()) {
                    // pos0.print("full");
                    auto k12=Kip.submat(posImp,pos0).eval();
                    arma::vec s;
                    arma::Mat<T> U, V;
                    svd(U,s,V,k12);
                    // arma::cx_mat rot1(L, L, arma::fill::eye);
                    // rot1.cols(pos0)=(rot1.cols(pos0).eval()*V).eval();
                    // Kip=(rot1.t()*Kip*rot1).eval();
                    // out.rot=out.rot*rot1;
                    // Kip.cols(pos0)=Kip.cols(pos0).eval()*V;
                    // Kip.rows(pos0)=V.t()*Kip.rows(pos0).eval();

                    Kip.submat(posImp,pos0).fill(0);
                    Kip.submat(0,pos0[0],nImp-1,pos0[0])=U*s.cols(0,0);
                    Kip.submat(pos0,posImp)=Kip.submat(posImp,pos0).t();
                    out.rot.cols(pos0)=out.rot.cols(pos0)*V;
                    //arma::abs(Kip).eval().clean(1e-6).print("kip full");
                }
            }

        }

        // std::cout<<"f0 and f1:"<<t0.sincemark()<<std::endl;
        t0.mark();

        std::vector<GivensRot<T>> givens;
        if (extractf){// the circuit to extract f orbitals
            auto Kip2=Kip.eval();
            // arma::abs(Kip2).eval().clean(1e-6).print("Kip before f");
            int p1=std::min(L-1,p0+2);
            if (out.to==L) p1=p0+1;
            auto k12=Kip.submat(0,nImp,nImp-1,p1);
            arma::vec s;
            arma::Mat<T> U, V;
            svd_econ(U,s,V,k12);
            int nSv=arma::find(s>tolSv*s[0]).eval().size();
            //std::cout<<"nSV="<<nSv<<std::endl;
            givens=GivensRotForRot_left(arma::conj(V.head_cols(nSv)).eval());
            for(auto& g:givens) g.b+=nImp;
            // arma::cx_mat rot1(L, L, arma::fill::eye);
            // rot1.cols(0,p1)=rot1.cols(0,p1).eval() * matrot_from_Givens(givens, k12.n_cols+nImp).st();
            // Kip=(rot1.t()*Kip*rot1).eval();
            // out.rot = out.rot * rot1;
            arma::cx_mat rot1=matrot_from_Givens(givens, k12.n_cols+nImp).st();
            Kip.cols(0,p1)=Kip.cols(0,p1).eval()*rot1;
            Kip.rows(0,p1)=rot1.t()*Kip.rows(0,p1).eval();
            out.rot.cols(0,p1)=out.rot.cols(0,p1)*rot1;
            // V.head_cols(nSv).eval().clean(1e-6).print("V for f");
            // arma::cx_mat(rot1).clean(1e-6).print("rot1 for f");
            // std::cout<<"\n is rot = "<<arma::norm(rot1.t()*rot1-arma::eye(arma::size(rot1)))<<"\n";
            // arma::abs(Kip-Kip2).eval().clean(1e-6).print("kip diff");
        }

        // std::cout<<"f final:"<<t0.sincemark()<<std::endl;
        t0.mark();

        // auto h=hImp;
        // for(auto i=0; i<3; i++) // We have 3 orbitals
        //     for(auto j=0; j<3; j++)
        //     if (std::abs(Kip(i,j))>tolSv)
        //         h += Kip(i,j),"Cdag",i+1,"C",j+1;
        // auto mpo=itensor::toMPO(h);

        // prepare the output
        out.ham=HamSys{sites};
        out.Kip=Kip;
        out.givens=givens;
        return out;
    }

    /// return the HamSys and the list of Givens rotations.
    template<class T>
    HamIPOut<T> HamIP_f2_opt(arma::Mat<T> const& rot, int nImp, arma::Mat<T> cc, double dt, bool extractf, double tolSv=1e-9) const
    {
        if (nImp==rot.n_rows) return {Ham(rot),{},{}};
        HamIPOut<T> out;

        int L=rot.n_cols;
        arma::vec ni=cc.diag();
        arma::cx_mat  Kip; // interaction picture
        {
            arma::Mat<T> K0=rot.t()*K*rot;
            arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, rot.n_rows-1) *
                               K0.submat(nImp,nImp,rot.n_rows-1, rot.n_rows-1) * arma::cx_double(0,-0.5*dt); //the commutator
            Kip=K0 * arma::cx_double(1,0);
            Kip.submat(nImp, nImp, rot.n_rows-1, rot.n_rows-1).fill(0.0);
            Kip.submat(0,0,nImp-1,nImp-1)=K0.submat(0,0,nImp-1,nImp-1) * arma::cx_double(1,0);
            Kip.submat(0, nImp, nImp-1, rot.n_rows-1)+=K1;
            Kip.submat(nImp, 0, rot.n_rows-1, nImp-1)+=K1.t();
        }
        out.rot=this->rotIP(rot,nImp,dt);

        int p0=rot.n_cols-1; {// the position where Slater starts
            for(; p0>nImp; p0--)
                if (ni[p0]>tolSv && ni[p0]<1-tolSv) { p0++; break; }
        }
        out.from=p0+1;
        if (p0+2<L) // in fact there is Slater
        {
            // 1) find the first site where the occupation is not the one at posS
            auto isEmpty=[](double x){ return x<0.5; };  // helper function
            out.to = std::find_if(ni.begin()+p0+1, ni.end(),
                                  [&,n0=ni[p0]](double x) {return isEmpty(x)!=isEmpty(n0); })
                    - ni.begin();

            // 2) swap sites in the Hamiltonian
            std::swap(ni[out.from], ni[out.to]);
            arma::cx_mat rot1(L, L, arma::fill::eye);
            rot1.swap_cols(out.from, out.to);
            // 3) extract f orbital of the sites with ni=0 and ni=1 independently
            arma::vec nSlater=ni.rows(p0,L-1).eval();
            arma::uvec posImp(nImp);
            for(auto i=0; i<nImp; i++) posImp[i]=i;

            { // ni==0
                arma::uvec pos0=arma::find(nSlater<0.5)+p0;
                auto k12=Kip.submat(posImp,pos0).eval();
                arma::vec s;
                arma::Mat<T> U, V;
                svd(U,s,V,k12);
                rot1.cols(pos0)=(rot1.cols(pos0).eval()*V).eval();
            }
            { // ni==1  TODO: this should add a fermionic sign to the state --> det(V)
                arma::uvec pos0=arma::find(nSlater>0.5)+p0;
                auto k12=Kip.submat(posImp,pos0).eval();
                arma::vec s;
                arma::Mat<T> U, V;
                svd(U,s,V,k12);
                rot1.cols(pos0)=(rot1.cols(pos0).eval()*V).eval();
            }
            Kip=(rot1.t()*Kip*rot1).eval();
            out.rot=out.rot*rot1;
        }

        std::vector<GivensRot<T>> givens;
        if (extractf){// the circuit to extract f orbitals. TODO: replace this by the Nat Orb Hamiltonian
            int p1=std::min(L-1,p0+1);
            auto k12=Kip.submat(0,nImp,nImp-1,p1);
            arma::vec s;
            arma::Mat<T> U, V;
            svd_econ(U,s,V,arma::conj(k12));
            int nSv=arma::find(s>tolSv*s[0]).eval().size();
            //std::cout<<"nSV="<<nSv<<std::endl;
            givens=GivensRotForRot_left(V.head_cols(nSv).eval());
            for(auto& g:givens) g.b+=nImp;
            arma::Mat<T> rot1=matrot_from_Givens(givens, rot.n_cols).st();
            Kip=(rot1.t()*Kip*rot1).eval();
            out.rot = out.rot * rot1;
        }

        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kip(i,j))>tolSv)
                h += Kip(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);

        // prepare the output
        out.ham=HamSys{sites,mpo,mpo};
        out.Kip=Kip;
        out.givens=givens;
        return out;
    }

    /// return the HamSys and the list of Givens rotations to be applied to the state.
    /// It implement the interaction picture for the Slater part only.
    /// nImp is the number of active orbitals.
    template<class T>
    std::tuple<HamSys, std::vector<GivensRot<T>>, arma::Mat<T> > HamIPS(arma::Mat<T> const& rot, int nImp, double dt, bool extractf,double tolSv=1e-12) const
    {
        if (nImp==rot.n_rows) return {Ham(rot),{},{}};
        arma::Mat<T> K0=rot.t()*K*rot;
        arma::cx_mat K1 = K0.submat(0, nImp, nImp-1, rot.n_rows-1) *
                           K0.submat(nImp,nImp,rot.n_rows-1, rot.n_rows-1) * arma::cx_double(0,+0.5*dt); //the commutator
        arma::cx_mat Kip=K0 * arma::cx_double(1,0);
        Kip.submat(nImp, nImp, rot.n_rows-1, rot.n_rows-1).fill(0.0);
        Kip.submat(0,0,nImp-1,nImp-1)=K0.submat(0,0,nImp-1,nImp-1) * arma::cx_double(1,0);
        Kip.submat(0, nImp, nImp-1, rot.n_rows-1)+=K1;
        Kip.submat(nImp, 0, rot.n_rows-1, nImp-1)+=K1.t();

        std::vector<GivensRot<T>> givens;
        if (extractf){// the circuit to extract f orbitals
            auto k12=Kip.submat(0,nImp,nImp-1,rot.n_rows-1);
            arma::vec s;
            arma::Mat<T> U, V;
            svd_econ(U,s,V,arma::conj(k12));
            int nSv=arma::find(s>tolSv*s[0]).eval().size();
            //std::cout<<"nSV="<<nSv<<std::endl;
            givens=GivensRotForRot_left(V.head_cols(nSv).eval());
            for(auto& g:givens) g.b+=nImp;
            arma::Mat<T> rot1=matrot_from_Givens(givens, rot.n_cols).st();
            Kip=(rot1.t()*Kip*rot1).eval();
        }

        auto h=hImp;
        for(auto i=0; i<sites.length(); i++)
            for(auto j=0; j<sites.length(); j++)
            if (std::abs(Kip(i,j))>1e-15)
                h += Kip(i,j),"Cdag",i+1,"C",j+1;
        auto mpo=itensor::toMPO(h);
        HamSys ham{sites,mpo,mpo};
        return make_tuple(ham, givens, Kip);
    }


    /// return the phase of the Fermi sea: exp(-i*dt*H2)
    template<class T>
    cmpx rotIPS(arma::Mat<T> const& rot, int nImp, double dt, arma::Mat<T> const& cc)
    {
        arma::Mat<T> K0=rot.t()*K*rot;
        T sum=0;
        for(size_t i=nImp; i<K0.n_rows; i++)
            sum += K0(i,i)*cc(i,i);
        return std::exp(cmpx(0,-dt)*sum);
    }

    template<class T>
    auto TrotterGates(arma::Mat<T> const& Kip,int nTB,double dt) const
    {
        using namespace itensor;

        auto gates = std::vector<BondGate>();

        //Create the gates exp(-i*tstep/2*hterm)
        for(int b = 1; b <= nTB-1; ++b)
        {
            auto hterm =
                     Kip(b-1,b  )*op(sites,"Adag",b) *op(sites,"A",b+1);
            hterm += Kip(b,b-1  )*op(sites,"A",b)    *op(sites,"Adag",b+1);
            hterm += Kip(b-1,b-1)*op(sites,"N",b)    *op(sites,"Id",b+1);
           if (b==nTB-1) hterm += Kip(b,b    )*op(sites,"Id",b)   *op(sites,"N",b+1);
            if (b==1) hterm += irlm.U*op(sites,"N",b)*op(sites,"N",b+1);

            auto g = BondGate(sites,b,b+1,BondGate::tReal,dt/2.,hterm);
            gates.push_back(g);
        }
        //Create the gates exp(-i*tstep/2*hterm) in reverse
        for(int b = nTB-1; b >= 1; --b)
        {
            auto hterm = Kip(b-1,b)*op(sites,"Adag",b)*op(sites,"A",b+1);
            hterm += Kip(b,b-1)*op(sites,"A",b)*op(sites,"Adag",b+1);
            hterm += Kip(b-1,b-1)*op(sites,"N",b)*op(sites,"Id",b+1);
            if (b==nTB-1) hterm += Kip(b,b)*op(sites,"Id",b)*op(sites,"N",b+1);
            if (b==1) hterm += irlm.U*op(sites,"N",b)*op(sites,"N",b+1);

            auto g = BondGate(sites,b,b+1,BondGate::tReal,dt/2.,hterm);
            gates.push_back(g);
        }
        return gates;
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
            auto s1 = itensor::dag(sites(b));
            auto s2 = itensor::dag(sites(b+1));
            auto s1p = prime(sites(b));
            auto s2p = prime(sites(b+1));
            itensor::ITensor hterm(s1,s2,s1p,s2p);
            hterm.set(s1(1),s2(1),s1p(1),s2p(1), rot(0,0));
            hterm.set(s1(2),s2(2),s1p(2),s2p(2), rot(3,3));
            hterm.set(s1(2),s2(1),s1p(2),s2p(1), rot(1,1));
            hterm.set(s1(2),s2(1),s1p(1),s2p(2), rot(1,2));
            hterm.set(s1(1),s2(2),s1p(2),s2p(1), rot(2,1));
            hterm.set(s1(1),s2(2),s1p(1),s2p(2), rot(2,2));
            return BondGate(sites,b,b+1,hterm);
        };

        auto mykron=[](mat22 const& A,mat22 const& B) { return mat44 {kron(B,A).st()}; };

        auto gates = std::vector<BondGate>();

        //Create the gates exp(-i*tstep/2*hterm)
        for(int i=0; i<nTB-1; ++i)
        {
            cx_mat44 hloc = Kip(i,i+1)*mykron(Cdag,C);
            hloc += Kip(i+1,i)*mykron(C,Cdag);
            hloc += Kip(i,i)*mykron(N,Id);
            if (i==nTB-2) hloc += Kip(i+1,i+1)*mykron(Id,N);
            if (i==0) hloc += T(irlm.U)*mykron(N,N);

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
            if (i==0) hloc += T(irlm.U)*mykron(N,N);

            cx_mat44 rot=expIH<T>(hloc * (0.5*dt));
            gates.push_back(to_itgate(i,rot));
        }
        return gates;
    }


};


#endif // IRLM_H
