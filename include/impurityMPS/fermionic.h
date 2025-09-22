#ifndef FERMIONIC_H
#define FERMIONIC_H

#include "givens_rotation.h"

#include<armadillo>
#include <map>
#include <array>
#include <itensor/all.h>
#include<mkl_lapacke.h>

inline std::pair<arma::vec,arma::mat> FullDiagonalizeTridiagonal(arma::vec an, arma::vec bn)
{
    lapack_int n=an.size(), M;
    arma::vec eval(n);
    arma::mat evec(n,n);
    std::vector<lapack_int> ifail(n);
    int info=LAPACKE_dstevr(LAPACK_COL_MAJOR,'V','A', n, an.memptr(), bn.memptr(),
                              0.0, 0.0,1,1,2e-11,&M,eval.memptr(),evec.memptr(),n,ifail.data());
    if (info!=0) throw
            std::runtime_error("LAPACKE_dstevx inside DiagonalizeTridiagonal, info!=0");
    return std::make_pair(eval,evec);
}

auto addTWoSites(itensor::Fermion sites, itensor::MPS const& psi)
{
    using namespace itensor;
    int N=psi.length();
    auto state = itensor::InitState(sites);
    for(int i = 1; i <= sites.length(); ++i)
    {
        if(i%2 == 0) state.set(i,"1");
        else         state.set(i,"0");
    }
    auto psi2 = itensor::MPS(state);
    for(int i = 1; i+1 <= psi.length(); ++i)
        psi2.Aref(i)=psi(i);
    { // manually copy the last tensor
        auto a=commonIndex(psi(N-1), psi(N));
        auto s=uniqueIndex(psi(N),psi(N-1));
        auto b=commonIndex(psi2(N),psi2(N+1));
        auto T=ITensor(a,s,b);
        for(auto ai:range1(a))
            for(auto si:range1(s)) {
                auto value=psi(N).eltC(a(ai), s(si));
                T.set(a(ai), s(si).dag(), b(1), value);
            }
        psi2.set(N,T);
    }
    psi2.replaceSiteInds(sites.inds());
    return psi2;
}


struct DmrgParam {
    int max_bond_dim=512;
    int nIter_diag=4;
    double noise=1e-8;
};

struct HamSys {
    itensor::Fermion sites;
    itensor::MPO ham;
    itensor::MPO hamEnrich;
};

struct HamSysV {
    itensor::Fermion sites;
    std::vector<itensor::MPO> ham;
    itensor::MPO hamEnrich;
};

struct HamSysExact {
    itensor::Fermion sites;
    itensor::AutoMPO ampo;
};

struct Fermionic {
    arma::mat Kmat, Umat;
    std::map<std::array<int,4>, double> Vijkl;
    arma::mat Rot;
    itensor::Fermion sites;


    explicit Fermionic(arma::mat const& Kmat_, arma::mat const& Umat_={}, std::map<std::array<int,4>, double> const& Vijkl_={})
        : Kmat(Kmat_), Umat(Umat_), Vijkl(Vijkl_), sites(Kmat_.n_rows, {"ConserveNf",true})
    {}

    Fermionic(arma::mat const& Kmat_, arma::mat const& Umat_,
              arma::mat const& Rot_, bool rotateKin=true)
        : Kmat(rotateKin ? Rot_.t()*Kmat_*Rot_ : Kmat_)
        , Umat(Umat_), Rot(Rot_)
        , sites(Kmat_.n_rows, {"ConserveNf",true})
    {}

    int length() const { return Kmat.n_rows; }

    void Kin(itensor::AutoMPO& h) const
    {
        int L=length();
        // kinetic energy bath
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (fabs(Kmat(i,j))>1e-15)
                    h += Kmat(i,j),"Cdag",i+1,"C",j+1;
    }

    std::vector<itensor::MPO> KinV() const
    {
        int L=length();
        std::vector<itensor::MPO> h(L);
        // kinetic energy bath
        #pragma omp parallel for
        for(int i=0;i<L; i++) {
            itensor::AutoMPO ampo(sites);
            for(int j=0;j<L; j++)
                if (fabs(Kmat(i,j))>1e-15)
                    ampo += Kmat(i,j),"Cdag",i+1,"C",j+1;
            h[i]=itensor::toMPO(ampo);
        }
        return h;
    }

    void Interaction(itensor::AutoMPO& h) const
    {
        if (Umat.empty() && Vijkl.empty()) return;
        if (!Rot.empty()) return InteractionRot(h);
        // Uij ni nj
        for(int i=0u;i<Umat.n_rows; i++)
            for(int j=0;j<Umat.n_cols; j++)
                if (fabs(Umat(i,j))>1e-15)
                    h += Umat(i,j),"Cdag",i+1,"C",i+1,"Cdag",j+1,"C",j+1;

        for(const auto& [pos,coeff] : Vijkl)
            if (fabs(coeff)>1e-15)
                h += coeff,"Cdag",pos[0]+1,"C",pos[1]+1,"Cdag",pos[2]+1,"C",pos[3]+1;
    }

    std::vector<itensor::MPO> InteractionV() const
    {
        if (Umat.empty() && Vijkl.empty()) return {};
        if (!Rot.empty()) throw std::invalid_argument("rot of interactionV is not implemented yet");

        std::vector<itensor::MPO> h;
        // Uij ni nj
        for(int i=0;i<Umat.n_rows; i++) {
            itensor::AutoMPO ampo(sites);
            for(int j=0;j<Umat.n_cols; j++)
                if (fabs(Umat(i,j))>1e-15)
                    ampo += Umat(i,j),"Cdag",i+1,"C",i+1,"Cdag",j+1,"C",j+1;
            if (ampo.size()) h.push_back(itensor::toMPO(ampo));
        }

        itensor::AutoMPO ampo(sites);
        for(const auto& [pos,coeff] : Vijkl)
            if (fabs(coeff)>1e-15)
                ampo += coeff,"Cdag",pos[0]+1,"C",pos[1]+1,"Cdag",pos[2]+1,"C",pos[3]+1;
        if (ampo.size()) h.push_back(itensor::toMPO(ampo));
        return h;
    }

    void InteractionRot(itensor::AutoMPO& h) const
    {
        int L=length();
        // Uij ni nj
        for(auto a=0; a<L; a++)
            for(auto b=0; b<L; b++)
                for(auto c=0; c<L; c++)
                    for(auto d=0; d<L; d++) {
                        double Vabcd=0;
                        for(int i=0;i<Umat.n_rows; i++)
                            for(int j=0;j<Umat.n_cols; j++)
//                                if (fabs(Umat(i,j))>1e-15)
                                    Vabcd += Umat(i,j)*Rot(i,a)*Rot(i,b)*Rot(j,c)*Rot(j,d);
                        if (fabs(Vabcd)>1e-15)
                            h += Vabcd,"Cdag",a+1,"C",b+1,"Cdag",c+1,"C",d+1;
                    }
    }

    HamSys Ham() const
    {
        itensor::AutoMPO h(sites);
        Kin(h);
        Interaction(h);
        return {sites, itensor::toMPO(h)};
    }

    HamSysV HamV() const
    {
        std::vector<itensor::MPO> hk=KinV();
        std::vector<itensor::MPO> hi=InteractionV();
        for(auto const& x:hi) hk.push_back(x);
        return {sites, hk};
    }

    static arma::cx_mat cc_matrix(itensor::MPS const& gs, itensor::Fermion const& sites, int L=-1)
    {
        if (L==-1) L=sites.length();
        auto ccz=correlationMatrixC(gs, sites,"Cdag","C",itensor::range1(L));
        arma::cx_mat cc(ccz.size(), ccz.size());
        for(auto i=0u; i<ccz.size(); i++)
            for(auto j=0u; j<ccz[i].size(); j++)
                cc(i,j)=ccz.at(i).at(j);
        return cc;
    }

    static arma::cx_mat cc_matrix_kondo(itensor::MPS const& gs, itensor::Fermion const& sites, int L=-1)
    {
        itensor::MPS psi=gs;
        {// apply the excitation
            psi.position(1);
            auto newA = op(sites, "A", 1)* psi(1);
            newA.noPrime();
            psi.ref(1)=newA;
        }
        double norma=itensor::norm(psi);
        return cc_matrix(psi, sites, L)*pow(norma,2);
    }

    // return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
    template<class T>
    static std::vector<GivensRot<T>> NOGivensRot(arma::Mat<T> const& cc, int nExclude=2, size_t blockSize=8, double tolEvec=1e-10, int pfinal=-1)
    {
        if (pfinal==-1 || pfinal>cc.n_rows-1) pfinal=cc.n_rows-1;
        using namespace arma;
        arma::Mat<T> cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        std::vector<GivensRot<T>> gs;
        arma::Mat<T> evec;
        arma::vec eval;
        size_t d=blockSize;
        pfinal -= nExclude;
        for(auto p2=pfinal; p2>0u; p2--) {
            size_t p1= (p2+1>d) ? p2+1-d : 0u ;
            if(p2+4>pfinal) p1=0;
            arma::Mat<T> cc2=cc1.submat(p1,p1,p2,p2);
            arma::eig_sym(eval,evec,cc2);
            // select the less active
            size_t pos=0;
            if (1-eval.back()<eval(0)) pos=eval.size()-1;
            arma::Col<T> v=evec.col(pos);
            //if (1-std::abs(v.back())<tolEvec) continue; // already done
            std::vector<GivensRot<T>> gs1;
            for(auto i=0u; i+1<v.size(); i++)
            {
                //if (std::abs(v[i])<tolEvec) continue; // already done
                auto b=i+p1;
                auto g=GivensRot<T>::createFromPair(b,v[i],v[i+1],true, &v[i+1]);
                gs1.push_back(g);
            }
            auto rot1=matrot_from_Givens(gs1,p2+1);
            cc1.submat(0,0,p2,p2)=rot1*cc1.submat(0,0,p2,p2)*rot1.t();
            for(auto g : gs1) { g.b+=nExclude; gs.push_back(g); }
        }
        return gs;
    }

    // return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
    // from the wannierized inactive orbitals, move right the one with biggest <X>
    template<class T>
    static std::vector<GivensRot<T>> NOGivensRot_W(arma::Mat<T> const& cc, int nExclude=2, size_t blockSize=8, double tolActivity=1e-9, int pfinal=-1)
    {
        if (pfinal==-1) pfinal=cc.n_rows-1;
        using namespace arma;
        arma::Mat<T> cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        std::vector<GivensRot<T>> gs;
        arma::Mat<T> evec;
        arma::vec eval;
        size_t d=blockSize;
        pfinal -= nExclude;
        for(auto p2=pfinal; p2>0u; p2--) {
            size_t p1= (p2+1>d) ? p2+1-d : 0u ;
            if(p2+4>pfinal) p1=0;
            arma::Mat<T> cc2=cc1.submat(p1,p1,p2,p2);
            arma::eig_sym(eval,evec,cc2);

            // select the less active
            size_t pos=0;
            if (1-eval.back()<eval(0)) pos=eval.size()-1;
            arma::Col<T> v=evec.col(pos);

            arma::mat xOp=arma::diagmat(arma::regspace(0,cc2.n_rows-1));
            double xBest=0;
            {// group natural orbitals with occupation 0
                std::vector<size_t> ieval0v;
                for(auto i=0u; i<eval.size(); i++)
                    if (eval[i]<tolActivity) ieval0v.push_back(i);
                if (!ieval0v.empty()) {
                    uvec ieval0=conv_to<uvec>::from(ieval0v);
                    arma::Mat<T> evec0=evec.cols(ieval0);
                    arma::Mat<T> X=evec0.t()* xOp * evec0;
                    arma::vec Xeval0;
                    arma::Mat<T> Xevec0;
                    arma::eig_sym(Xeval0,Xevec0,X);
                    xBest=Xeval0.back();
                    v=evec0 * Xevec0.col(X.n_cols-1);
                }
            }
            {// group natural orbitals with occupation 1
                std::vector<size_t> ieval0v;
                for(auto i=0u; i<eval.size(); i++)
                    if (std::abs(eval[i]-1)<tolActivity) ieval0v.push_back(i);
                if (!ieval0v.empty()) {
                    uvec ieval0=conv_to<uvec>::from(ieval0v);
                    arma::Mat<T> evec0=evec.cols(ieval0);
                    arma::Mat<T> X=evec0.t()* xOp * evec0;
                    arma::vec Xeval0;
                    arma::Mat<T> Xevec0;
                    arma::eig_sym(Xeval0,Xevec0,X);
                    if (Xeval0.back()>xBest)
                        v=evec0 * Xevec0.col(X.n_cols-1);
                }
            }

            //if (1-std::abs(v.back())<tolEvec) continue; // already done
            std::vector<GivensRot<T>> gs1;
            for(auto i=0u; i+1<v.size(); i++)
            {
                // if (std::abs(v[i])*blockSize<tolActivity) continue; // already done
                auto b=i+p1;
                auto g=GivensRot<T>::createFromPair(b,v[i],v[i+1],true, &v[i+1]);
                gs1.push_back(g);
            }
            auto rot1=matrot_from_Givens(gs1,p2+1);
            cc1.submat(0,0,p2,p2)=rot1*cc1.submat(0,0,p2,p2)*rot1.t();
            for(auto g : gs1) { g.b+=nExclude; gs.push_back(g); }
        }
        return gs;
    }

    static std::pair<int,int> bestMatching(std::vector<double> const& a, std::vector<double> const& b)
    {
        assert(a.size() && b.size());
        auto best=std::numeric_limits<double>::max();
        int i0,j0;
        for(auto i=0u; i<a.size(); i++)
            for(auto j=0u; j<b.size(); j++)
                if (auto d=std::abs(a[i]-b[j]); d<best) {best=d; i0=i; j0=j;}
        return {i0,j0};
    }


    /// return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
    /*static std::vector<GivensRot<>> GivensRotForMatrix(arma::mat const& cc, int nExclude=2, size_t blockSize=8, double tolEvec=1e-10)
    {
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        std::vector<GivensRot<>> gs;
        arma::mat evec;
        arma::vec eval;
        auto evalRef=conv_to<std::vector<double>>::from( eig_sym(cc1) );
        size_t d=blockSize;
        for(auto p2=cc1.n_rows-1; p2>0u; p2--) {
            size_t p1= (p2+1>d) ? p2+1-d : 0u ;
            arma::mat cc2=cc1.submat(p1,p1,p2,p2);
            arma::eig_sym(eval,evec,cc2);
            // select the less active
            auto [i0,j0]=bestMatching(conv_to<std::vector<double>>::from(eval), evalRef);
            evalRef.erase(evalRef.begin()+j0);
            arma::vec v=evec.col(i0);
            if (1-std::abs(v.back())<tolEvec) continue; // already done
            std::vector<GivensRot<>> gs1;
            for(auto i=0u; i+1<v.size(); i++)
            {
                auto b=i+p1;
                auto g=GivensRot<>::createFromPair(b,v[i],v[i+1]);
                gs1.push_back(g);
                v[i+1]=g.r;
            }
            auto rot1=matrot_from_Givens(gs1);
            cc1.submat(0,0,p2,p2)=rot1*cc1.submat(0,0,p2,p2)*rot1.t();
            for(auto g : gs1) { g.b+=nExclude; gs.push_back(g); }
        }
        return gs;
    }
    */


    /// return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
    template<class T>
    static std::vector<itensor::BondGate> NOGates1(itensor::Fermion const& sites, std::vector<GivensRot<T>> const& gs)
    {
        using itensor::BondGate;
        using itensor::Cplx_i;
        std::vector<itensor::BondGate> gates;
        for(const GivensRot<T>& g : gs)
        {
            int b=g.b+1;
            auto Kin=(g.ilogMatrix()*(-1.0)).eval();
            // auto hterm = ( sites.op("Adag",b)*sites.op("A",b+1)
            //               -sites.op("A",b+1)*sites.op("A",b))* (g.angle()*Cplx_i);
            itensor::ITensor hterm;
            if (std::abs(Kin(0,1))>1e-15) hterm += sites.op("Adag",b)*sites.op("A",b+1) * Kin(1,0);
            if (std::abs(Kin(1,0))>1e-15) hterm += sites.op("A",b)*sites.op("Adag",b+1) * Kin(0,1);
            if (std::abs(Kin(0,0))>1e-15) hterm += sites.op("N",b)*sites.op("Id",b+1) * Kin(0,0);
            if (std::abs(Kin(1,1))>1e-15) hterm += sites.op("Id",b)*sites.op("N",b+1) * Kin(1,1);

            if (hterm) {
                auto bg=BondGate(sites,b,b+1,BondGate::tReal,1,hterm);
                gates.push_back(bg);
            }
        }
        return gates;
    }

    // return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
    template<class T>
    static std::vector<itensor::BondGate> NOGates(itensor::Fermion const& sites, std::vector<GivensRot<T>> const& gs)
    {
        using itensor::BondGate;
        using itensor::Cplx_i;
        std::vector<itensor::BondGate> gates;
        for(const GivensRot<T>& g : gs)
        {
            int b=g.b+1;
            auto rot=g.matrix().t().eval();

            auto s1 = itensor::dag(sites(b));
            auto s2 = itensor::dag(sites(b+1));
            auto s1p = prime(sites(b));
            auto s2p = prime(sites(b+1));
            itensor::ITensor hterm(s1,s2,s1p,s2p);
            hterm.set(s1(1),s2(1),s1p(1),s2p(1), 1);
            hterm.set(s1(2),s2(2),s1p(2),s2p(2), 1);
            hterm.set(s1(2),s2(1),s1p(2),s2p(1), rot(0,0));
            hterm.set(s1(2),s2(1),s1p(1),s2p(2), rot(0,1));
            hterm.set(s1(1),s2(2),s1p(2),s2p(1), rot(1,0));
            hterm.set(s1(1),s2(2),s1p(1),s2p(2), rot(1,1));

            if (hterm) {
                auto bg=BondGate(sites,b,b+1,hterm);
                gates.push_back(bg);
            }
        }
        return gates;
    }

    static arma::mat rotNO(arma::mat const& cc, int nExclude=2)
    {
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);
        arma::vec activity(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            activity[i]=-std::min(eval[i], -eval[i]+1);
        arma::uvec iev=arma::stable_sort_index(activity.clean(1e-14));
        eval(iev).print("evals");
        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1)=evec.cols(iev);
        return rot;
    }

    static arma::mat rotNO2(arma::mat const& cc, int nExclude=2, double tolWannier=1e-5)
    {
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);
        arma::vec eval2(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            eval2[i]=-std::min(eval[i], -eval[i]+1);    //activity sorting
        arma::uvec iev=arma::sort_index(eval2);
        eval(iev).print("evals");

        arma::mat evec2=evec.cols(iev);
        arma::mat evec3=evec2;

        // Wannier after activity sorting
        arma::mat J=arma::diagmat(arma::regspace(0,eval.size()-1));
        {
            std::vector<size_t> ieval0v;
            for(auto i=0u; i<eval.size(); i++)
                if (eval[iev[i]]<tolWannier) ieval0v.push_back(i);
            uvec ieval0=conv_to<uvec>::from(ieval0v);
            arma::mat evec0=evec2.cols(ieval0);
            arma::mat X=evec0.t()* J * evec0;
            arma::mat wevec;
            arma::vec weval;
            arma::eig_sym(weval,wevec,X);
            evec3.cols(ieval0) = evec2.cols(ieval0) * wevec;
        }
        arma::mat evec4=evec3;
        {
            std::vector<size_t> ieval0v;
            for(auto i=0u; i<eval.size(); i++)
                if (std::abs(1.0-eval[iev[i]])<tolWannier) ieval0v.push_back(i);
            uvec ieval0=conv_to<uvec>::from(ieval0v);
            arma::mat evec0=evec3.cols(ieval0);
            arma::mat X=evec0.t()* J * evec0;
            arma::mat wevec;
            arma::vec weval;
            arma::eig_sym(weval,wevec,X);
            evec4.cols(ieval0) = evec3.cols(ieval0) * wevec;
        }

        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1)=evec4;
        return rot;
    }

//    static arma::mat rotNO3(arma::mat const& cc, int nExclude=2, int nActive=8, double maxBlock=0)
    static arma::mat rotNO3(arma::mat const& cc, arma::mat xOp, int nExclude=2,double tolWannier=1e-5, double maxBlock=0)
    {
        if (maxBlock==0) maxBlock=cc.n_rows;
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat J=xOp.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat evec;
        arma::vec eval;
        arma::eig_sym(eval,evec,cc1);  // +1e-5/cc1.n_rows*J
        arma::vec activity(eval.size());
        for(auto i=0u; i<eval.size(); i++)
            activity[i]=-std::min(eval[i], -eval[i]+1);    //activity sorting
        arma::uvec iev=arma::stable_sort_index(activity.clean(1e-15));
        eval(iev).print("evals");

        arma::vec eval2=eval(iev);
        arma::mat evec2=evec.cols(iev);

        // fix the sign and collect distance to 1
        arma::vec diff1(evec2.n_rows);
        {
            arma::mat one(evec2.n_rows, evec2.n_cols, fill::eye);
            for(auto j=0u; j<evec2.n_cols; j++) {
                auto i=arma::index_max(arma::abs(evec2.col(j)));
                evec2.col(j) /= arma::sign(evec2(i,j));
                diff1(j)=arma::norm(evec2.col(j)-one.col(j));
            }
        }

        // Wannier after activity sorting
        uvec weval;
        arma::vec Xeval;
        arma::mat Xevec;

        arma::vec x_sigma;
        {
            vec xi=(arma::mat {evec2.t()*J*evec2}).diag();
            vec xi2=(arma::mat {evec2.t()*(J-diagmat(xi)) * (J-diagmat(xi))*evec2}).diag();
            x_sigma=arma::sqrt(xi2.clean(1e-15));
        }
        //x_sigma.print("xsigma initial");

        {// group empty natural orbitals
            std::vector<size_t> ieval0v;
            for(auto i=0u; i<eval2.size(); i++)
                if (eval2[i]<tolWannier ||
//                if ((i>=nActive && eval2[i]<0.5) ||
//                   (diff1[i]>1.0 && eval2[i]<0.5) ||
                   (2*x_sigma[i]>maxBlock && eval2[i]<0.5)) ieval0v.push_back(i);
            uvec ieval0=conv_to<uvec>::from(ieval0v);
            arma::mat evec0=evec2.cols(ieval0);
            arma::mat X=evec0.t()* J * evec0;
            arma::vec Xeval0;
            arma::mat Xevec0;
            arma::eig_sym(Xeval0,Xevec0,X);

            weval=ieval0;
            Xeval=Xeval0;
            Xevec=Xevec0;
        }
        {// group full natural orbitals
            std::vector<size_t> ieval1v;
            for(auto i=0u; i<eval2.size(); i++)
                if (std::abs(1.0-eval2[i])<tolWannier ||
//                if ((i>=nActive && eval2[i]>=0.5) ||
//                   (diff1[i]>1.0 && eval2[i]>=0.5) ||
                   (2*x_sigma[i]>maxBlock && eval2[i]>=0.5)) ieval1v.push_back(i);
            uvec ieval1=conv_to<uvec>::from(ieval1v);
            arma::mat evec1=evec2.cols(ieval1);
            arma::mat X=evec1.t()* J * evec1;
            arma::vec Xeval1;
            arma::mat Xevec1;
            arma::eig_sym(Xeval1,Xevec1,X);

            weval=join_vert(weval,ieval1);
            Xeval=join_vert(Xeval,Xeval1);
            arma::mat rot(Xeval.size(), Xeval.size(), fill::zeros); // tensor addition of the two rotations
            rot.submat(0,0,Xevec.n_rows-1,Xevec.n_cols-1)=Xevec;
            rot.submat(Xevec.n_rows,Xevec.n_cols,rot.n_rows-1,rot.n_cols-1)=Xevec1;
            Xevec=rot;
        }

        // apply Wannierization
        arma::vec eval3=eval2;
        arma::mat evec3=evec2;
        evec3.cols(weval) = evec2.cols(weval) * Xevec;

        // sort Wanier orbitals according to position
        arma::uvec Xiev(Xeval.size());
        {
            Xiev=arma::stable_sort_index(Xeval);
//            for(auto i=0u; i<Xeval.size(); i++) Xiev[i]=i;
//            int c=0;
//            for(auto i=0u; i<neval0; i++) Xiev[2*i]=i;
//            for(auto i=0; i+neval0<Xeval.size(); i++)
//                if (i<2*neval0) Xiev[2*i+1]=i+neval0;
//                else Xiev[2*neval0+c++]=i+neval0;
        }        

        arma::vec eval4=eval3;
        arma::mat evec4=evec3;
        eval4(arma::sort(weval))=eval3(weval(Xiev));
        evec4.cols(arma::sort(weval)) = evec3.cols(weval(Xiev));

        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1)=evec4;

//        eval4.print("eval cicj");
//        Xeval(Xiev).print("orbitals position");

        {
            vec xi=(arma::mat {evec4.t()*J*evec4}).diag();
            vec xi2=(arma::mat {evec4.t()*(J-diagmat(xi)) * (J-diagmat(xi))*evec4}).diag();
            x_sigma=arma::sqrt(xi2.clean(1e-15));
            arma::join_horiz(xi,x_sigma).print("<X> sigmaX");
        }

        // fix the sign
        for(auto j=0u; j<rot.n_cols; j++) {
            auto i=arma::index_max(arma::abs(rot.col(j)));
            rot.col(j) /= arma::sign(rot(i,j));
        }

        std::cout<<"norm(1-rot)="<<arma::norm(rot-arma::mat(rot.n_rows, rot.n_cols, fill::eye))<< std::endl;
//        if ( arma::norm(rot-arma::mat(rot.n_rows, rot.n_cols, fill::eye))>0.1 )
//                       rot.print("rotation");

        return rot;
    }

    static arma::mat rotNO4(arma::mat const& orb, arma::mat const& cc, int nExclude=2, double tolWannier=1e-5)
    {
        using namespace arma;
        arma::mat cc1=cc.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat orb1=orb.submat(nExclude,nExclude,cc.n_rows-1,cc.n_cols-1);
        arma::mat J=orb1.t() * arma::diagmat(arma::regspace(0,cc1.n_rows-1)) * orb1;


        // Wannier after activity sorting
        arma::uvec active;
        arma::vec Xeval;
        arma::mat Xevec;

        {// group active natural orbitals
            active=arma::find(cc1.diag()>tolWannier && cc1.diag()<1-tolWannier);
            arma::mat X=J(active,active);
            arma::eig_sym(Xeval,Xevec,X);
        }

        // apply Wannierization

        // sort Wannier orbitals according to position
        arma::uvec Xiev(Xeval.size());
        {
            Xiev=arma::stable_sort_index(Xeval);
//            for(auto i=0u; i<Xeval.size(); i++) Xiev[i]=i;
        }

        arma::mat evec4=Xevec.cols(Xiev);

        arma::mat rot(cc.n_rows,cc.n_cols,arma::fill::eye);
        rot.submat(nExclude,nExclude,arma::size(evec4))=evec4;

//        eval4.print("eval cicj");
//        Xeval(Xiev).print("orbitals position");

        {
            mat X=J(active,active);
            vec xi=(arma::mat {evec4.t()*X*evec4}).diag();
            vec xi2=(arma::mat {evec4.t()*(X-diagmat(xi)) * (X-diagmat(xi))*evec4}).diag();
            vec x_sigma=arma::sqrt(xi2.clean(1e-15));
            arma::join_horiz(xi,x_sigma).print("<X> sigmaX");
        }

        // fix the sign
        for(auto j=0u; j<rot.n_cols; j++) {
            auto i=arma::index_max(arma::abs(rot.col(j)));
            rot.col(j) /= arma::sign(rot(i,j));
        }

        return rot;
    }

    static HamSysExact rotOpExact(arma::mat const& rot)
    {
        arma::mat rott=rot.t();
        arma::cx_mat logrot;
        const auto im=arma::cx_double(0,1);
        {
            arma::cx_mat evec;
            arma::cx_colvec eval;
            arma::eig_gen(eval,evec,rott);
            logrot=evec*arma::diagmat(arma::log(eval)*im)*evec.i();
        }
        arma::cx_mat kin=logrot; //arma::logmat(rott)*im; // we need to invert the rotation
        auto L=rot.n_cols;
        itensor::Fermion sites(L, {"ConserveNf",true});
        itensor::AutoMPO h(sites);
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
                if (std::abs(kin(i,j))>1e-15)
                    h += kin(i,j),"Cdag",i+1,"C",j+1;
        arma::imag(kin).print("Hrot");
        return {sites, h};
    }

    // static HamSys rotOp(arma::mat const& rot, int nExclude=2)
    // {
    //     arma::mat rott=rot.t();
    //     arma::cx_mat logrot;
    //     const auto im=arma::cx_double(0,1);
    //     {
    //         auto [eval,evec]=eig_unitary(rott, nExclude);
    //         arma::cx_vec logeval=-arma::log(eval)*im;
    //         for(auto& x : logeval) { // fix sign of the
    //             if (std::real(x)>M_PI/2) x -= M_PI;
    //             else if (std::real(x)<-M_PI/2) x += M_PI;
    //         }
    //         logrot=evec*arma::diagmat(logeval)*evec.t();
    //         if (norm(logeval)>0.1) logeval.print("log(eval)");
    //         double err=arma::norm(rott-arma::expmat(im*logrot));
    //         if (err>1e-13) std::cout<<"exp error="<<err<<std::endl;

    //         double err_herm=norm(logrot.t()-logrot);
    //         if (err_herm>1e-13) std::cout<<"Hermitian error="<<err_herm<<std::endl;
    //     }
    //     arma::cx_mat kin=logrot; //arma::logmat(rott)*im; // we need to invert the rotation

    //     if (norm(kin)>0.1) kin.print("kin");
    //     auto L=rot.n_cols;
    //     itensor::Fermion sites(L, {"ConserveNf=",true});
    //     itensor::AutoMPO h(sites);
    //     for(int i=0;i<L; i++)
    //         for(int j=0;j<L; j++)
    //             if (std::abs(kin(i,j))>1e-15)
    //                 h += kin(i,j),"Cdag",i+1,"C",j+1;
    //     return {sites, itensor::toMPO(h)};
    // }

};

#endif // FERMIONIC_H
