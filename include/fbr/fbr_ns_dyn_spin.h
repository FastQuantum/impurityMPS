#ifndef FBR_NS_DYN_SPIN_H
#define FBR_NS_DYN_SPIN_H

#include "fbr_dyn_spin.h"
#include "orbital_update.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace fbr {

/// Spin-flip-symmetric SIAM evolution of several states in one orbital basis.
struct FbrNsDynSpin {
    ImpurityParamSpin param;
    double dt;
    arma::cx_mat Kbath;
    arma::cx_mat Kip0;
    arma::uvec imp_pos;
    arma::uvec bath_pos;
    arma::cx_mat rotS;

    std::vector<Fb_mps_spin<cmpx>> states;
    arma::cx_mat K;
    std::vector<double> energies;
    int nIter=0;

    explicit FbrNsDynSpin(ImpuritySpin const& imp,
                          std::vector<Fb_mps_spin<cmpx>> states_,
                          double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , states(std::move(states_))
    {
        if (states.empty())
            throw std::invalid_argument("FbrNsDynSpin: at least one state is required");
        checkCommonOrbitals();
        energies.assign(states.size(),-1000.0);

        int L=param.length();
        imp_pos=arma::conv_to<arma::uvec>::from(param.impPos);
        bath_pos=arma::conv_to<arma::uvec>::from(set_diff(L,param.impPos));

        arma::mat Kstar=param.Kmat;
        arma::vec d(L,arma::fill::zeros);
        d(bath_pos)=arma::vec(Kstar.diag())(bath_pos);
        arma::mat c1=Kstar; c1.each_row()%=d.t();
        arma::mat c2=Kstar; c2.each_col()%=d;
        arma::mat commutator=c1-c2;

        Kip0=Kstar*cmpx(1,0);
        Kip0(bath_pos,bath_pos).zeros();
        Kip0-=cmpx(0,0.5*dt)*commutator;
        rotS=states.front().rot;
        Kbath=Kstar.submat(bath_pos,bath_pos)*cmpx(1,0);

        auto [a_imp,b_imp]=states.front().interval_impurity(dw);
        auto [a_sla,b_sla]=states.front().interval_slater(dw);
        int nSv=0;
        if (a_imp<b_imp && a_sla<b_sla) {
            arma::cx_mat k12=param.Kmat.submat(a_imp,a_sla,b_imp-1,b_sla-1)*cmpx(1,0);
            arma::vec s; arma::cx_mat U,V;
            arma::svd_econ(U,s,V,k12);
            nSv=(s.empty() || s[0]==0)
                  ? 0
                  : (int)arma::find(s>states.front().tol*s[0]).eval().size();
        }
        for (auto& state : states)
            state.nSv=nSv;
    }

    void iterate(TdvpParam args={})
    {
        checkCommonOrbitals();
        K=buildK();
        nIter++;

        applyPlan(states.front().planRepresentative(K,0));
        applyPlan(states.front().planRepresentative(K,1));
        applyPlan(states.front().planActiveRepresentative(K));
        evolveAll(args);
        applyPlan(states.front().planNaturalOrbitals(combinedCc()));
    }

    arma::cx_mat effective_rot(std::size_t state=0) const
    {
        int L=param.length();
        arma::cx_mat exp_ih(L,L,arma::fill::eye);
        if (nIter>0)
            exp_ih.submat(bath_pos,bath_pos)=expIH<cmpx>(Kbath*(static_cast<double>(nIter)*dt));
        return rotS*exp_ih*rotS.t()*states.at(state).rot;
    }

    arma::cx_mat correlator_all(std::size_t state=0) const
    {
        arma::cx_mat Q=effective_rot(state);
        return arma::conj(Q)*states.at(state).cc*Q.st();
    }

    cmpx correlator(int i,int j,std::size_t state=0) const
    {
        arma::cx_mat Q=effective_rot(state);
        arma::cx_vec ccQj=states.at(state).cc*Q.row(j).st();
        return arma::cdot(Q.row(i).st(),ccQj);
    }

private:
    void checkCommonOrbitals() const
    {
        auto const& first=states.front();
        int L=param.length();
        if (first.length()!=L)
            throw std::invalid_argument("FbrNsDynSpin: state length does not match the model");
        for (std::size_t n=1; n<states.size(); ++n) {
            auto const& state=states[n];
            if (state.length()!=L || state.p1!=first.p1 || state.p2!=first.p2
                || state.imp_size!=first.imp_size)
                throw std::invalid_argument("FbrNsDynSpin: states do not share the same orbital layout");
            if (arma::norm(state.rot-first.rot,"fro")>10*first.tol)
                throw std::invalid_argument("FbrNsDynSpin: states do not share the same orbital rotation");
            for (int i=1; i<=L; ++i)
                if (state.sites(i)!=first.sites(i))
                    throw std::invalid_argument("FbrNsDynSpin: states must share ITensor site indices");
        }

        double slater_tol=std::max(100*first.tol,1e-10);
        for (int i=0; i<L; ++i) {
            if (i>=first.p1 && i<first.p2) continue;
            double occupation=std::real(first.cc(i,i))>0.5 ? 1.0 : 0.0;
            for (auto const& state : states)
                if (std::abs(state.cc(i,i)-occupation)>slater_tol)
                    throw std::invalid_argument("FbrNsDynSpin: states do not share the same Slater state");
            for (std::size_t n=1; n<states.size(); ++n)
                for (int j=0; j<L; ++j) {
                    if (j>=first.p1 && j<first.p2) continue;
                    if (std::abs(states[n].cc(i,j)-first.cc(i,j))>slater_tol)
                        throw std::invalid_argument("FbrNsDynSpin: states do not share the same Slater correlator");
                }
        }
    }

    arma::cx_vec ipPhase(int n) const
    {
        arma::cx_vec d(param.length(),arma::fill::ones);
        if (n>0)
            d(bath_pos)=arma::exp(-imag_1*Kbath.diag()*(static_cast<double>(n)*dt));
        return d;
    }

    arma::cx_mat buildK() const
    {
        arma::cx_vec d=ipPhase(nIter);
        arma::cx_mat A=rotS.cols(imp_pos).t()*states.front().rot;
        arma::cx_mat B=Kip0.rows(imp_pos);
        B.each_row()%=d.st();
        B=B*rotS.t();
        B=B*states.front().rot;
        arma::cx_mat D=Kip0.submat(imp_pos,imp_pos);
        return A.t()*B+B.t()*A-A.t()*(D*A);
    }

    arma::cx_mat combinedCc() const
    {
        arma::cx_mat cc(states.front().cc.n_rows,states.front().cc.n_cols,arma::fill::zeros);
        for (auto const& state : states) cc+=state.cc;
        cc/=static_cast<double>(states.size());
        return cc;
    }

    void applyPlan(OrbitalUpdate<cmpx> const& update)
    {
        update.applyAsBasis(K);
        Fb_mps_spin<cmpx>::ensure_reflection_mat(K);
        for (auto& state : states)
            state.applyUpdate(update);
    }

    void evolveAll(TdvpParam args)
    {
        auto [a,b]=states.front().interval_active_full();
        auto mpo=fullHamiltonian(a,b);

        for (std::size_t n=0; n<states.size(); ++n) {
            auto& state=states[n];
            auto sweeps=itensor::Sweeps(1);
            sweeps.maxdim()=args.max_bond_dim;
            sweeps.cutoff()=state.tol;
            sweeps.niter()=args.nIter_diag;
            sweeps.noise()=args.noise;

            if (args.epsilonM!=0) {
                std::vector<double> epsilonK(args.nKrylov,args.epsilonK);
                itensor::addBasis(state.psi,mpo,epsilonK,
                                  {"Cutoff",args.epsilonM,
                                   "Method","DensityMatrix",
                                   "KrylovOrd",args.nKrylov,
                                   "DoNormalize",true,
                                   "Quiet",true,
                                   "Silent",true});
            }

            energies[n]=itensor::tdvp(state.psi,mpo,-imag_1*dt,sweeps,
                                      {"Truncate",true,
                                       "DoNormalize",true,
                                       "Quiet",true,
                                       "Silent",true,
                                       "NumCenter",2,
                                       "ErrGoal",args.err_goal});
            energies[n]+=state.SlaterEnergy(K);
            state.update_cc();
        }
    }

    itensor::MPO fullHamiltonian(int a,int b) const
    {
        itensor::AutoMPO h(states.front().sites);
        int L=param.length();
        for (int i=0; i<L; ++i)
            for (int j=0; j<L; ++j)
                if (std::abs(param.Umat(i,j))>1e-15)
                    h+=param.Umat(i,j),"N",i+1,"N",j+1;
        for (int i=a; i<b; ++i)
            for (int j=a; j<b; ++j)
                if (std::abs(K(i,j))>states.front().tol)
                    h+=K(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }
};

} // namespace fbr

#endif // FBR_NS_DYN_SPIN_H
