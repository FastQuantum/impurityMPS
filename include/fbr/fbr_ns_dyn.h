#ifndef FBR_NS_DYN_H
#define FBR_NS_DYN_H

#include "fbr_dyn.h"
#include "orbital_update.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace fbr {

/// Few-body real-time evolution of several states in one common orbital basis.
///
/// The orbital transformations are found once from the collection of states and
/// applied identically to every MPS. Each state is nevertheless evolved by its
/// own TDVP call, since the TDVP projection and truncation are state-dependent.
struct FbrNsDyn {
    ImpurityParam param;
    double dt;
    arma::cx_mat exp_ih;
    arma::cx_mat Kip0;
    int nChannel;
    arma::uvec imp_pos;
    arma::uvec bath_pos;

    std::vector<Fb_mps<cmpx>> states;
    arma::cx_mat K;
    std::vector<double> energies;

    explicit FbrNsDyn(Impurity const& imp,
                      std::vector<Fb_mps<cmpx>> states_,
                      double dt_=0.1)
        : param(imp.param)
        , dt(dt_)
        , states(std::move(states_))
    {
        if (states.empty())
            throw std::invalid_argument("FbrNsDyn: at least one state is required");

        checkCommonOrbitals();
        energies.assign(states.size(), -1000.0);

        int L=param.length();
        auto const& first=states.front();
        arma::cx_mat K0 = first.rot * param.Kmat * first.rot.t();
        imp_pos = arma::conv_to<arma::uvec>::from(param.impPos);
        bath_pos = arma::conv_to<arma::uvec>::from(set_diff(L,param.impPos));

        exp_ih = arma::cx_mat(L,L,arma::fill::eye);
        exp_ih.submat(bath_pos,bath_pos)=expIH<cmpx>(K0.submat(bath_pos,bath_pos)*dt);
        arma::vec s=arma::svd(param.Kmat.submat(imp_pos,bath_pos));
        nChannel=(s.empty() || s[0]==0)
                   ? 0
                   : (int)arma::find(s>first.tol*s[0]).eval().size();

        arma::cx_mat K1 = K0.submat(imp_pos,bath_pos)
                        * K0.submat(bath_pos,bath_pos)
                        * arma::cx_double(0,-0.5*dt);
        Kip0=K0;
        Kip0.submat(bath_pos,bath_pos).fill(0.0);
        Kip0.submat(imp_pos,bath_pos)+=K1;
        Kip0.submat(bath_pos,imp_pos)+=K1.t();

        K=Kip0;
        K.rows(imp_pos)=K.rows(imp_pos).eval()*first.rot;
        K.cols(imp_pos)=first.rot.t()*K.cols(imp_pos).eval();
    }

    void iterate(TdvpParam args={})
    {
        checkCommonOrbitals();

        // Rebuild the interaction-picture Hamiltonian in the common basis.
        K=Kip0;
        K.rows(imp_pos)=K.rows(imp_pos).eval()*states.front().rot;
        K.cols(imp_pos)=states.front().rot.t()*K.cols(imp_pos).eval();
        for (auto& state : states)
            state.rot=exp_ih*state.rot;

        applyPlan(states.front().planRepresentative(K,0));
        applyPlan(states.front().planRepresentative(K,1));
        applyPlan(states.front().planActiveRepresentative(K));
        evolveAll(args);
        applyPlan(states.front().planNaturalOrbitals(combinedCc()));
    }

    arma::cx_mat correlator_all(std::size_t state=0) const
    {
        return states.at(state).correlator_all();
    }

    cmpx correlator(int i,int j,std::size_t state=0) const
    {
        return states.at(state).correlator(i,j);
    }

private:
    void checkCommonOrbitals() const
    {
        auto const& first=states.front();
        int L=param.length();
        if (first.sites.length()!=L)
            throw std::invalid_argument("FbrNsDyn: state length does not match the model");

        for (std::size_t n=1; n<states.size(); ++n) {
            auto const& state=states[n];
            if (state.sites.length()!=L || state.nActive!=first.nActive
                || state.imp_size!=first.imp_size || state.spin!=first.spin)
                throw std::invalid_argument("FbrNsDyn: states do not share the same orbital layout");
            if (arma::norm(state.rot-first.rot,"fro")>10*first.tol)
                throw std::invalid_argument("FbrNsDyn: states do not share the same orbital rotation");
            for (int i=1; i<=L; ++i)
                if (state.sites(i)!=first.sites(i))
                    throw std::invalid_argument("FbrNsDyn: states must share the same ITensor site indices");
        }

        double slater_tol=std::max(100*first.tol,1e-10);
        for (int i=first.nActive; i<L; ++i) {
            double occupation=std::real(first.cc(i,i))>0.5 ? 1.0 : 0.0;
            for (auto const& state : states)
                if (std::abs(state.cc(i,i)-occupation)>slater_tol)
                    throw std::invalid_argument("FbrNsDyn: states do not share the same Slater state");
            for (std::size_t n=1; n<states.size(); ++n)
                for (int j=first.nActive; j<L; ++j)
                    if (std::abs(states[n].cc(i,j)-first.cc(i,j))>slater_tol)
                        throw std::invalid_argument("FbrNsDyn: states do not share the same Slater correlator");
        }
    }

    arma::cx_mat combinedCc() const
    {
        arma::cx_mat cc(states.front().cc.n_rows,states.front().cc.n_cols,arma::fill::zeros);
        for (auto const& state : states)
            cc+=state.cc;
        cc/=static_cast<double>(states.size());
        return cc;
    }

    void applyPlan(OrbitalUpdate<cmpx> const& update)
    {
        update.applyAsBasis(K);
        for (auto& state : states)
            state.applyUpdate(update);
    }

    void evolveAll(TdvpParam args)
    {
        int localL=param.nImp()+nChannel;
        auto mpo=fullHamiltonian(K.submat(0,0,localL-1,localL-1));

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
                                      {"MaxSite",localL,
                                       "Truncate",true,
                                       "DoNormalize",true,
                                       "Quiet",true,
                                       "Silent",true,
                                       "NumCenter",2,
                                       "ErrGoal",args.err_goal});
            energies[n]+=state.SlaterEnergy(K);
            state.update_cc();
        }
    }

    itensor::MPO fullHamiltonian(arma::cx_mat const& kin) const
    {
        itensor::AutoMPO h(states.front().sites);
        for (auto i=0; i<param.nImp(); ++i)
            for (auto j=0; j<param.nImp(); ++j)
                if (std::abs(param.Umat(i,j))>1e-15)
                    h+=param.Umat(i,j),"N",i+1,"N",j+1;
        for (int i=0; i<(int)kin.n_rows; ++i)
            for (int j=0; j<(int)kin.n_cols; ++j)
                if (std::abs(kin(i,j))>states.front().tol)
                    h+=kin(i,j),"Cdag",i+1,"C",j+1;
        return itensor::toMPO(h);
    }
};

} // namespace fbr

#endif // FBR_NS_DYN_H
