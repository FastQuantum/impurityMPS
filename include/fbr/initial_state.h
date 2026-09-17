#ifndef FBR_INITIAL_STATE_H
#define FBR_INITIAL_STATE_H

#include "impurity_param.h"
#include "fb_mps.h"

namespace fbr {

/// The Slater state a model starts from: the model's own frame, filling and
/// geometry, so the state cannot disagree with the model it will be evolved with.
///
/// A model built directly in star geometry may omit rot (identity) and Umat
/// (zero interaction). Its impurity positions must already match the geometry.
///
/// ek is the energy of every orbital, deciding which ones are filled; it
/// defaults to the diagonal of Kmat. Pass your own to force an occupation, e.g.
/// ek[i]=-10 fills orbital i and ek[i]=+10 empties it.
template<class T=double>
Fb_mps<T> slater(ImpurityParam const& param, arma::vec ek={})
{
    param.validate();
    int L=param.length();
    int impurity_begin=param.geometry==standard ? 0 : L/2-param.n_imp()/2;
    for (int i=0; i<param.n_imp(); ++i)
        if (param.imp_pos[i]!=impurity_begin+i)
            throw std::invalid_argument("slater: impurity positions must match the geometry; call to_star first");
    if (!ek.empty() && ((int)ek.n_elem!=L || !ek.is_finite()))
        throw std::invalid_argument("slater: ek must contain L finite energies");
    if (ek.empty()) ek=arma::vec {param.Kmat.diag()};
    arma::Mat<T> rot=param.rot.empty() ? arma::Mat<T>(L,L,arma::fill::eye)
                                      : arma::conv_to<arma::Mat<T>>::from(param.rot);
    return Fb_mps<T>::from_slater(rot, ek,
                                  param.n_part(), param.n_imp(), param.geometry);
}

} // namespace fbr

#endif // FBR_INITIAL_STATE_H
