#ifndef FBR_INITIAL_STATE_H
#define FBR_INITIAL_STATE_H

#include "impurity_param.h"
#include "fb_mps.h"

namespace fbr {

/// The Slater state a model starts from: the model's own frame, filling and
/// layout, so the state cannot disagree with the model it will be evolved with.
///
/// ek is the energy of every orbital, deciding which ones are filled; it
/// defaults to the diagonal of Kmat. Pass your own to force an occupation, e.g.
/// ek[i]=-10 fills orbital i and ek[i]=+10 empties it.
template<class T=double>
Fb_mps<T> slater(ImpurityParam const& param, arma::vec ek={})
{
    if (ek.empty()) ek=arma::vec {param.Kmat.diag()};
    return Fb_mps<T>::from_slater(arma::conv_to<arma::Mat<T>>::from(param.rot), ek,
                                  param.n_part(), param.n_imp(), param.layout);
}

} // namespace fbr

#endif // FBR_INITIAL_STATE_H
