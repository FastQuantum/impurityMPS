#ifndef FBR_ORBITAL_UPDATE_H
#define FBR_ORBITAL_UPDATE_H

#include "givens_rotation.h"

#include <complex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace fbr {

/// One elementary change of orbital basis acting on absolute orbitals a and b.
template<class T>
struct OrbitalGate {
    int a;
    int b;
    T c=1;
    T s=0;
    bool swap=false;

    OrbitalGate(int a_,int b_,GivensRot<T> const& g)
        : a(a_), b(b_), c(g.c), s(g.s) {}

    OrbitalGate(int a_,int b_)
        : a(a_), b(b_), swap(true) {}

    GivensRot<T> givens(int bond=0) const
    {
        if (swap)
            throw std::logic_error("a swap is not a Givens rotation");
        return {.b=(size_t)bond,.c=c,.s=s};
    }

    /// Apply K -> R^dagger K R, or swap its rows and columns.
    void apply_as_basis(arma::Mat<T>& K) const
    {
        if (swap) {
            K.swap_cols(a,b);
            K.swap_rows(a,b);
            return;
        }

        auto g=givens();
        apply_right(K,g);
        apply_left(g.dagger(),K);
    }

    /// Apply rot -> rot R, or swap its columns.
    void apply_as_frame(arma::Mat<T>& rot) const
    {
        if (swap) {
            rot.swap_cols(a,b);
            return;
        }
        apply_right(rot,givens());
    }

    /// Apply cc -> R^T cc R*, or swap its rows and columns.
    void apply_as_correlator(arma::Mat<T>& cc) const
    {
        if (swap) {
            cc.swap_cols(a,b);
            cc.swap_rows(a,b);
            return;
        }

        auto m=givens().matrix();
        arma::Col<T> ca=cc.col(a), cb=cc.col(b);
        cc.col(a)=ca*conjugate(m(0,0))+cb*conjugate(m(1,0));
        cc.col(b)=ca*conjugate(m(0,1))+cb*conjugate(m(1,1));

        arma::Row<T> ra=cc.row(a), rb=cc.row(b);
        cc.row(a)=m(0,0)*ra+m(1,0)*rb;
        cc.row(b)=m(0,1)*ra+m(1,1)*rb;
    }

private:
    static T conjugate(T const& x)
    {
        if constexpr (std::is_arithmetic<T>::value) return x;
        else return std::conj(x);
    }

    void apply_right(arma::Mat<T>& matrix,GivensRot<T> const& g) const
    {
        auto m=g.matrix();
        arma::Col<T> ca=matrix.col(a), cb=matrix.col(b);
        matrix.col(a)=ca*m(0,0)+cb*m(1,0);
        matrix.col(b)=ca*m(0,1)+cb*m(1,1);
    }

    void apply_left(GivensRot<T> const& g,arma::Mat<T>& matrix) const
    {
        auto m=g.matrix();
        arma::Row<T> ra=matrix.row(a), rb=matrix.row(b);
        matrix.row(a)=m(0,0)*ra+m(0,1)*rb;
        matrix.row(b)=m(1,0)*ra+m(1,1)*rb;
    }
};

/// Ordered orbital gates and the resulting active interval [a,b).
template<class T>
struct OrbitalUpdate {
    std::vector<OrbitalGate<T>> gates;
    std::pair<int,int> active;

    OrbitalUpdate(int a,int b) : active{a,b} {}

    /// Convert local Givens bonds into absolute orbital pairs. The stored order
    /// is the order in which the basis rotations act on matrices.
    void append(arma::uvec const& orbitals,
                std::vector<GivensRot<T>> const& givens)
    {
        for (auto it=givens.crbegin(); it!=givens.crend(); ++it)
            gates.emplace_back((int)orbitals[it->b],
                               (int)orbitals[it->b+1],*it);
    }

    void apply_as_basis(arma::Mat<T>& K) const
    {
        for (auto const& gate : gates) gate.apply_as_basis(K);
    }
};

} // namespace fbr

#endif // FBR_ORBITAL_UPDATE_H
