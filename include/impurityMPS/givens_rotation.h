#ifndef GIVENS_ROTATION_H
#define GIVENS_ROTATION_H


#include<armadillo>
#include <itensor/all.h>

using cmpx=std::complex<double>;
const cmpx imag_1 = {0.0, 1.0};

/// compute the exp(-i H) assuming H is Hermitian
template<class T>
arma::cx_mat expIH(arma::Mat<T> const& H)
{
    arma::Mat<T> evec;
    arma::vec eval;
    arma::eig_sym(eval,evec,H);
    return evec * arma::diagmat(arma::exp(-eval*imag_1)) * evec.t();
}


template<class T>
std::pair<arma::cx_vec,arma::cx_mat> eig_unitary(const arma::Mat<T>& A)
{
    using namespace arma;
    cx_vec eval, eval2;
    cx_mat evec, Q, R;
    eig_gen(eval, evec, A);
    qr(Q, R, evec);
    cx_mat RDR=R*diagmat(eval)*R.i();
    eval2=RDR.diag();

#ifndef NDEBUG
        double err=norm(A-Q*diagmat(eval2)*Q.t());
        std::cout<<"error diag RDR="<<norm(RDR-diagmat(eval))<<std::endl;
        std::cout<<"err eig_unitary="<<err<<std::endl;
        std::cout<<"err |eval|-1="<<norm(arma::abs(eval2)-ones(A.n_cols))<<std::endl;
#endif

    return {eval2,Q};
}

/// assuming n_rows < n_col for matrix K, and even/odd positions for spin up/down
template<class T>
void svd_spin(arma::Mat<T> &U, arma::vec &s, arma::Mat<T>&V, arma::Mat<T> const& K)
{
    int L1=K.n_rows;
    int L2=K.n_cols;
    U=arma::Mat<T>(L1,L1,arma::fill::zeros);
    s=arma::vec(L1);
    V=arma::Mat<T>(L2,L1,arma::fill::zeros);
    for(auto spin : {0,1}) {
        arma::uvec pos1(L1/2);
        for(auto i=0u; i<pos1.size(); i++) pos1[i]=2*i+spin;
        arma::uvec pos2(L2/2);
        for(auto i=0u; i<pos2.size(); i++) pos2[i]=2*i+spin;
        auto k12=K.submat(pos1,pos2).eval();
        arma::vec s0;
        arma::Mat<T> U0, V0;
        arma::svd_econ(U0,s0,V0,k12);
        U(pos1,pos1)=U0;
        s.rows(pos1)=s0;
        V(pos2,pos1)=V0;
    }
}

template<class T>
void my_svd(arma::Mat<T> &U, arma::vec &s, arma::Mat<T>&V, arma::Mat<T> const& K, bool spin)
{
    if (spin) svd_spin(U,s,V,K);
    else arma::svd_econ(U,s,V,K);
}

/// assuming even/odd for spin up/down
template<class T>
void eig_sym_spin(arma::vec &eval, arma::Mat<T> &evec, arma::Mat<T> const& A)
{
    eval=arma::vec(A.n_rows);
    evec=arma::Mat<T>(arma::size(A), arma::fill::zeros);
    for(auto spin : {0,1}) {
        int L_spin=A.n_rows/2;
        arma::uvec pos0(L_spin);
        for(auto i=0; i<L_spin; i++) pos0[i]=2*i+spin;
        auto A0=A.submat(pos0,pos0).eval();
        arma::vec eval0;
        arma::Mat<T> evec0;
        arma::eig_sym(eval0,evec0,A0);
        eval(pos0)=eval0;
        evec(pos0,pos0)=evec0;
    }
}

template<class T>
void my_eig_sym(arma::vec &eval, arma::Mat<T> &evec, arma::Mat<T> const& A, bool spin)
{
    if (spin) eig_sym_spin(eval,evec,A);
    else arma::eig_sym(eval,evec,A);
}

inline arma::uvec sort_index_spin(arma::vec const& x)
{
    return arma::stable_sort_index(x);
    arma::uvec idx(x.size());
    for(auto spin : {0,1}) {
        int L_spin=x.size()/2;
        arma::uvec pos0(L_spin);
        for(auto i=0; i<L_spin; i++) pos0[i]=2*i+spin;
        auto x0=x(pos0);
        arma::uvec idx0=arma::sort_index(x0);
        idx(pos0)=idx0*2+spin;
    }
    return idx;
}

inline arma::uvec my_sort_index(arma::vec const& x, bool spin)
{
    if (spin) return sort_index_spin(x);
    else return arma::sort_index(x);
}


///This class represents a Givens rotation
/// @see https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1JacobiRotation.html
template<class T=double>
struct GivensRot {
    using matrix22=typename arma::Mat<T>::template fixed<2,2>;

    size_t b;     ///< bond b --- b+1
    T c=1, s=0;  ///< cos, sin, radius

    //GivensRot(size_t b_) : b(b_) {}

    /// build the J s.t.  J * (p,q)=(0,r) is go_right=true. Adapted from eigen.tuxfamily.org
    static GivensRot<T> createFromPair(size_t b, T p,  T q, bool go_right, T* r=nullptr);

    //double angle() const { return atan2(s,c); }

    matrix22 matrix() const;

    /// the underline "Hamiltonian" the output is Hermitian.
    arma::cx_mat ilogMatrix() const
    {
        matrix22 rot=matrix();
        auto [eval,evec]=eig_unitary(rot);
        arma::vec eval2=arma::real( arma::log(eval)*cmpx(0,1) );
        return evec * arma::diagmat(eval2) * evec.t();
    }

    /// assuming that |z|=1 ??
    GivensRot<cmpx> operator*(cmpx z) const { GivensRot<cmpx> g{.b=b}; g.c=c*z; g.s=s*z; return g;}

    /// return transpose conjugate
    GivensRot<T> dagger() const;

    /// return transpose
    GivensRot<T> transpose() const;

    /// return reflection wrt L: the bond b (sites b, b+1) is mapped to bond L-2-b
    /// (sites L-2-b, L-1-b), and the rotation is conjugated by the 2x2 swap P=[[0,1],[1,0]].
    GivensRot<T> reflect(int L) const;
};

template<class T>
void applyGivens(GivensRot<T> const& g, arma::Mat<T>& A)
{
    auto Ar=A.rows(g.b,g.b+1).eval();
    A.rows(g.b,g.b+1)=g.matrix()*Ar;
}

template<class T>
void applyGivens(arma::Mat<T>& A, GivensRot<T> const& g)
{
    auto Ac=A.cols(g.b,g.b+1).eval();
    A.cols(g.b,g.b+1) = Ac * g.matrix();
}


template<>
inline GivensRot<double> GivensRot<double>::createFromPair(size_t b, double p,  double q, bool go_right, double *r)
{
    using Scalar=double;
    using std::sqrt;
    using std::abs;

    GivensRot<double> g {.b=b};
    if (go_right) std::swap(p,q); // to eliminate the p instead of q.
    if(q==Scalar(0))
    {
        g.c = p<Scalar(0) ? Scalar(-1) : Scalar(1);
        g.s = Scalar(0);
        if (r) *r = abs(p);
    }
    else if(p==Scalar(0))
    {
        g.c = Scalar(0);
        g.s = q<Scalar(0) ? Scalar(1) : Scalar(-1);
        if (r) *r = abs(q);
    }
    else if(abs(p) > abs(q))
    {
        Scalar t = q/p;
        Scalar u = sqrt(Scalar(1) + t*t);
        if(p<Scalar(0))
            u = -u;
        g.c = Scalar(1)/u;
        g.s = -t * g.c;
        if (r) *r = p * u;
    }
    else
    {
        Scalar t = p/q;
        Scalar u = sqrt(Scalar(1) + t*t);
        if(q<Scalar(0))
            u = -u;
        g.s = -Scalar(1)/u;
        g.c = -t * g.s;
        if (r) *r = q * u;
    }
    if (go_right) { g.s=-g.s; }
    return g;
}

template<>
inline GivensRot<double>::matrix22 GivensRot<double>::matrix() const { return {{c, -s},{s, c}}; }

template<>
inline GivensRot<double> GivensRot<double>::dagger() const { return {.b=b, .c=c, .s=-s}; }

template<>
inline GivensRot<double> GivensRot<double>::transpose() const { return {.b=b, .c=c, .s=-s}; }

template<>
inline GivensRot<double> GivensRot<double>::reflect(int L) const { return {.b=L-2-b, .c=c, .s=-s}; }

template<>
inline GivensRot<cmpx> GivensRot<cmpx>::createFromPair(size_t b, cmpx p, cmpx q, bool go_right, cmpx *r)
{
    using Scalar=cmpx;
    using RealScalar=double;
    using std::sqrt;
    using std::abs;
    using std::conj;

    GivensRot<cmpx> g {.b=b};
    if (go_right) std::swap(p,q); // to eliminate the p instead of q.
    if(q==Scalar(0))
    {
        g.c = std::real(p)<0 ? Scalar(-1) : Scalar(1);
        g.s = 0;
        if (r) *r = g.c * p;
    }
    else if(p==Scalar(0))
    {
        g.c = 0;
        g.s = -q/abs(q);
        if (r) *r = abs(q);
    }
    else
    {
        RealScalar p1 = std::abs(p);
        RealScalar q1 = std::abs(q);
        if(p1>=q1)
        {
            Scalar ps = p / p1;
            RealScalar p2 = std::norm(ps);
            Scalar qs = q / p1;
            RealScalar q2 = std::norm(qs);

            RealScalar u = sqrt(RealScalar(1) + q2/p2);
            if(std::real(p)<RealScalar(0))
                u = -u;

            g.c = Scalar(1)/u;
            g.s = -qs*conj(ps)*(g.c/p2);
            if (r) *r = p * u;
        }
        else
        {
            Scalar ps = p / q1;
            RealScalar p2 = std::norm(ps);
            Scalar qs = q / q1;
            RealScalar q2 = std::norm(qs);

            RealScalar u = q1 * sqrt(p2 + q2);
            if(std::real(p)<RealScalar(0))
                u = -u;

            p1 = abs(p);
            ps = p/p1;
            g.c = p1/u;
            g.s = -conj(ps) * (q/u);
            if (r) *r = ps * u;
        }
    }
    if (go_right) { g.s=-conj(g.s); } // assuming g.c is real!!

    return g;
}

template<>
inline GivensRot<cmpx>::matrix22 GivensRot<cmpx>::matrix() const { return {{std::conj(c), -std::conj(s)},{s, c}}; }

template<>
inline GivensRot<cmpx> GivensRot<cmpx>::dagger() const { return {.b=b, .c=std::conj(c), .s=-s}; }

template<>
inline GivensRot<cmpx> GivensRot<cmpx>::transpose() const { return {.b=b, .c=c, .s=-std::conj(s)}; }

template<>
inline GivensRot<cmpx> GivensRot<cmpx>::reflect(int L) const { return {.b=L-2-b, .c=std::conj(c), .s=-std::conj(s)}; }


//------------------------- set of Givens rotations -----------------------------------------

template<class T>
void applyGivens(std::vector<GivensRot<T>> const& gs,arma::Mat<T>& A)
{
    for(auto const& g:gs)
        applyGivens(g,A);
}

template<class T>
void applyGivens(arma::Mat<T>& A, std::vector<GivensRot<T>> const& gs)
{
    for(auto it=gs.crbegin(); it!=gs.crend(); ++it)
        applyGivens(A,*it);
}


template<class T>
arma::Mat<T> matrot_from_Givens(std::vector<GivensRot<T>> const& gates, size_t n)
{
    if (n==0) { // read the length from the gates
        for(const GivensRot<T>& g : gates) if (g.b>n) n=g.b;
        n+=2;
    }
    arma::Mat<T> rot(n,n, arma::fill::eye);
    for(int i=gates.size()-1; i>=0; i--) { // apply to the right in reverse
        const GivensRot<T>& g=gates[i];
        rot.cols(g.b,g.b+1) = rot.cols(g.b,g.b+1).eval() * g.matrix();
    }
    return rot;
}


/// generate the corresponding Givens rotations: every column is one (left-stair-like) layer of gates
template<class T>
static std::vector<GivensRot<T>> GivensRotForRot_left(arma::Mat<T> rot)
{
    std::vector<GivensRot<T>> givens;
    for(int j=0u; j<rot.n_cols; j++) {
        arma::Col<T> v=rot.col(j);
        std::vector<GivensRot<T>> gs1;
        for(int i=v.size()-2; i>=j; i--)
        {
            auto g=GivensRot<T>::createFromPair(i,v[i],v[i+1], false, &v[i]);
            gs1.push_back(g);
        }
        applyGivens(gs1,rot);
        for(auto g : gs1) givens.push_back(g);
    }
    return givens;
}

/// generate the corresponding Givens rotations: every column is one (right-stair-like) layer of gates
template<class T>
static std::vector<GivensRot<T>> GivensRotForRot_right(arma::Mat<T> rot)
{
    std::vector<GivensRot<T>> givens;
    for(int j=0u; j<rot.n_cols; j++) {
        arma::Col<T> v=rot.col(j);
        std::vector<GivensRot<T>> gs1;
        for(int i=0u; i+1+j<v.size(); i++)
        {
            auto g=GivensRot<T>::createFromPair(i,v[i],v[i+1], true, &v[i+1]);
            gs1.push_back(g);
        }
        applyGivens(gs1,rot);
        for(auto g : gs1) givens.push_back(g);
    }
    return givens;
}

// return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
template<class T>
std::vector<GivensRot<T>> GivensRotForCC_right(arma::Mat<T> cc, int depth=-1, int pfinal=-1)
{
    if (pfinal==-1 || pfinal>cc.n_rows-1) pfinal=cc.n_rows-1;
    if (depth==-1) depth=cc.n_rows;
    using namespace arma;
    std::vector<GivensRot<T>> givens;
    arma::Mat<T> evec;
    arma::vec eval;
    for(auto p2=pfinal; p2>0u; p2--) {
        size_t p1= (p2+1>depth) ? p2+1-depth : 0u ;
        if(p2+depth>pfinal) p1=0;
        arma::Mat<T> cc2=cc.submat(p1,p1,p2,p2);
        arma::eig_sym(eval,evec,cc2);
        // select the less active
        size_t pos=0;
        if (1-eval.back()<eval(0)) pos=eval.size()-1;
        arma::Col<T> v=evec.col(pos);
        std::vector<GivensRot<T>> gs1;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i+p1;
            auto g=GivensRot<T>::createFromPair(b,v[i],v[i+1],true, &v[i+1]);
            gs1.push_back(g);
        }
        auto rot1=matrot_from_Givens(gs1,p2+1);
        cc.submat(0,0,p2,p2)=rot1*cc.submat(0,0,p2,p2)*rot1.t();
        for(auto g : gs1) givens.push_back(g);
    }
    return givens;
}

// return a list of local 2-site gates: see fig5a of PRB 92, 075132 (2015)
template<class T>
std::vector<GivensRot<T>> GivensRotForCC_right_inactive(arma::Mat<T> cc, double tol)
{
    int pfinal=cc.n_rows-1;
    int depth=cc.n_rows;
    using namespace arma;
    std::vector<GivensRot<T>> givens;
    arma::Mat<T> evec;
    arma::vec eval;
    for(auto p2=pfinal; p2>0u; p2--) {
        size_t p1= (p2+1>depth) ? p2+1-depth : 0u ;
        if(p2+depth>pfinal) p1=0;
        arma::Mat<T> cc2=cc.submat(p1,p1,p2,p2);
        arma::eig_sym(eval,evec,cc2);
        // select the less active
        size_t pos=0;
        if (1-eval.back()<eval(0)) pos=eval.size()-1;
        if (eval(pos)>tol && eval(pos)<1-tol) break;
        arma::Col<T> v=evec.col(pos);
        std::vector<GivensRot<T>> gs1;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i+p1;
            auto g=GivensRot<T>::createFromPair(b,v[i],v[i+1],true, &v[i+1]);
            gs1.push_back(g);
        }
        auto rot1=matrot_from_Givens(gs1,p2+1);
        cc.submat(0,0,p2,p2)=rot1*cc.submat(0,0,p2,p2)*rot1.t();
        for(auto g : gs1) givens.push_back(g);
    }
    return givens;
}

template<class T>
void GivensDaggerInPlace(std::vector<GivensRot<T>> &givens)
{
    for(auto& g:givens) g=g.dagger();
    std::reverse(givens.begin(),givens.end());
}

template<class T>
std::vector<GivensRot<T>> GivensDagger(std::vector<GivensRot<T>> const& givens)
{
    auto out=givens;
    GivensDaggerInPlace(out);
    return out;
}

template<class T>
std::vector<GivensRot<T>> GivensTranspose(std::vector<GivensRot<T>> givens)
{
    std::reverse(givens.begin(),givens.end());
    for(auto& g:givens) g=g.transpose();
    return givens;
}

template<class T>
std::vector<GivensRot<T>> GivensReflect(std::vector<GivensRot<T>> givens, int L)
{
    for(auto& g:givens) g=g.reflect(L);
    return givens;
}


#endif // GIVENS_ROTATION_H
