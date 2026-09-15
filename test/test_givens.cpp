#include<catch2/catch.hpp>
#include "fbr/givens_rotation.h"
#include "fbr/orbital_update.h"

using namespace arma;
using namespace std;
using namespace fbr;

TEST_CASE("OrbitalGate transformations match dense matrices", "[orbital_gate]")
{
    arma::arma_rng::set_seed(1701);
    constexpr int L=7;
    constexpr int a=2;
    constexpr int b=3;

    cx_vec pair(2,fill::randn);
    auto givens=GivensRot<cmpx>::create_from_pair(0,pair[0],pair[1],true);
    OrbitalGate<cmpx> gate(a,b,givens);

    cx_mat R(L,L,fill::eye);
    R.submat(a,a,b,b)=givens.matrix();
    cx_mat K=cx_mat(L,L,fill::randn)+imag_1*cx_mat(L,L,fill::randn);
    K=K+K.t();
    cx_mat frame=cx_mat(L,L,fill::randn)+imag_1*cx_mat(L,L,fill::randn);
    cx_mat cc=cx_mat(L,L,fill::randn)+imag_1*cx_mat(L,L,fill::randn);
    cc=cc+cc.t();

    auto K_actual=K;
    auto frame_actual=frame;
    auto cc_actual=cc;
    gate.apply_as_basis(K_actual);
    gate.apply_as_frame(frame_actual);
    gate.apply_as_correlator(cc_actual);

    REQUIRE(abs(K_actual-R.t()*K*R).max()<1e-13);
    REQUIRE(abs(frame_actual-frame*R).max()<1e-13);
    REQUIRE(abs(cc_actual-R.st()*cc*conj(R)).max()<1e-13);

    OrbitalGate<cmpx> swap(1,5);
    cx_mat P(L,L,fill::eye);
    P.swap_cols(1,5);
    K_actual=K;
    frame_actual=frame;
    cc_actual=cc;
    swap.apply_as_basis(K_actual);
    swap.apply_as_frame(frame_actual);
    swap.apply_as_correlator(cc_actual);

    REQUIRE(abs(K_actual-P.t()*K*P).max()<1e-13);
    REQUIRE(abs(frame_actual-frame*P).max()<1e-13);
    REQUIRE(abs(cc_actual-P.st()*cc*conj(P)).max()<1e-13);
}

TEST_CASE("arma") {
    mat A = { {1, 3, 5},
              {2, 4, 6} };
    //A.print("A=");
}

TEST_CASE("index-aware apply_givens matches dense embedding", "[givens]")
{
    arma::arma_rng::set_seed(777);
    const int n = 6;      // local block size
    const int n_sv = 2;    // number of columns to rotate out
    const int L = 14;     // full matrix size

    // Build a Givens list as the dynamics does (daggered left-rotation).
    cx_mat Vfull = cx_mat(n, n, fill::randn) + imag_1 * cx_mat(n, n, fill::randn);
    cx_mat Q, R;
    qr(Q, R, Vfull);
    cx_mat V = Q.head_cols(n_sv);
    auto givens = givens_for_rot_left(V);
    givens_dagger_in_place(givens);

    // Non-contiguous target positions of size n inside [0,L).
    uvec pos = {1, 2, 4, 7, 9, 12};
    REQUIRE(pos.n_elem == (uword)n);

    // Dense embedding E (eye with the rotation block placed at pos).
    cx_mat rot_block = matrot_from_givens(givens, n);
    cx_mat E(L, L, fill::eye);
    E(pos, pos) = rot_block;

    cx_mat A = cx_mat(L, L, fill::randn) + imag_1 * cx_mat(L, L, fill::randn);

    cx_mat Acols = A;
    apply_givens_cols(Acols, givens, pos);           // A * E
    REQUIRE(abs(Acols - A * E).max() < 1e-12);

    cx_mat Arows = A;
    apply_givens_rows(givens, Arows, pos);           // E * A
    REQUIRE(abs(Arows - E * A).max() < 1e-12);

    // Full conjugation E^dag * A * E built from the two primitives.
    cx_mat Aconj = A;
    apply_givens_cols(Aconj, givens, pos);
    apply_givens_rows(givens_dagger(givens), Aconj, pos);
    REQUIRE(abs(Aconj - E.t() * A * E).max() < 1e-12);
}

TEST_CASE( "GivensRotation real" )
{
    arma::arma_rng::set_seed(42);  // deterministic v, independent of test order
    double tol=1e-14;
    // The ilog_matrix -> expmat round-trip goes through an eigendecomposition, whose
    // reconstruction error reaches ~1e-11 for unlucky angles (measured over 2e6 draws),
    // so it needs a looser bound than the exact algebraic checks. A real bug is O(1).
    double tolExp=1e-9;
    vec v(2, fill::randu);
    auto g=GivensRot<>::create_from_pair(0, v[0], v[1], true);

    SECTION( "definition" )
    {
        vec y=g.matrix()*v;
        REQUIRE(std::abs(norm(y)/norm(v)-1)<tol);
        REQUIRE(std::abs(y[0]/y[1])<tol);
    }

    SECTION("ilogmat")
    {
        cx_mat h=g.ilog_matrix();
        REQUIRE(norm(g.matrix()-expmat(h*cmpx(0,-1)))<tolExp);
        REQUIRE(norm(h-h.t())<tol);
    }

    SECTION("3d case")
    {
        vector<GivensRot<>> gs;
        arma::vec v={0.5,1.5,-1}, vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<>::create_from_pair(b,v[i],v[i+1], true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_givens(gs, v.size());
        vec y=rot*vc;
        REQUIRE(std::abs(norm(y)/norm(vc)-1)<tol);
        REQUIRE(std::abs(y[0]/y[2])<tol);
        REQUIRE(std::abs(y[1]/y[2])<tol);
    }

    SECTION("matrix case")
    {
        vector<GivensRot<>> gs;
        arma::mat A(3,3, arma::fill::randu), evec;
        arma::vec eval;
        A = A*A.t();
        arma::eig_sym(eval,evec,A);
        arma::vec v=evec.col(0), vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<>::create_from_pair(b,v[i],v[i+1], true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_givens(gs,A.n_cols);
        REQUIRE(norm(rot*rot.t()-eye(size(rot)))<tol);
        arma::mat Arot=rot*A*rot.t();
        REQUIRE(std::abs(Arot(2,2)/eval(0)-1)<tol*norm(A));
        REQUIRE(std::abs(Arot(0,2)/norm(A))<tol);
        REQUIRE(std::abs(Arot(1,2)/norm(A))<tol);
    }
}

TEST_CASE( "GivensRotation complex" )
{
    arma::arma_rng::set_seed(42);  // deterministic v, independent of test order
    double tol=1e-14;
    double tolExp=1e-9;  // eig-based ilog_matrix round-trip; see "GivensRotation real"
    cx_vec v(2, fill::randu);
    auto g=GivensRot<cmpx>::create_from_pair(0, v[0], v[1], true);

    SECTION( "definition" )
    {
        cx_vec y=g.matrix()*v;
        REQUIRE(std::abs(arma::norm(y)/arma::norm(v)-1)<tol);
        REQUIRE(std::abs(y[0]/y[1])<tol);
    }

    SECTION("ilogmat")
    {
        cx_mat h=g.ilog_matrix();
        REQUIRE(norm(h-h.t())<tol);
        REQUIRE(norm(g.matrix()-exp_iH(h))<tolExp);
    }

    SECTION("3d case")
    {
        vector<GivensRot<cmpx>> gs;
        arma::cx_vec v(3,fill::randu), vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<cmpx>::create_from_pair(b,v[i],v[i+1], true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_givens(gs, v.size());
        cx_vec y=rot*vc;
        REQUIRE(std::abs(norm(y)/norm(vc)-1)<tol);
        REQUIRE(std::abs(y[0]/y[2])<tol);
        REQUIRE(std::abs(y[1]/y[2])<tol);
    }

    SECTION("matrix case")
    {
        vector<GivensRot<cmpx>> gs;
        arma::cx_mat A(3,3, arma::fill::randu), evec;
        arma::vec eval;
        A = A*A.t();
        arma::eig_sym(eval,evec,A);
        arma::cx_vec v=evec.col(0), vc=v;
        for(auto i=0u; i+1<v.size(); i++)
        {
            auto b=i;
            auto g=GivensRot<cmpx>::create_from_pair(b,v[i],v[i+1],true, &v[i+1]);
            gs.push_back(g);
        }
        auto rot=matrot_from_givens(gs,A.n_cols);
        REQUIRE(norm(rot*rot.t()-eye(size(rot)))<tol);
        REQUIRE(norm(rot.t()*rot-eye(size(rot)))<tol);
        arma::cx_mat Arot=rot*A*rot.t();
        REQUIRE(std::abs(Arot(2,2)/eval(0)-1.0)<tol*norm(A));
        REQUIRE(std::abs(Arot(0,2)/norm(A))<tol);
        REQUIRE(std::abs(Arot(1,2)/norm(A))<tol);
    }
}


TEST_CASE( "GivensRotation complex left" )
{
    arma::arma_rng::set_seed(42);  // deterministic v, independent of test order
    double tol=1e-14;
    double tolExp=1e-9;  // eig-based ilog_matrix round-trip; see "GivensRotation real"
    cx_vec v(2, fill::randu);
    auto g=GivensRot<cmpx>::create_from_pair(0, v[0], v[1], false);

    SECTION( "definition" )
    {
        cx_vec y=g.matrix()*v;
        REQUIRE(std::abs(arma::norm(y)/arma::norm(v)-1)<tol);
        REQUIRE(std::abs(y[1]/y[0])<tol);
    }

    SECTION("ilogmat")
    {
        cx_mat h=g.ilog_matrix();
        REQUIRE(norm(h-h.t())<tol);
        REQUIRE(norm(g.matrix()-exp_iH(h))<tolExp);
    }

    SECTION("3d case")
    {
        vector<GivensRot<cmpx>> gs;
        arma::cx_vec v(3,fill::randu), vc=v;
        for(int i=v.size()-2; i>=0; i--)
        {
            auto b=i;
            auto g=GivensRot<cmpx>::create_from_pair(b,v[i],v[i+1], false, &v[i]);
            gs.push_back(g);
        }
        auto rot=matrot_from_givens(gs, v.size());
        cx_vec y=rot*vc;
        REQUIRE(std::abs(norm(y)/norm(vc)-1)<tol);
        REQUIRE(std::abs(y[2]/y[0])<tol);
        REQUIRE(std::abs(y[1]/y[0])<tol);
    }
}

// Helper: build the n×n index-reversal permutation matrix P where P(i, n-1-i) = 1.
// Satisfies P*P = I and (P*A*P)(i,j) = A(n-1-i, n-1-j).
namespace {
template<class T>
arma::Mat<T> reversal_perm(int n)
{
    arma::Mat<T> P(n, n, arma::fill::zeros);
    for (int i = 0; i < n; i++) P(i, n-1-i) = T(1);
    return P;
}
} // namespace

// givens_reflect maps each gate at bond b to bond n-2-b and conjugates the Givens matrix
// (for real: transposes; for complex: applies reflect(L)).
// Key algebraic property (real and complex alike):
//   matrot_from_givens(givens_reflect(givens, n), n)  ==  P * rot1 * P
// where P is the reversal permutation of size n.
TEST_CASE("givens_reflect real", "[givens_reflect]")
{
    const double tol = 1e-14;
    const int n = 6;

    // Build a non-trivial set of Givens via a 3-column SVD
    arma::mat V(n, 3, arma::fill::randu);
    auto givens = givens_for_rot_left(V);
    givens_dagger_in_place(givens);

    arma::mat rot1    = matrot_from_givens(givens, n);
    arma::mat rot1_up = matrot_from_givens(givens_reflect(givens, n), n);

    SECTION("rot1_up is orthogonal") {
        REQUIRE(arma::norm(rot1_up * rot1_up.t() - arma::eye<arma::mat>(n, n)) < tol);
        REQUIRE(arma::norm(rot1_up.t() * rot1_up - arma::eye<arma::mat>(n, n)) < tol);
    }

    SECTION("rot1_up = P * rot1 * P") {
        arma::mat P = reversal_perm<double>(n);
        REQUIRE(arma::norm(rot1_up - P * rot1 * P) < tol);
    }
}

TEST_CASE("givens_reflect complex", "[givens_reflect]")
{
    const double tol = 1e-14;
    const int n = 6;

    arma::cx_mat V(n, 3, arma::fill::randu);
    auto givens = givens_for_rot_left(V);
    givens_dagger_in_place(givens);

    arma::cx_mat rot1    = matrot_from_givens(givens, n);
    arma::cx_mat rot1_up = matrot_from_givens(givens_reflect(givens, n), n);

    SECTION("rot1_up is unitary") {
        REQUIRE(arma::norm(rot1_up * rot1_up.t() - arma::eye<arma::cx_mat>(n, n)) < tol);
        REQUIRE(arma::norm(rot1_up.t() * rot1_up - arma::eye<arma::cx_mat>(n, n)) < tol);
    }

    SECTION("rot1_up = P * rot1 * P") {
        arma::cx_mat P = reversal_perm<cmpx>(n);
        REQUIRE(arma::norm(rot1_up - P * rot1 * P) < tol);
    }
}

TEST_CASE("set of Givens")
{
    SECTION("basic")
    {
        arma::cx_mat X(5,5, fill::randu), U, V;
        vec s;
        svd(U,s,V,X);
        //s.print("s");
        //V.print("V");
        // auto givens=givens_for_rot_right(V,V.n_cols-1);
        auto givens=givens_for_rot_left(V.head_cols(5).eval());
        auto G=matrot_from_givens(givens,V.n_rows).t().eval();
        //G.print("givens");
        //cout<<norm(G.t()*G-eye<decltype(X)>(size(V)))<<endl;
    }

    SECTION("kin")
    {
        int len=8;
        cx_mat x(len,len,fill::randu), U, V;
        cx_mat kin= (x.t()*x).eval();
        kin.submat(2,2,len-1,len-1).fill(0);
        auto k12=kin.submat(0,2,1,len-1).eval();
        vec s;
        svd_econ(U,s,V,k12);
        auto givens=givens_for_rot_left(V.head_cols(2).eval());
        for(auto& g:givens) g.b+=2;
        givens_dagger_in_place(givens);

        kin.print("kin");
        SECTION("using global rot")
        {
            cx_mat rot(len,len,fill::eye);
            rot=matrot_from_givens(givens,len);
            (rot.t()*kin*rot).eval().clean(1e-13).print("kin after rot f");
        }
        SECTION("using gates")
        {
            auto k1=kin;
            apply_givens(k1,givens);
            apply_givens(givens_dagger(givens),k1);
            k1.clean(1e-13).print("kin after rot f with Givens");
        }
    }

    SECTION("diagonalize")
    {
        int len=8;
        cx_mat x(len,len,fill::randn);
        cx_mat A= (x.t()*x).eval();
        vec eval;
        cx_mat evec;
        eig_sym(eval,evec,A);
        SECTION("left stair") {
            auto givens=givens_for_rot_left(evec.head_cols(2).eval());
            givens_dagger_in_place(givens);

            eval.as_row().eval().print("eval");
            auto k1=A;
            apply_givens(k1,givens);
            apply_givens(givens_dagger(givens),k1);
            k1.clean(1e-13).print("A after rot with Givens");
        }
        SECTION("right stair") {
            auto givens=givens_for_rot_right(evec.head_cols(3).eval());
            givens_dagger_in_place(givens);

            eval.as_row().eval().print("eval");
            auto k1=A;
            apply_givens(k1,givens);
            apply_givens(givens_dagger(givens),k1);
            k1.clean(1e-13).print("A after rot with Givens");
        }
    }

    SECTION( "sparse" )
    {
        int L=10;
        arma::mat A(L,L);
        A.diag().randn();
        // arma::mat(A).print("before Givens");
        arma::mat U, V, k12(1,L-2,arma::fill::randu);
        vec s;
        svd(U,s,V,k12);
        auto givens=givens_for_rot_left(V.head_cols(1).eval());
        for(auto &g:givens) g.b+=2;
        //givens_dagger_in_place(givens);
        for(auto &g:givens) {
//            std::cout<<"gate "<<g.b<<" "<<g.b+1<<std::endl;
            apply_givens(g,A);
//            arma::mat(A).print("after Givens");
        }
    }

}
