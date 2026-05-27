#ifndef FBR_GRAPH_H
#define FBR_GRAPH_H

#include <armadillo>
#include <vector>
#include <algorithm>

namespace fbr {

inline std::vector<int> iota(int n)
{
    std::vector<int> all(n);
    for(int i=0;i<n;i++) all[i]=i;
    return all;
}

inline std::vector<int> regspace(int a,int b)
{
    return arma::conv_to<std::vector<int>>::from(arma::regspace(a,b-1));
}

inline std::vector<int> set_diff(int n, std::vector<int> Iset)
{
    std::vector<int> all=iota(n);
    std::sort(Iset.begin(),Iset.end());
    std::vector<int> diff;
    diff.reserve(n-Iset.size());
    std::set_difference(all.begin(),all.end(),
                        Iset.begin(),Iset.end(),back_inserter(diff));
    return diff;
}

namespace graph {

static void set_label(arma::umat const& K, int i0, int label, std::vector<int> &out)
{
    if (out[i0]!=-1) return;
    out.at(i0)=label;
    for(auto i=0; i<(int)K.n_rows; i++)
        if (K(i0,i) != 0) set_label(K,i,label,out);
}

/// return the island label of each index
inline std::vector<int> find_islands(arma::umat const& K)
{
    int L=K.n_rows;
    std::vector<int> out(L,-1);
    int label=0;
    for(auto i=0;i<L;i++)
        if (out[i]==-1) set_label(K,i,label++,out);
    return out;
}

/// return the island label of each index
inline std::vector<int> find_islands(arma::mat const& K, double tol=1e-12)
{
    arma::umat pos = arma::find(arma::abs(K)>tol).eval();
    arma::umat K_bool(size(K), arma::fill::zeros);
    K_bool(pos).fill(1);
    return find_islands(K_bool);
}

} // namespace graph

} // namespace fbr

#endif // FBR_GRAPH_H
