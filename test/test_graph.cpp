#include<catch2/catch.hpp>
#include "fbr/graph.h"

using namespace arma;
using namespace std;
using namespace fbr;
using namespace fbr::graph;

bool same_component(vector<int> labels,int a, int b) { return labels[a]==labels[b]; }

TEST_CASE("Single node graph", "[graph]")
{
    arma::mat K(1, 1, fill::value(1.0));
    auto labels = find_islands(K);
    REQUIRE (labels == vector {0});
}

TEST_CASE("Two disconnected nodes", "[graph]")
{
    arma::mat K = {
        {0.0, 0.0},
        {0.0, 0.0}
    };
    auto labels = find_islands(K);
    REQUIRE(labels == vector{0,1});
}

TEST_CASE("Two connected nodes", "[graph]")
{
    arma::mat K = {
        {0.0, 1.0},
        {1.0, 0.0}
    };
    auto labels = find_islands(K);
    REQUIRE(labels == vector{0,0});
}

TEST_CASE("Fully connected 4-node graph", "[graph]")
{
    arma::mat K = arma::ones<arma::mat>(4, 4);
    K.diag().zeros();   // no self-loops
    auto labels = find_islands(K);
    REQUIRE(labels == vector{0,0,0,0});
}

TEST_CASE("Three isolated nodes", "[graph]")
{
    arma::mat K = arma::zeros<arma::mat>(3, 3);
    auto labels = find_islands(K);
    REQUIRE(labels == vector {0,1,2});
}

TEST_CASE("Two separate components of size 2", "[graph]")
{
    // 0-1   2-3  (no edge between the pairs)
    arma::mat K = arma::zeros<arma::mat>(4, 4);
    K(0,1) = K(1,0) = 1.0;
    K(2,3) = K(3,2) = 1.0;

    auto labels = find_islands(K);
    REQUIRE(labels == vector{0,0,1,1});
}

TEST_CASE("Near-zero edge weights are ignored", "[graph]")
{
    // Edge weight below threshold should not connect the nodes
    arma::mat K = {
        {0.0,  1e-13},
        {1e-13, 0.0 }
    };
    auto labels = find_islands(K);
    REQUIRE(labels == vector{0,1});
}

TEST_CASE("Edge weight above threshold is treated as present", "[graph]")
{
    // 1e-11 > 1e-12, so the edge should connect the nodes
    arma::mat K = {
        {0.0,  1e-11},
        {1e-11, 0.0 }
    };
    auto labels = find_islands(K);
    REQUIRE(labels== vector{0,0});
}

TEST_CASE("Chain graph 0-1-2-3", "[graph]")
{
    arma::mat K = arma::zeros<arma::mat>(4, 4);
    for (int i = 0; i < 3; i++)
        K(i, i+1) = K(i+1, i) = 1.0;

    auto labels = find_islands(K);
    REQUIRE(labels == vector{0,0,0,0}); // ends of the chain are connected
}