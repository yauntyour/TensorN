#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../Tensor.hpp"
#include <cmath>
#include <vector>

using namespace TensorN;
using Catch::Approx;

// ============================================================
// Tensor Construction
// ============================================================

TEST_CASE("Tensor default construction", "[tensor]")
{
    Tensor<int> t;
    REQUIRE(t.size() == 0);
    REQUIRE(t.shape().empty());
    REQUIRE(t.data.empty());
}

TEST_CASE("Tensor shape construction", "[tensor]")
{
    Tensor<int> t({3, 4});
    REQUIRE(t.size() == 12);
    REQUIRE(t.shape().size() == 2);
    REQUIRE(t.shape()[0] == 3);
    REQUIRE(t.shape()[1] == 4);
    REQUIRE(t.data.size() == 12);
}

TEST_CASE("Tensor scalar construction (empty shape)", "[tensor]")
{
    Tensor<int> t({}, {42});
    REQUIRE(t.size() == 1);
    REQUIRE(t.shape().empty());
    REQUIRE(t.data[0] == 42);
}

TEST_CASE("Tensor shape+data construction", "[tensor]")
{
    Tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
    REQUIRE(t.size() == 6);
    REQUIRE(t[{0, 0}] == 1);
    REQUIRE(t[{0, 2}] == 3);
    REQUIRE(t[{1, 0}] == 4);
    REQUIRE(t[{1, 2}] == 6);
}

TEST_CASE("Tensor shape/data mismatch throws", "[tensor]")
{
    REQUIRE_THROWS_AS((Tensor<int>({2, 3}, {1, 2, 3})), std::invalid_argument);
}

TEST_CASE("Tensor copy construction", "[tensor]")
{
    Tensor<int> a({2, 2}, {1, 2, 3, 4});
    Tensor<int> b(a);
    REQUIRE(b == a);
    b[{0, 0}] = 99;
    REQUIRE(a[{0, 0}] == 1); // original unchanged
}

TEST_CASE("Tensor move construction", "[tensor]")
{
    Tensor<int> a({2, 2}, {1, 2, 3, 4});
    Tensor<int> b(std::move(a));
    REQUIRE(b.size() == 4);
    REQUIRE(b[{0, 0}] == 1);
    REQUIRE(a.size() == 0);
    REQUIRE(a.shape().empty());
    REQUIRE(a.data.empty());
}

TEST_CASE("Tensor is_isomorphic", "[tensor]")
{
    Tensor<int> a({2, 3});
    Tensor<int> b({2, 3});
    Tensor<int> c({3, 2});
    REQUIRE(a.is_isomorphic(b));
    REQUIRE_FALSE(a.is_isomorphic(c));
}

TEST_CASE("Tensor multi-index access", "[tensor]")
{
    Tensor<int> t({2, 3, 4});
    t[{1, 2, 3}] = 42;
    REQUIRE(t[{1, 2, 3}] == 42);
}

TEST_CASE("Tensor index out of range", "[tensor]")
{
    Tensor<int> t({2, 3});
    REQUIRE_THROWS_AS((t[{2, 0}]), std::out_of_range);
    REQUIRE_THROWS_AS((t[{0, 3}]), std::out_of_range);
}

// ============================================================
// Element-wise Arithmetic
// ============================================================

TEST_CASE("Tensor += tensor", "[tensor][arithmetic]")
{
    Tensor<int> a({2, 2}, {1, 2, 3, 4});
    Tensor<int> b({2, 2}, {5, 6, 7, 8});
    a += b;
    REQUIRE(a[{0, 0}] == 6);
    REQUIRE(a[{1, 1}] == 12);
}

TEST_CASE("Tensor + scalar", "[tensor][arithmetic]")
{
    Tensor<int> a({2, 2}, {1, 2, 3, 4});
    auto result = a + 10;
    REQUIRE(result[{0, 0}] == 11);
    REQUIRE(result[{1, 1}] == 14);
}

TEST_CASE("Tensor * scalar", "[tensor][arithmetic]")
{
    Tensor<int> a({2, 2}, {1, 2, 3, 4});
    auto result = a * 3;
    REQUIRE(result[{0, 0}] == 3);
    REQUIRE(result[{1, 1}] == 12);
}

TEST_CASE("Tensor + tensor (opt)", "[tensor][arithmetic]")
{
    Tensor<int> a({2, 2}, {1, 2, 3, 4});
    Tensor<int> b({2, 2}, {10, 20, 30, 40});
    auto result = a + b;
    REQUIRE(result[{0, 0}] == 11);
    REQUIRE(result[{1, 1}] == 44);
}

// ============================================================
// Factory Functions
// ============================================================

TEST_CASE("zeros", "[factory]")
{
    auto t = zeros<float>({3, 3});
    REQUIRE(t.size() == 9);
    for (auto v : t.data)
        REQUIRE(v == 0.0f);
}

TEST_CASE("ones", "[factory]")
{
    auto t = ones<int>({2, 4});
    REQUIRE(t.size() == 8);
    for (auto v : t.data)
        REQUIRE(v == 1);
}

TEST_CASE("eye", "[factory]")
{
    auto t = eye<double>(3);
    REQUIRE(t.size() == 9);
    REQUIRE(t[{0, 0}] == 1.0);
    REQUIRE(t[{1, 1}] == 1.0);
    REQUIRE(t[{2, 2}] == 1.0);
    REQUIRE(t[{0, 1}] == 0.0);
    REQUIRE(t[{1, 0}] == 0.0);
}

TEST_CASE("arange", "[factory]")
{
    auto t = arange<int>(0, 5, 1);
    REQUIRE(t.size() == 5);
    REQUIRE(t[0] == 0);
    REQUIRE(t[4] == 4);
}

// ============================================================
// Einsum - Basic Operations
// ============================================================

TEST_CASE("einsum: matrix multiply", "[einsum]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({3, 2}, {7, 8, 9, 10, 11, 12});

    auto C = einsum<int>("ij,jk->ik", A, B);
    REQUIRE(C.tensor.shape().size() == 2);
    REQUIRE(C.tensor.shape()[0] == 2);
    REQUIRE(C.tensor.shape()[1] == 2);
    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    REQUIRE(C.tensor[{0, 0}] == 58);
    // C[0,1] = 1*8 + 2*10 + 3*12 = 64
    REQUIRE(C.tensor[{0, 1}] == 64);
    // C[1,0] = 4*7 + 5*9 + 6*11 = 139
    REQUIRE(C.tensor[{1, 0}] == 139);
    // C[1,1] = 4*8 + 5*10 + 6*12 = 154
    REQUIRE(C.tensor[{1, 1}] == 154);
}

TEST_CASE("einsum: vector dot product", "[einsum]")
{
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({3}, {4, 5, 6});

    auto result = einsum<int>("i,i->", a, b);
    REQUIRE(result.tensor.size() == 1);
    // 1*4 + 2*5 + 3*6 = 32
    REQUIRE(result.tensor[0] == 32);
}

TEST_CASE("einsum: outer product", "[einsum]")
{
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({4}, {4, 5, 6, 7});

    auto result = einsum<int>("i,j->ij", a, b);
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor.shape()[1] == 4);
    REQUIRE(result.tensor[{0, 0}] == 4);
    REQUIRE(result.tensor[{2, 3}] == 21);
}

TEST_CASE("einsum: element-wise multiply", "[einsum]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({2, 3}, {7, 8, 9, 10, 11, 12});

    auto result = einsum<int>("ij,ij->ij", A, B);
    REQUIRE(result.tensor[{0, 0}] == 7);
    REQUIRE(result.tensor[{0, 1}] == 16);
    REQUIRE(result.tensor[{1, 2}] == 72);
}

TEST_CASE("einsum: trace", "[einsum]")
{
    Tensor<int> sq({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto result = einsum<int>("ii->", sq);
    // 1 + 5 + 9 = 15
    REQUIRE(result.tensor[0] == 15);
}

TEST_CASE("einsum: diagonal", "[einsum]")
{
    Tensor<int> sq({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    auto result = einsum<int>("ii->i", sq);
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor[0] == 1);
    REQUIRE(result.tensor[1] == 5);
    REQUIRE(result.tensor[2] == 9);
}

TEST_CASE("einsum: transpose", "[einsum]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});

    auto result = einsum<int>("ij->ji", A);
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor.shape()[1] == 2);
    REQUIRE(result.tensor[{0, 0}] == 1);
    REQUIRE(result.tensor[{0, 1}] == 4);
    REQUIRE(result.tensor[{1, 0}] == 2);
    REQUIRE(result.tensor[{2, 1}] == 6);
}

TEST_CASE("einsum: batch matrix multiply", "[einsum]")
{
    Tensor<int> A({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor<int> B({2, 2, 2}, {1, 0, 0, 1, 2, 1, 1, 2});

    auto result = einsum<int>("bij,bjk->bik", A, B);
    REQUIRE(result.tensor.shape()[0] == 2);
    REQUIRE(result.tensor.shape()[1] == 2);
    REQUIRE(result.tensor.shape()[2] == 2);
    // Batch 0: [[1,2],[3,4]] * [[1,0],[0,1]] = [[1,2],[3,4]]
    REQUIRE(result.tensor[{0, 0, 0}] == 1);
    REQUIRE(result.tensor[{0, 0, 1}] == 2);
    REQUIRE(result.tensor[{0, 1, 0}] == 3);
    REQUIRE(result.tensor[{0, 1, 1}] == 4);
    // Batch 1: [[5,6],[7,8]] * [[2,1],[1,2]] = [[16,17],[22,23]]
    REQUIRE(result.tensor[{1, 0, 0}] == 16);
    REQUIRE(result.tensor[{1, 0, 1}] == 17);
    REQUIRE(result.tensor[{1, 1, 0}] == 22);
    REQUIRE(result.tensor[{1, 1, 1}] == 23);
}

TEST_CASE("einsum: 3D contraction", "[einsum]")
{
    Tensor<int> A({2, 3, 4});
    Tensor<int> B({2, 4, 5});

    int val = 1;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                A[{i, j, k}] = val++;

    val = 10;
    for (size_t i = 0; i < 2; ++i)
        for (size_t k = 0; k < 4; ++k)
            for (size_t l = 0; l < 5; ++l)
                B[{i, k, l}] = val++;

    auto result = einsum<int>("ijk,ikl->ijl", A, B);
    REQUIRE(result.tensor.shape()[0] == 2);
    REQUIRE(result.tensor.shape()[1] == 3);
    REQUIRE(result.tensor.shape()[2] == 5);
}

TEST_CASE("einsum: sum all elements", "[einsum]")
{
    Tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});

    auto result = einsum<int>("...->", t);
    // 1+2+3+4+5+6 = 21
    REQUIRE(result.tensor[0] == 21);
}

// ============================================================
// Einsum - Implicit Mode (no arrow)
// ============================================================

TEST_CASE("einsum: implicit mode matrix multiply", "[einsum][implicit]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({3, 2}, {7, 8, 9, 10, 11, 12});

    // ij,jk -> ik (j appears twice, summed out)
    auto result = einsum<int>("ij,jk", A, B);
    REQUIRE(result.tensor.shape()[0] == 2);
    REQUIRE(result.tensor.shape()[1] == 2);
    REQUIRE(result.tensor[{0, 0}] == 58);
    REQUIRE(result.tensor[{0, 1}] == 64);
    REQUIRE(result.tensor[{1, 0}] == 139);
    REQUIRE(result.tensor[{1, 1}] == 154);
}

TEST_CASE("einsum: implicit mode dot product", "[einsum][implicit]")
{
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({3}, {4, 5, 6});

    // i,i -> scalar (i appears twice, summed out)
    auto result = einsum<int>("i,i", a, b);
    REQUIRE(result.tensor.size() == 1);
    REQUIRE(result.tensor[0] == 32);
}

TEST_CASE("einsum: implicit mode outer product", "[einsum][implicit]")
{
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({4}, {4, 5, 6, 7});

    // i,j -> ij (i,j each appear once, kept)
    auto result = einsum<int>("i,j", a, b);
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor.shape()[1] == 4);
    REQUIRE(result.tensor[{0, 0}] == 4);
    REQUIRE(result.tensor[{2, 3}] == 21);
}

TEST_CASE("einsum: implicit mode trace", "[einsum][implicit]")
{
    Tensor<int> sq({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});

    // ii -> scalar (i appears twice)
    auto result = einsum<int>("ii", sq);
    REQUIRE(result.tensor[0] == 15);
}

// ============================================================
// Einsum - Repeated Indices
// ============================================================

TEST_CASE("einsum: repeated index trace 4x4", "[einsum][repeated]")
{
    Tensor<double> A({4, 4});
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A[{i, j}] = static_cast<double>(i * 4 + j + 1);

    auto result = einsum<double>("ii->", A);
    // Diagonal: 1 + 6 + 11 + 16 = 34
    REQUIRE(result.tensor[0] == Approx(34.0));
}

TEST_CASE("einsum: repeated index diagonal extraction", "[einsum][repeated]")
{
    Tensor<int> A({3, 3}, {10, 20, 30, 40, 50, 60, 70, 80, 90});

    auto result = einsum<int>("ii->i", A);
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor[0] == 10);
    REQUIRE(result.tensor[1] == 50);
    REQUIRE(result.tensor[2] == 90);
}

// ============================================================
// Einsum - Ellipsis (Broadcast)
// ============================================================

TEST_CASE("einsum: ellipsis sum all", "[einsum][ellipsis]")
{
    Tensor<int> t({2, 3, 4});
    int val = 1;
    for (auto &v : t.data)
        v = val++;

    auto result = einsum<int>("...->", t);
    // sum 1..24 = 300
    REQUIRE(result.tensor[0] == 300);
}

TEST_CASE("einsum: ellipsis batch matmul", "[einsum][ellipsis]")
{
    Tensor<int> A({2, 3, 4});
    Tensor<int> B({2, 4, 5});

    int val = 1;
    for (size_t b = 0; b < 2; ++b)
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                A[{b, i, j}] = val++;

    val = 1;
    for (size_t b = 0; b < 2; ++b)
        for (size_t j = 0; j < 4; ++j)
            for (size_t k = 0; k < 5; ++k)
                B[{b, j, k}] = val++;

    // ...ij,...jk->...ik should behave like batch matmul
    auto result = einsum<int>("...ij,...jk->...ik", A, B);
    REQUIRE(result.tensor.shape()[0] == 2);
    REQUIRE(result.tensor.shape()[1] == 3);
    REQUIRE(result.tensor.shape()[2] == 5);
}

TEST_CASE("einsum: ellipsis hadamard", "[einsum][ellipsis]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({2, 3}, {7, 8, 9, 10, 11, 12});

    auto result = einsum<int>("...,...->...", A, B);
    REQUIRE(result.tensor[{0, 0}] == 7);
    REQUIRE(result.tensor[{0, 1}] == 16);
    REQUIRE(result.tensor[{1, 2}] == 72);
}

// ============================================================
// Einsum - Multi-input (3+ tensors)
// ============================================================

TEST_CASE("einsum: triple contraction", "[einsum][multi]")
{
    Tensor<int> A({2, 2}, {1, 2, 3, 4});
    Tensor<int> B({2, 2}, {5, 6, 7, 8});
    Tensor<int> C({2, 2}, {1, 0, 0, 1});

    // A_ij * B_jk * C_kl -> il
    auto result = einsum<int>("ij,jk,kl->il", A, B, C);
    REQUIRE(result.tensor.shape()[0] == 2);
    REQUIRE(result.tensor.shape()[1] == 2);
    // A*B = [[19,22],[43,50]]
    // (A*B)*C = [[19,22],[43,50]] (since C is identity)
    REQUIRE(result.tensor[{0, 0}] == 19);
    REQUIRE(result.tensor[{0, 1}] == 22);
    REQUIRE(result.tensor[{1, 0}] == 43);
    REQUIRE(result.tensor[{1, 1}] == 50);
}

// ============================================================
// Operations
// ============================================================

TEST_CASE("matmul", "[operations]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({3, 2}, {7, 8, 9, 10, 11, 12});

    auto C = matmul(A, B);
    REQUIRE(C.tensor[{0, 0}] == 58);
    REQUIRE(C.tensor[{1, 1}] == 154);
}

TEST_CASE("dot", "[operations]")
{
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({3}, {4, 5, 6});
    auto result = dot(a, b);
    REQUIRE(result.tensor[0] == 32);
}

TEST_CASE("outer", "[operations]")
{
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({2}, {4, 5});
    auto result = outer(a, b);
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor.shape()[1] == 2);
    REQUIRE(result.tensor[{0, 0}] == 4);
    REQUIRE(result.tensor[{2, 1}] == 15);
}

TEST_CASE("hadamard", "[operations]")
{
    Tensor<int> A({2, 2}, {1, 2, 3, 4});
    Tensor<int> B({2, 2}, {5, 6, 7, 8});
    auto result = hadamard(A, B);
    REQUIRE(result.tensor[{0, 0}] == 5);
    REQUIRE(result.tensor[{0, 1}] == 12);
    REQUIRE(result.tensor[{1, 0}] == 21);
    REQUIRE(result.tensor[{1, 1}] == 32);
}

TEST_CASE("trace", "[operations]")
{
    Tensor<int> sq({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    REQUIRE(trace(sq) == 15);
}

TEST_CASE("transpose", "[operations]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    auto T = transpose(A);
    REQUIRE(T.tensor.shape()[0] == 3);
    REQUIRE(T.tensor.shape()[1] == 2);
    REQUIRE(T.tensor[{0, 0}] == 1);
    REQUIRE(T.tensor[{0, 1}] == 4);
    REQUIRE(T.tensor[{2, 0}] == 3);
    REQUIRE(T.tensor[{2, 1}] == 6);
}

TEST_CASE("sum all", "[operations]")
{
    Tensor<int> t({2, 3}, {1, 2, 3, 4, 5, 6});
    REQUIRE(sum(t) == 21);
}

TEST_CASE("sum axis", "[operations]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    auto result = sum(A, 0); // sum along rows -> shape {3}
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor[0] == 5);  // 1+4
    REQUIRE(result.tensor[1] == 7);  // 2+5
    REQUIRE(result.tensor[2] == 9);  // 3+6
}

TEST_CASE("contract", "[operations]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    auto result = contract(A, {1}); // sum columns -> shape {2}
    REQUIRE(result.tensor.shape()[0] == 2);
    REQUIRE(result.tensor[0] == 6);   // 1+2+3
    REQUIRE(result.tensor[1] == 15);  // 4+5+6
}

TEST_CASE("gram", "[operations]")
{
    Tensor<int> X({2, 3}, {1, 2, 3, 4, 5, 6});
    auto G = gram(X);
    // G = X * X^T
    // G[0,0] = 1+4+9 = 14
    // G[0,1] = 4+10+18 = 32
    // G[1,0] = 32
    // G[1,1] = 16+25+36 = 77
    REQUIRE(G.tensor.shape()[0] == 2);
    REQUIRE(G.tensor.shape()[1] == 2);
    REQUIRE(G.tensor[{0, 0}] == 14);
    REQUIRE(G.tensor[{0, 1}] == 32);
    REQUIRE(G.tensor[{1, 1}] == 77);
}

// ============================================================
// Math Functions
// ============================================================

TEST_CASE("math exp", "[math]")
{
    Tensor<double> v({3}, {0.0, 1.0, 2.0});
    auto result = math::exp(v);
    REQUIRE(result.tensor[0] == Approx(1.0));
    REQUIRE(result.tensor[1] == Approx(std::exp(1.0)));
    REQUIRE(result.tensor[2] == Approx(std::exp(2.0)));
}

TEST_CASE("math log", "[math]")
{
    Tensor<double> v({3}, {1.0, 2.718281828, 7.389056099});
    auto result = math::log(v);
    REQUIRE(result.tensor[0] == Approx(0.0).margin(1e-6));
    REQUIRE(result.tensor[1] == Approx(1.0).margin(1e-4));
    REQUIRE(result.tensor[2] == Approx(2.0).margin(1e-4));
}

TEST_CASE("math sqrt", "[math]")
{
    Tensor<double> v({3}, {4.0, 9.0, 16.0});
    auto result = math::sqrt(v);
    REQUIRE(result.tensor[0] == Approx(2.0));
    REQUIRE(result.tensor[1] == Approx(3.0));
    REQUIRE(result.tensor[2] == Approx(4.0));
}

TEST_CASE("math mean", "[math]")
{
    Tensor<double> v({4}, {1.0, 2.0, 3.0, 4.0});
    REQUIRE(math::mean(v) == Approx(2.5));
}

TEST_CASE("math var", "[math]")
{
    Tensor<double> v({4}, {1.0, 2.0, 3.0, 4.0});
    // mean=2.5, var = ((1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2)/4 = 1.25
    REQUIRE(math::var(v) == Approx(1.25));
}

TEST_CASE("math norm", "[math]")
{
    Tensor<double> v({3}, {3.0, 4.0, 0.0});
    REQUIRE(math::norm(v) == Approx(5.0));
}

TEST_CASE("math frobenius_norm", "[math]")
{
    Tensor<double> M({2, 2}, {1.0, 2.0, 3.0, 4.0});
    // sqrt(1+4+9+16) = sqrt(30)
    REQUIRE(math::frobenius_norm(M) == Approx(std::sqrt(30.0)));
}

// ============================================================
// Edge Cases
// ============================================================

TEST_CASE("einsum: 1D tensor operations", "[einsum][edge]")
{
    Tensor<int> a({5}, {1, 2, 3, 4, 5});
    Tensor<int> b({5}, {2, 3, 4, 5, 6});

    // Hadamard product on 1D
    auto result = einsum<int>("i,i->i", a, b);
    REQUIRE(result.tensor.shape()[0] == 5);
    REQUIRE(result.tensor[0] == 2);
    REQUIRE(result.tensor[1] == 6);
    REQUIRE(result.tensor[4] == 30);
}

TEST_CASE("einsum: identity matrix multiply", "[einsum][edge]")
{
    auto I = eye<int>(3);
    Tensor<int> v({3, 1}, {1, 2, 3});

    auto result = einsum<int>("ij,jk->ik", I, v);
    REQUIRE(result.tensor.shape()[0] == 3);
    REQUIRE(result.tensor.shape()[1] == 1);
    REQUIRE(result.tensor[{0, 0}] == 1);
    REQUIRE(result.tensor[{1, 0}] == 2);
    REQUIRE(result.tensor[{2, 0}] == 3);
}

TEST_CASE("einsum: scalar result from large tensor", "[einsum][edge]")
{
    Tensor<int> t({10, 10});
    for (size_t i = 0; i < 100; ++i)
        t.data[i] = 1;

    auto result = einsum<int>("...->", t);
    REQUIRE(result.tensor[0] == 100);
}

TEST_CASE("einsum: wrong number of tensors throws", "[einsum][error]")
{
    Tensor<int> A({2, 2}, {1, 2, 3, 4});
    REQUIRE_THROWS_AS(einsum<int>("ij,jk->ik", A), std::invalid_argument);
}

TEST_CASE("einsum: inconsistent dimensions throws", "[einsum][error]")
{
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    REQUIRE_THROWS_AS(einsum<int>("ij,jk->ik", A, B), std::invalid_argument);
}

TEST_CASE("Tensor operator<< for empty tensor", "[tensor][output]")
{
    Tensor<int> t;
    std::ostringstream oss;
    oss << t;
    REQUIRE(oss.str() == "Tensor[]");
}

TEST_CASE("Tensor operator<< for scalar", "[tensor][output]")
{
    Tensor<int> t({}, {42});
    std::ostringstream oss;
    oss << t;
    REQUIRE(oss.str() == "42");
}

TEST_CASE("Tensor equality", "[tensor]")
{
    Tensor<int> a({2, 2}, {1, 2, 3, 4});
    Tensor<int> b({2, 2}, {1, 2, 3, 4});
    Tensor<int> c({2, 2}, {1, 2, 3, 5});
    REQUIRE(a == b);
    REQUIRE_FALSE(a == c);
}
