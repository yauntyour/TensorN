#include "TensorN.hpp"
#include <iostream>

using namespace TensorN;

int main()
{
    std::cout << "=== exp3: Linear Algebra Operations ===\n" << std::endl;

    // 1. Matrix multiplication
    std::cout << "1. Matrix multiplication:" << std::endl;
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({3, 2}, {7, 8, 9, 10, 11, 12});
    auto C = matmul(A, B);
    std::cout << "  A(2x3) * B(3x2) = " << C << std::endl;

    // 2. Vector dot product
    std::cout << "\n2. Vector dot product:" << std::endl;
    Tensor<double> v1({3}, {1.0, 2.0, 3.0});
    Tensor<double> v2({3}, {4.0, 5.0, 6.0});
    auto d = dot(v1, v2);
    std::cout << "  dot(v1, v2) = " << d << "  (expected 32)" << std::endl;

    // 3. Outer product
    std::cout << "\n3. Outer product:" << std::endl;
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({2}, {10, 20});
    auto O = outer(a, b);
    std::cout << "  outer(a, b) = " << O << std::endl;

    // 4. Bilinear form: x^T A y
    std::cout << "\n4. Bilinear form (x^T A y):" << std::endl;
    Tensor<double> x({3}, {1.0, 2.0, 3.0});
    Tensor<double> M({3, 3}, {1.0, 0.0, 0.0,
                               0.0, 2.0, 0.0,
                               0.0, 0.0, 3.0});
    Tensor<double> y({3}, {4.0, 5.0, 6.0});
    double bl = bilinear(x, M, y);
    std::cout << "  bilinear(x, diag(1,2,3), y) = " << bl << std::endl;
    std::cout << "  expected: 1*1*4 + 2*2*5 + 3*3*6 = " << (1*1*4 + 2*2*5 + 3*3*6) << std::endl;

    // 5. Gram matrix: X X^T
    std::cout << "\n5. Gram matrix (X X^T):" << std::endl;
    Tensor<int> X({2, 3}, {1, 2, 3, 4, 5, 6});
    auto G = gram(X);
    std::cout << "  gram(X) = " << G << std::endl;

    // 6. Trace
    std::cout << "\n6. Trace:" << std::endl;
    Tensor<int> Sq({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::cout << "  trace(Sq) = " << trace(Sq) << "  (expected 15)" << std::endl;

    // 7. Transpose
    std::cout << "\n7. Transpose:" << std::endl;
    Tensor<int> T({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    std::cout << "  original (2x4) = " << T << std::endl;
    std::cout << "  transpose        = " << transpose(T) << std::endl;
    std::cout << "  transpose({1,0}) = " << transpose(T, {1, 0}) << std::endl;

    // 8. Diag and diag_matrix
    std::cout << "\n8. Diag / diag_matrix:" << std::endl;
    auto di = diag(Sq);
    std::cout << "  diag(Sq) = " << di << std::endl;
    auto dm = diag_matrix(di.tensor);
    std::cout << "  diag_matrix(diag(Sq)) = " << dm << std::endl;

    // 9. Tensor contraction
    std::cout << "\n9. Tensor contraction:" << std::endl;
    auto c2 = contract(Sq, {1});  // sum over columns → shape {3}
    std::cout << "  contract(Sq, {1}) = " << c2 << std::endl;
    auto c2_full = contract(Sq, {0, 1});  // trace (scalar)
    std::cout << "  contract(Sq, {0,1}) = " << c2_full << std::endl;

    return 0;
}
