#include "TensorN.hpp"
#include <iostream>

using namespace TensorN;

int main()
{
    std::cout << "=== exp1: Tensor Creation & Manipulation ===\n" << std::endl;

    // 1. Constructors
    std::cout << "1. Construction:" << std::endl;
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({2, 3});   // uninitialized
    Tensor<int> C(A);        // copy
    Tensor<int> D(std::move(A)); // move (A becomes empty)
    std::cout << "  B (shape 2x3, uninit) = " << B << std::endl;
    std::cout << "  C (copy) = " << C << std::endl;
    std::cout << "  D (moved) = " << D << std::endl;

    // 2. Factory functions
    std::cout << "\n2. Factory functions:" << std::endl;
    auto Z = zeros<float>({2, 3});
    auto O = ones<int>({3, 2});
    auto I = eye<double>(4);
    auto R = arange<int>(0, 10, 2);

    std::cout << "  zeros<float>(2,3) = " << Z << std::endl;
    std::cout << "  ones<int>(3,2) = " << O << std::endl;
    std::cout << "  eye<double>(4) = " << I << std::endl;
    std::cout << "  arange<int>(0,10,2) = " << R << std::endl;

    // 3. Indexing
    std::cout << "\n3. Indexing:" << std::endl;
    Tensor<int> M({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::cout << "  M = " << M << std::endl;
    std::cout << "  M[{0,0}] = " << M[{0, 0}] << std::endl;
    std::cout << "  M[{1,2}] = " << M[{1, 2}] << std::endl;
    std::cout << "  M[4] (flat) = " << M[4] << std::endl;

    // 4. Shape and size
    std::cout << "\n4. Shape & size:" << std::endl;
    std::cout << "  M.shape() = {";
    for (auto s : M.shape()) std::cout << s << " ";
    std::cout << "}" << std::endl;
    std::cout << "  M.size() = " << M.size() << std::endl;

    // 5. Clone, view, shallow_copy
    std::cout << "\n5. Clone & view:" << std::endl;
    auto clone = M.clone();
    auto view = M.view();
    M[{0, 0}] = 99;
    std::cout << "  After M[{0,0}]=99:" << std::endl;
    std::cout << "  M     = " << M << std::endl;
    std::cout << "  clone = " << clone << "  (unchanged)" << std::endl;
    std::cout << "  view  = " << view << "  (reflects change)" << std::endl;

    // 6. Reshape
    std::cout << "\n6. Reshape:" << std::endl;
    auto R2 = M.reshape({1, 9});
    std::cout << "  M.reshape({1,9}) = " << R2 << std::endl;

    // 7. fill / zero
    std::cout << "\n7. fill_ / zero_:" << std::endl;
    Tensor<int> F({2, 2}, {1, 2, 3, 4});
    std::cout << "  before: " << F << std::endl;
    F.fill_(42);
    std::cout << "  after fill_(42): " << F << std::endl;
    F.zero_();
    std::cout << "  after zero_(): " << F << std::endl;

    // 8. Scalar tensor
    std::cout << "\n8. Scalar tensor:" << std::endl;
    Tensor<float> s({}, {3.14f});
    std::cout << "  scalar = " << s << std::endl;
    std::cout << "  s[0] = " << s[0] << std::endl;

    // 9. Equality check
    std::cout << "\n9. Equality:" << std::endl;
    Tensor<int> X({2}, {1, 2});
    Tensor<int> Y({2}, {1, 2});
    Tensor<int> W({2}, {3, 4});
    std::cout << "  X == Y: " << (X == Y ? "true" : "false") << std::endl;
    std::cout << "  X == W: " << (X == W ? "true" : "false") << std::endl;

    return 0;
}
