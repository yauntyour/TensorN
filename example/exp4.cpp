#include "TensorN.hpp"
#include <iostream>

using namespace TensorN;

int main()
{
    std::cout << "=== exp4: Einstein Summation (einsum) ===\n" << std::endl;

    // 1. Basic matrix multiplication: ij,jk->ik
    std::cout << "1. Matrix multiplication (ij,jk->ik):" << std::endl;
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({3, 2}, {7, 8, 9, 10, 11, 12});
    auto C1 = einsum<int>("ij,jk->ik", A, B);
    std::cout << "  " << C1 << std::endl;
    std::cout << "  verify vs matmul: " << matmul(A, B).tensor[0] << "==" << C1.tensor[0] << std::endl;

    // 2. Dot product: i,i->
    std::cout << "\n2. Dot product (i,i->):" << std::endl;
    Tensor<float> v1({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> v2({4}, {0.5f, 1.5f, 2.5f, 3.5f});
    auto dot_r = einsum<float>("i,i->", v1, v2);
    std::cout << "  " << dot_r << std::endl;

    // 3. Outer product: i,j->ij
    std::cout << "\n3. Outer product (i,j->ij):" << std::endl;
    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({4}, {4, 5, 6, 7});
    auto outer_r = einsum<int>("i,j->ij", a, b);
    std::cout << "  " << outer_r << std::endl;

    // 4. Element-wise multiply: ij,ij->ij
    std::cout << "\n4. Element-wise multiply (ij,ij->ij):" << std::endl;
    Tensor<double> M1({2, 3}, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
    Tensor<double> M2({2, 3}, {0.5, 1.0, 1.5, 2.0, 2.5, 3.0});
    auto ew = einsum<double>("ij,ij->ij", M1, M2);
    std::cout << "  " << ew << std::endl;

    // 5. Contraction: ijk,ikl->ijl
    std::cout << "\n5. 3D contraction (ijk,ikl->ijl):" << std::endl;
    Tensor<int> T1({2, 3, 4});
    Tensor<int> T2({2, 4, 5});
    int v = 1;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                T1[{i, j, k}] = v++;
    v = 10;
    for (size_t i = 0; i < 2; ++i)
        for (size_t k = 0; k < 4; ++k)
            for (size_t l = 0; l < 5; ++l)
                T2[{i, k, l}] = v++;
    auto T3 = einsum<int>("ijk,ikl->ijl", T1, T2);
    std::cout << "  output shape: {";
    for (auto s : T3.tensor.shape()) std::cout << s << " ";
    std::cout << "}" << std::endl;

    // 6. Trace: ii->
    std::cout << "\n6. Trace (ii->):" << std::endl;
    Tensor<int> square({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto tr = einsum<int>("ii->", square);
    std::cout << "  " << tr << "  (expected " << (1+5+9) << ")" << std::endl;

    // 7. Batched matmul: bij,bjk->bik
    std::cout << "\n7. Batched matmul (bij,bjk->bik):" << std::endl;
    Tensor<int> bat1({2, 2, 3});
    Tensor<int> bat2({2, 3, 2});
    v = 1;
    for (size_t b = 0; b < 2; ++b)
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 3; ++j)
                bat1[{b, i, j}] = v++;
    v = 100;
    for (size_t b = 0; b < 2; ++b)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 2; ++k)
                bat2[{b, j, k}] = v++;
    auto batch = einsum<int>("bij,bjk->bik", bat1, bat2);
    std::cout << "  output shape: {";
    for (auto s : batch.tensor.shape()) std::cout << s << " ";
    std::cout << "}" << std::endl;

    // 8. Ellipsis: sum all ...->
    std::cout << "\n8. Sum all with ellipsis (...->):" << std::endl;
    Tensor<int> t3d({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto sa = einsum<int>("...->", t3d);
    std::cout << "  " << sa << "  (expected " << 78 << ")" << std::endl;

    // 9. Broadcast with ellipsis: ij,j->ij
    std::cout << "\n9. Broadcast multiply (ij,j->ij):" << std::endl;
    Tensor<int> mat({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor<int> vec({4}, {2, 3, 4, 5});
    auto br = einsum<int>("ij,j->ij", mat, vec);
    std::cout << "  " << br << std::endl;

    // 10. Complex contraction: ijk,ijl->kl
    std::cout << "\n10. Complex contraction (ijk,ijl->kl):" << std::endl;
    Tensor<int> X({2, 3, 4});
    Tensor<int> Y({2, 3, 5});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k)
                X[{i, j, k}] = i*100 + j*10 + k;
            for (size_t l = 0; l < 5; ++l)
                Y[{i, j, l}] = i*200 + j*20 + l;
        }
    auto Z = einsum<int>("ijk,ijl->kl", X, Y);
    std::cout << "  output shape: {";
    for (auto s : Z.tensor.shape()) std::cout << s << " ";
    std::cout << "}" << std::endl;

    return 0;
}
