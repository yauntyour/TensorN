#include "core/core.hpp"
#include <iostream>

int main()
{
    using namespace TensorN;

    // 创建张量
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({3, 2}, {1, 2, 3, 4, 5, 6});

    // 使用 << 运算符输出
    std::cout << "A = " << A << std::endl;
    // 输出: A = [[1, 2, 3], [4, 5, 6]]

    std::cout << "B = " << B << std::endl;

    // 矩阵乘法
    auto C = matmul(A, B);
    std::cout << "A * B = " << C << std::endl;

    // 使用工厂函数
    auto Z = zeros<float>({2, 2});
    auto I = eye<double>(3);

    std::cout << "\nZeros: " << Z << std::endl;
    std::cout << "Identity: " << I << std::endl;

    // 向量点积
    Tensor<int> v1({3}, {1, 2, 3});
    Tensor<int> v2({3}, {4, 5, 6});
    auto dot_product = dot(v1, v2);
    std::cout << "\nv1 · v2 = " << dot_product << std::endl;

    // 标量运算
    auto A_plus_10 = A + 10;
    std::cout << "\nA + 10 = " << A_plus_10 << std::endl;


    return 0;
}
