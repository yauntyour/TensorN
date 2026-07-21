#include <iostream>
#include "Tensor.hpp"

using namespace TensorN;

int main()
{
    std::cout << "=== 爱因斯坦约定求和 (einsum) 示例 ===" << std::endl;

    // =================== 1. 矩阵乘法 ===================
    std::cout << "\n1. 矩阵乘法 (ij,jk->ik):" << std::endl;

    // 创建两个矩阵
    Tensor<int> A1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B1({3, 2}, {7, 8, 9, 10, 11, 12});

    std::cout << "A (2x3) = " << A1 << std::endl;
    std::cout << "B (3x2) = " << B1 << std::endl;

    // 使用einsum进行矩阵乘法
    auto C1 = einsum<int>("ij,jk->ik", A1, B1);
    std::cout << "C = einsum('ij,jk->ik', A, B) = " << C1.tensor << std::endl;

    // 验证：使用matmul函数应该得到相同结果
    auto C1_check = matmul(A1, B1);
    std::cout << "matmul(A, B) = " << C1_check.tensor << std::endl;

    // =================== 2. 向量点积 ===================
    std::cout << "\n\n2. 向量点积 (i,i->):" << std::endl;

    Tensor<float> v1({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor<float> v2({4}, {0.5f, 1.5f, 2.5f, 3.5f});

    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;

    auto dot_result = einsum<float>("i,i->", v1, v2);
    std::cout << "dot = einsum('i,i->', v1, v2) = " << dot_result.tensor << std::endl;

    // =================== 3. 外积 ===================
    std::cout << "\n\n3. 外积 (i,j->ij):" << std::endl;

    Tensor<int> a({3}, {1, 2, 3});
    Tensor<int> b({4}, {4, 5, 6, 7});

    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;

    auto outer_result = einsum<int>("i,j->ij", a, b);
    std::cout << "outer = einsum('i,j->ij', a, b) = " << outer_result.tensor << std::endl;

    // =================== 4. 逐元素乘法 ===================
    std::cout << "\n\n4. 逐元素乘法 (ij,ij->ij):" << std::endl;

    Tensor<double> M1({2, 3}, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
    Tensor<double> M2({2, 3}, {0.5, 1.0, 1.5, 2.0, 2.5, 3.0});

    std::cout << "M1 = " << M1 << std::endl;
    std::cout << "M2 = " << M2 << std::endl;

    auto elemwise_result = einsum<double>("ij,ij->ij", M1, M2);
    std::cout << "elemwise = einsum('ij,ij->ij', M1, M2) = " << elemwise_result.tensor << std::endl;

    // =================== 5. 张量收缩：三维示例 ===================
    std::cout << "\n\n5. 张量收缩：三维张量 (ijk,ikl->ijl):" << std::endl;

    // 创建两个三维张量
    Tensor<int> T1({2, 3, 4});
    Tensor<int> T2({2, 4, 5});

    // 填充测试数据
    int val1 = 1;
    for (size_t i = 0; i < T1.shape()[0]; ++i)
    {
        for (size_t j = 0; j < T1.shape()[1]; ++j)
        {
            for (size_t k = 0; k < T1.shape()[2]; ++k)
            {
                T1[{i, j, k}] = val1++;
            }
        }
    }

    int val2 = 10;
    for (size_t i = 0; i < T2.shape()[0]; ++i)
    {
        for (size_t k = 0; k < T2.shape()[1]; ++k)
        {
            for (size_t l = 0; l < T2.shape()[2]; ++l)
            {
                T2[{i, k, l}] = val2++;
            }
        }
    }

    std::cout << "T1 形状: (";
    for (auto s : T1.shape())
        std::cout << s << " ";
    std::cout << ")" << std::endl;

    std::cout << "T2 形状: (";
    for (auto s : T2.shape())
        std::cout << s << " ";
    std::cout << ")" << std::endl;

    // 执行einsum: 对第3个维度进行收缩
    auto T3 = einsum<int>("ijk,ikl->ijl", T1, T2);
    std::cout << "T3 = einsum('ijk,ikl->ijl', T1, T2)" << std::endl;
    std::cout << "T3 形状: (";
    for (auto s : T3.tensor.shape())
        std::cout << s << " ";
    std::cout << ")" << std::endl;

    // 打印部分结果
    std::cout << "T3[0][0] = [";
    for (size_t l = 0; l < 5 && l < T3.tensor.shape()[2]; ++l)
    {
        std::cout << T3.tensor[{0, 0, l}];
        if (l < 4)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // =================== 6. 迹（对角线求和） ===================
    std::cout << "\n\n6. 矩阵的迹 (ii->):" << std::endl;

    Tensor<int> square({3, 3}, {1, 2, 3,
                                4, 5, 6,
                                7, 8, 9});

    std::cout << "方阵 = " << square << std::endl;

    auto trace_result = einsum<int>("ii->", square);
    std::cout << "迹 = einsum('ii->', 方阵) = " << trace_result.tensor << std::endl;
    std::cout << "验证: 对角线元素和 = " << (1 + 5 + 9) << std::endl;

    // =================== 7. 批量矩阵乘法 ===================
    std::cout << "\n\n7. 批量矩阵乘法 (bij,bjk->bik):" << std::endl;

    Tensor<int> batch1({2, 3, 4}); // 2个3x4矩阵
    Tensor<int> batch2({2, 4, 5}); // 2个4x5矩阵

    // 填充数据
    for (size_t b = 0; b < 2; ++b)
    {
        int counter = b * 100 + 1;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 4; ++j)
            {
                batch1[{b, i, j}] = counter++;
            }
        }
    }

    for (size_t b = 0; b < 2; ++b)
    {
        int counter = b * 200 + 1;
        for (size_t j = 0; j < 4; ++j)
        {
            for (size_t k = 0; k < 5; ++k)
            {
                batch2[{b, j, k}] = counter++;
            }
        }
    }

    auto batch_result = einsum<int>("bij,bjk->bik", batch1, batch2);
    std::cout << "批量结果形状: (";
    for (auto s : batch_result.tensor.shape())
        std::cout << s << " ";
    std::cout << ")" << std::endl;

    // =================== 8. 张量缩并：复杂示例 ===================
    std::cout << "\n\n8. 复杂缩并 (ijk,ijl->kl):" << std::endl;

    Tensor<int> X({2, 3, 4});
    Tensor<int> Y({2, 3, 5});

    // 填充简单数据
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            for (size_t k = 0; k < 4; ++k)
            {
                X[{i, j, k}] = (i + 1) * 100 + (j + 1) * 10 + (k + 1);
            }
            for (size_t l = 0; l < 5; ++l)
            {
                Y[{i, j, l}] = (i + 1) * 200 + (j + 1) * 20 + (l + 1);
            }
        }
    }

    auto Z = einsum<int>("ijk,ijl->kl", X, Y);
    std::cout << "Z 形状: (" << Z.tensor.shape()[0] << ", " << Z.tensor.shape()[1] << ")" << std::endl;

    // =================== 9. 求和所有元素 ===================
    std::cout << "\n\n9. 求和所有元素 (...->):" << std::endl;

    Tensor<int> tensor3D({2, 2, 3}, {1, 2, 3,
                                     4, 5, 6,
                                     7, 8, 9,
                                     10, 11, 12});

    std::cout << "3D张量 = " << tensor3D << std::endl;

    auto sum_all = einsum<int>("...->", tensor3D);
    std::cout << "所有元素和 = einsum('...->', 张量) = " << sum_all.tensor << std::endl;
    std::cout << "验证: 1+2+...+12 = " << (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12) << std::endl;

    // =================== 10. 广播乘法 ===================
    std::cout << "\n\n10. 广播乘法 (ij,j->ij):" << std::endl;

    Tensor<int> matrix({3, 4}, {1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12});

    Tensor<int> vector({4}, {2, 3, 4, 5});

    std::cout << "矩阵 = " << matrix << std::endl;
    std::cout << "向量 = " << vector << std::endl;

    auto broadcast_result = einsum("ij,j->ij", matrix, vector);
    std::cout << "广播结果 = einsum('ij,j->ij', 矩阵, 向量) = " << broadcast_result.tensor << std::endl;

    return 0;
}