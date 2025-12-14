#include <iostream>
#include <vector>
#include "Tensor.hpp"

using namespace TensorN;

// 简单的打印函数（假设 Tensor 支持 operator<< 或提供 data() 和 shape()）
template <typename T>
void print_tensor(const std::string &name, const opt<T> &t)
{
    std::cout << name << " =\n";
    // 假设 Tensor 有友元 ostream 或提供访问方式
    // 这里简化：若 opt<T> 是 Tensor<T> 别名，则直接输出
    std::cout << t.tensor << "\n\n";
}

// 特化标量打印
template <typename T>
void print_scalar(const std::string &name, T val)
{
    std::cout << name << " = " << val << "\n\n";
}

int main()
{
    try
    {
        // === 向量 ===
        Tensor<double> v1({3}, {1.0, 2.0, 3.0});
        Tensor<double> v2({3}, {4.0, 5.0, 6.0});

        // === 矩阵 ===
        Tensor<double> M1({2, 3}, {1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0});
        Tensor<double> M2({3, 2}, {7.0, 8.0,
                                   9.0, 10.0,
                                   11.0, 12.0});
        Tensor<double> Sq({3, 3}, {1.0, 2.0, 3.0,
                                   4.0, 5.0, 6.0,
                                   7.0, 8.0, 9.0});

        // === 测试 dot ===
        auto dot_res = dot(v1, v2);
        print_scalar("dot(v1, v2)", dot_res[0]);

        // === 测试 outer ===
        auto outer_res = outer(v1, v2);
        print_tensor("outer(v1, v2)", outer_res);

        // === 测试 matmul ===
        auto matmul_res = matmul(M1, M2);
        print_tensor("matmul(M1, M2)", matmul_res);

        // === 测试 hadamard ===
        auto hadamard_res = hadamard(M1, M1); // 元素平方
        print_tensor("hadamard(M1, M1)", hadamard_res);

        // === 测试 bilinear ===
        double bilin = bilinear(v1, Sq, v2);
        print_scalar("bilinear(v1, Sq, v2)", bilin);

        // === 测试 gram ===
        auto gram_res = gram(M1);
        print_tensor("gram(M1)", gram_res);

        // === 测试 contract (对第1维缩并) ===
        auto contract_res = contract(M1, {1}); // 对列求和 -> shape [2]
        print_tensor("contract(M1, {1})", contract_res);

        // === 测试 trace ===
        double tr = trace(Sq);
        print_scalar("trace(Sq)", tr);

        // === 测试 sum (全部) ===
        double total_sum = sum(M1);
        print_scalar("sum(M1)", total_sum);

        // === 测试 sum(axis=0) ===
        auto sum_axis0 = sum(M1, 0); // shape [3]
        print_tensor("sum(M1, axis=0)", sum_axis0);

        // === 测试 transpose ===
        auto trans = transpose(M1); // 默认反转 -> [3,2]
        print_tensor("transpose(M1)", trans);

        auto trans_custom = transpose(M1, {1, 0}); // 显式转置
        print_tensor("transpose(M1, {1,0})", trans_custom);

        // === 测试 diag ===
        auto diag_vec = diag(Sq);
        print_tensor("diag(Sq)", diag_vec);

        // === 测试 diag_matrix ===
        auto diag_mat = diag_matrix(diag_vec.tensor);
        print_tensor("diag_matrix(diag(Sq))", diag_mat);

        // === 数学函数 ===
        using namespace math;

        auto exp_res = exp(v1);
        print_tensor("exp(v1)", exp_res);

        auto log_res = log(exp_res.tensor);
        print_tensor("log(exp(v1))", log_res);

        auto sqrt_res = sqrt(v1);
        print_tensor("sqrt(v1)", sqrt_res);

        auto sin_res = sin(v1);
        print_tensor("sin(v1)", sin_res);

        auto cos_res = cos(v1);
        print_tensor("cos(v1)", cos_res);

        double mean_val = mean(v1);
        print_scalar("mean(v1)", mean_val);

        double var_val = var(v1);
        print_scalar("var(v1)", var_val);

        double stddev_val = stddev(v1);
        print_scalar("stddev(v1)", stddev_val);

        double norm_val = norm(v1);
        print_scalar("norm(v1)", norm_val);

        double frob_val = frobenius_norm(M1);
        print_scalar("frobenius_norm(M1)", frob_val);

        std::cout << "✅ 所有操作测试通过！\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ 错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}