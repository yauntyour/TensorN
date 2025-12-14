#pragma once
#ifndef __OPERATIONS__H__
#define __OPERATIONS__H__

#include "tensor.hpp"
#include "einsum.hpp"
#include <cmath>
#include <functional>

namespace TensorN
{
    // 向量点积 (Vector dot product)
    template <typename T>
    opt<T> dot(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 1 || B.shape().size() != 1)
        {
            throw std::invalid_argument("dot requires 1D tensors");
        }
        if (A.shape()[0] != B.shape()[0])
        {
            throw std::invalid_argument("Dimension mismatch for dot product");
        }
        return einsum<T>("i,i->", A, B);
    }

    // 外积 (Outer product)
    template <typename T>
    opt<T> outer(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 1 || B.shape().size() != 1)
        {
            throw std::invalid_argument("outer requires 1D tensors");
        }
        return einsum<T>("i,j->ij", A, B);
    }

    // 矩阵乘法 (Matrix multiplication)
    template <typename T>
    opt<T> matmul(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 2 || B.shape().size() != 2)
        {
            throw std::invalid_argument("matmul requires 2D tensors");
        }
        if (A.shape()[1] != B.shape()[0])
        {
            throw std::invalid_argument("Dimension mismatch for matrix multiplication");
        }
        return einsum<T>("ij,jk->ik", A, B);
    }

    // Hadamard 积 (逐元素乘)
    template <typename T>
    opt<T> hadamard(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (!A.is_isomorphic(B))
        {
            throw std::invalid_argument("Tensors must have same shape for Hadamard product");
        }
        return einsum<T>("...,...->...", A, B);
    }

    // 双线性型 (Bilinear form): x^T A y
    template <typename T>
    T bilinear(const Tensor<T> &x, const Tensor<T> &A, const Tensor<T> &y)
    {
        // x: vector (1D), A: matrix (2D), y: vector (1D)
        if (x.shape().size() != 1 || A.shape().size() != 2 || y.shape().size() != 1)
        {
            throw std::invalid_argument("bilinear: x must be 1D, A must be 2D, y must be 1D");
        }
        if (x.shape()[0] != A.shape()[0] || A.shape()[1] != y.shape()[0])
        {
            throw std::invalid_argument("Dimension mismatch for bilinear form");
        }

        // 使用 einsum: x_i * A_ij * y_j
        auto temp = einsum<T>("ij,j->i", A, y);
        auto result = einsum<T>("i,i->", x, temp.tensor);
        return result.tensor[0]; // 标量结果
    }

    // Gram 矩阵 (Gram matrix): X X^T
    template <typename T>
    opt<T> gram(const Tensor<T> &X)
    {
        if (X.shape().size() != 2)
        {
            throw std::invalid_argument("gram requires 2D tensor");
        }
        // X_ik * X_jk -> G_ij
        return einsum<T>("ik,jk->ij", X, X);
    }

    // 张量缩并 (Tensor contraction)
    // eg: contract(T, {0,1}, {1,2}) 表示对第1,2维求和
    template <typename T>
    opt<T> contract(const Tensor<T> &A, const std::vector<size_t> &axes)
    {
        if (axes.empty())
        {
            return A; // 不进行缩并
        }

        // 检查轴的有效性
        for (auto axis : axes)
        {
            if (axis >= A.shape().size())
            {
                throw std::invalid_argument("Axis out of range");
            }
        }

        // 构建 einsum 表达式
        std::string expr;
        std::string out_labels;

        // 输入标签
        char current_label = 'a';
        std::vector<char> labels(A.shape().size());
        for (size_t i = 0; i < A.shape().size(); ++i)
        {
            labels[i] = current_label++;
        }

        // 输出标签 (排除要缩并的轴)
        for (size_t i = 0; i < A.shape().size(); ++i)
        {
            if (std::find(axes.begin(), axes.end(), i) == axes.end())
            {
                out_labels += labels[i];
            }
        }

        // 构建表达式
        for (auto label : labels)
        {
            expr += label;
        }
        expr += "->";
        expr += out_labels.empty() ? "" : out_labels;

        return einsum<T>(expr, A);
    }

    // 迹 (Trace)
    template <typename T>
    T trace(const Tensor<T> &A)
    {
        if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
        {
            throw std::invalid_argument("trace requires a square matrix");
        }
        auto result = einsum<T>("ii->", A).tensor;
        return result[0];
    }

    // 求和所有元素
    template <typename T>
    T sum(const Tensor<T> &A)
    {
        auto result = einsum<T>("...->", A).tensor;
        return result[0];
    }

    // 按轴求和
    template <typename T>
    opt<T> sum(const Tensor<T> &A, size_t axis)
    {
        if (axis >= A.shape().size())
        {
            throw std::invalid_argument("Axis out of range");
        }

        // 构建表达式
        std::string expr;
        std::string out_labels;

        char current_label = 'a';
        for (size_t i = 0; i < A.shape().size(); ++i)
        {
            if (i == axis)
            {
                expr += current_label;
                // 不在输出中包含这个标签
            }
            else
            {
                expr += current_label;
                out_labels += current_label;
            }
            current_label++;
        }

        expr += "->";
        expr += out_labels.empty() ? "" : out_labels;

        return einsum<T>(expr, A);
    }

    // 转置 (Transpose)
    template <typename T>
    opt<T> transpose(const Tensor<T> &A, const std::vector<size_t> &axes = {})
    {
        auto shape = A.shape();
        std::vector<size_t> new_axes;

        if (axes.empty())
        {
            // 默认反转所有轴
            new_axes.resize(shape.size());
            for (size_t i = 0; i < shape.size(); ++i)
            {
                new_axes[i] = shape.size() - 1 - i;
            }
        }
        else
        {
            new_axes = axes;
        }

        // 验证轴的排列
        if (new_axes.size() != shape.size())
        {
            throw std::invalid_argument("Number of axes must match tensor dimension");
        }

        std::vector<bool> used(shape.size(), false);
        for (auto axis : new_axes)
        {
            if (axis >= shape.size())
            {
                throw std::invalid_argument("Axis out of range");
            }
            if (used[axis])
            {
                throw std::invalid_argument("Duplicate axis");
            }
            used[axis] = true;
        }

        // 构建 einsum 表达式
        std::string in_expr;
        std::string out_expr;
        char current_label = 'a';

        // 输入标签
        for (size_t i = 0; i < shape.size(); ++i)
        {
            in_expr += current_label;
            current_label++;
        }

        // 输出标签 (根据 new_axes 重新排列)
        out_expr = in_expr;
        for (size_t i = 0; i < new_axes.size(); ++i)
        {
            out_expr[i] = in_expr[new_axes[i]];
        }

        std::string expr = in_expr + "->" + out_expr;
        return einsum<T>(expr, A);
    }

    // 对角线元素 (Diagonal)
    template <typename T>
    opt<T> diag(const Tensor<T> &A)
    {
        if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
        {
            throw std::invalid_argument("diag requires a square matrix");
        }
        return einsum<T>("ii->i", A);
    }

    // 创建对角矩阵
    template <typename T>
    opt<T> diag_matrix(const Tensor<T> &v)
    {
        if (v.shape().size() != 1)
        {
            throw std::invalid_argument("diag_matrix requires a 1D tensor");
        }

        opt<T> oper({v.shape()[0], v.shape()[0]});
        for (size_t i = 0; i < v.shape()[0]; ++i)
        {
            oper.tensor[{i, i}] = v[i];
        }
        return oper;
    }

    // math functions
    namespace math
    {
        template <typename T, typename Func>
        opt<T> apply(const Tensor<T> &A, Func func)
        {
            Tensor<T> result(A.shape());
            std::transform(A.begin(), A.end(), result.begin(), func);
            return result;
        }

        // 指数函数
        template <typename T>
        opt<T> exp(const Tensor<T> &A)
        {
            return apply(A, [](T x)
                         { return std::exp(x); });
        }

        // 对数函数
        template <typename T>
        opt<T> log(const Tensor<T> &A)
        {
            return apply(A, [](T x)
                         { return std::log(x); });
        }

        // 平方根
        template <typename T>
        opt<T> sqrt(const Tensor<T> &A)
        {
            return apply(A, [](T x)
                         { return std::sqrt(x); });
        }

        // 正弦函数
        template <typename T>
        opt<T> sin(const Tensor<T> &A)
        {
            return apply(A, [](T x)
                         { return std::sin(x); });
        }

        // 余弦函数
        template <typename T>
        opt<T> cos(const Tensor<T> &A)
        {
            return apply(A, [](T x)
                         { return std::cos(x); });
        }

        // 均值
        template <typename T>
        T mean(const Tensor<T> &A)
        {
            return sum(A) / static_cast<T>(A.size());
        }

        // 方差
        template <typename T>
        T var(const Tensor<T> &A)
        {
            T m = mean(A);
            Tensor<T> centered = A - m;
            return sum(hadamard(centered, centered).tensor) / static_cast<T>(A.size());
        }

        // 标准差
        template <typename T>
        T stddev(const Tensor<T> &A)
        {
            return std::sqrt(var(A));
        }

        // 向量范数 (L2 norm)
        template <typename T>
        T norm(const Tensor<T> &v)
        {
            if (v.shape().size() != 1)
            {
                throw std::invalid_argument("norm requires a 1D tensor");
            }
            return std::sqrt(dot(v, v)[0]);
        }

        // 矩阵范数 (Frobenius norm)
        template <typename T>
        T frobenius_norm(const Tensor<T> &A)
        {
            if (A.shape().size() != 2)
            {
                throw std::invalid_argument("frobenius_norm requires a 2D tensor");
            }
            return std::sqrt(sum(hadamard(A, A).tensor));
        }
    } // namespace math

}

#endif // !__OPERATIONS__H__