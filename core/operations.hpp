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
            TENSOR_THROW("dot requires 1D tensors");
        }
        if (A.shape()[0] != B.shape()[0])
        {
            TENSOR_THROW("Dimension mismatch for dot product");
        }
        return einsum<T>("i,i->", A, B);
    }

    // 外积 (Outer product)
    template <typename T>
    opt<T> outer(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 1 || B.shape().size() != 1)
        {
            TENSOR_THROW("outer requires 1D tensors");
        }
        return einsum<T>("i,j->ij", A, B);
    }

    // 矩阵乘法 (Matrix multiplication)
    template <typename T>
    opt<T> matmul(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 2 || B.shape().size() != 2)
        {
            TENSOR_THROW("matmul requires 2D tensors");
        }
        if (A.shape()[1] != B.shape()[0])
        {
            TENSOR_THROW("Dimension mismatch for matrix multiplication");
        }
        return einsum<T>("ij,jk->ik", A, B);
    }

    // Hadamard 积 (逐元素乘)
    template <typename T>
    opt<T> hadamard(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (!A.is_isomorphic(B))
        {
            TENSOR_THROW("Tensors must have same shape for Hadamard product");
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
            TENSOR_THROW("bilinear: x must be 1D, A must be 2D, y must be 1D");
        }
        if (x.shape()[0] != A.shape()[0] || A.shape()[1] != y.shape()[0])
        {
            TENSOR_THROW("Dimension mismatch for bilinear form");
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
            TENSOR_THROW("gram requires 2D tensor");
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
                TENSOR_THROW("Axis out of range");
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
            TENSOR_THROW("trace requires a square matrix");
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
            TENSOR_THROW("Axis out of range");
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
            TENSOR_THROW("Number of axes must match tensor dimension");
        }

        std::vector<bool> used(shape.size(), false);
        for (auto axis : new_axes)
        {
            if (axis >= shape.size())
            {
                TENSOR_THROW("Axis out of range");
            }
            if (used[axis])
            {
                TENSOR_THROW("Duplicate axis");
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
            TENSOR_THROW("diag requires a square matrix");
        }
        return einsum<T>("ii->i", A);
    }

    // 创建对角矩阵
    template <typename T>
    opt<T> diag_matrix(const Tensor<T> &v)
    {
        if (v.shape().size() != 1)
        {
            TENSOR_THROW("diag_matrix requires a 1D tensor");
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
                TENSOR_THROW("norm requires a 1D tensor");
            }
            return std::sqrt(dot(v, v)[0]);
        }

        // 矩阵范数 (Frobenius norm)
        template <typename T>
        T frobenius_norm(const Tensor<T> &A)
        {
            if (A.shape().size() != 2)
            {
                TENSOR_THROW("frobenius_norm requires a 2D tensor");
            }
            return std::sqrt(sum(hadamard(A, A).tensor));
        }
    } // namespace math

    // ================================================================
    // Softmax
    // ================================================================

    template <typename T>
    Tensor<T> softmax(const Tensor<T>& A, int axis = -1)
    {
        size_t ndim = A.shape().size();
        if (axis < 0) axis = static_cast<int>(ndim) + axis;
        if (axis < 0 || static_cast<size_t>(axis) >= ndim)
            TENSOR_THROW("Softmax axis out of range");

        Tensor<T> result(A.shape());

        if (ndim == 1) {
            T max_val = *std::max_element(A.begin(), A.end());
            T sum = T(0);
            for (size_t i = 0; i < A.size(); ++i) {
                result[i] = std::exp(A[i] - max_val);
                sum += result[i];
            }
            for (size_t i = 0; i < A.size(); ++i)
                result[i] /= sum;
            return result;
        }

        if (ndim == 2) {
            size_t rows = A.shape()[0], cols = A.shape()[1];
            if (axis == 1) {
                for (size_t r = 0; r < rows; ++r) {
                    T max_val = A[{r, 0}];
                    for (size_t c = 1; c < cols; ++c)
                        if (A[{r, c}] > max_val) max_val = A[{r, c}];
                    T sum = T(0);
                    for (size_t c = 0; c < cols; ++c) {
                        result[{r, c}] = std::exp(A[{r, c}] - max_val);
                        sum += result[{r, c}];
                    }
                    for (size_t c = 0; c < cols; ++c)
                        result[{r, c}] /= sum;
                }
                return result;
            }
            if (axis == 0) {
                for (size_t c = 0; c < cols; ++c) {
                    T max_val = A[{0, c}];
                    for (size_t r = 1; r < rows; ++r)
                        if (A[{r, c}] > max_val) max_val = A[{r, c}];
                    T sum = T(0);
                    for (size_t r = 0; r < rows; ++r) {
                        result[{r, c}] = std::exp(A[{r, c}] - max_val);
                        sum += result[{r, c}];
                    }
                    for (size_t r = 0; r < rows; ++r)
                        result[{r, c}] /= sum;
                }
                return result;
            }
        }
        TENSOR_THROW("Softmax: only 1D/2D supported");
    }

    // ================================================================
    // Argmax / Argmin
    // ================================================================

    template <typename T>
    Tensor<int64_t> argmax(const Tensor<T>& A, int axis = -1)
    {
        const auto& shape = A.shape();
        size_t ndim = shape.size();
        if (axis < 0) axis = static_cast<int>(ndim) + axis;
        if (axis < 0 || static_cast<size_t>(axis) >= ndim)
            TENSOR_THROW("Argmax axis out of range");

        size_t outer = 1, reduce_dim = shape[axis], inner = 1;
        for (size_t d = 0; d < static_cast<size_t>(axis); ++d) outer *= shape[d];
        for (size_t d = static_cast<size_t>(axis) + 1; d < ndim; ++d) inner *= shape[d];

        std::vector<size_t> out_shape;
        for (size_t d = 0; d < ndim; ++d)
            if (d != static_cast<size_t>(axis)) out_shape.push_back(shape[d]);

        Tensor<int64_t> result(out_shape);
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                int64_t best_idx = 0;
                T best_val = A[o * reduce_dim * inner + i];
                for (size_t r = 1; r < reduce_dim; ++r) {
                    T v = A[o * reduce_dim * inner + r * inner + i];
                    if (v > best_val) { best_val = v; best_idx = static_cast<int64_t>(r); }
                }
                result[o * inner + i] = best_idx;
            }
        }
        return result;
    }

    template <typename T>
    Tensor<int64_t> argmin(const Tensor<T>& A, int axis = -1)
    {
        const auto& shape = A.shape();
        size_t ndim = shape.size();
        if (axis < 0) axis = static_cast<int>(ndim) + axis;
        if (axis < 0 || static_cast<size_t>(axis) >= ndim)
            TENSOR_THROW("Argmin axis out of range");

        size_t outer = 1, reduce_dim = shape[axis], inner = 1;
        for (size_t d = 0; d < static_cast<size_t>(axis); ++d) outer *= shape[d];
        for (size_t d = static_cast<size_t>(axis) + 1; d < ndim; ++d) inner *= shape[d];

        std::vector<size_t> out_shape;
        for (size_t d = 0; d < ndim; ++d)
            if (d != static_cast<size_t>(axis)) out_shape.push_back(shape[d]);

        Tensor<int64_t> result(out_shape);
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                int64_t best_idx = 0;
                T best_val = A[o * reduce_dim * inner + i];
                for (size_t r = 1; r < reduce_dim; ++r) {
                    T v = A[o * reduce_dim * inner + r * inner + i];
                    if (v < best_val) { best_val = v; best_idx = static_cast<int64_t>(r); }
                }
                result[o * inner + i] = best_idx;
            }
        }
        return result;
    }

    // ================================================================
    // Comparison operations
    // ================================================================

    template <typename T>
    Tensor<int> equal(const Tensor<T>& A, const Tensor<T>& B)
    {
        if (!A.is_isomorphic(B))
            TENSOR_THROW("Tensors must have same shape for equal");
        Tensor<int> result(A.shape());
        for (size_t i = 0; i < A.size(); ++i)
            result[i] = (A[i] == B[i]) ? 1 : 0;
        return result;
    }

    template <typename T>
    Tensor<int> greater(const Tensor<T>& A, const Tensor<T>& B)
    {
        if (!A.is_isomorphic(B))
            TENSOR_THROW("Tensors must have same shape for greater");
        Tensor<int> result(A.shape());
        for (size_t i = 0; i < A.size(); ++i)
            result[i] = (A[i] > B[i]) ? 1 : 0;
        return result;
    }

    // ================================================================
    // 2D Convolution (Native)
    // ================================================================

    template <typename T>
    Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& weight,
                     const Tensor<T>& bias, int stride = 1, int padding = 0)
    {
        if (input.shape().size() != 4 || weight.shape().size() != 4)
            TENSOR_THROW("conv2d: input and weight must be 4D");
        if (bias.shape().size() != 1)
            TENSOR_THROW("conv2d: bias must be 1D");

        size_t N = input.shape()[0], C = input.shape()[1];
        size_t H = input.shape()[2], W = input.shape()[3];
        size_t K = weight.shape()[0];
        size_t kH = weight.shape()[2], kW = weight.shape()[3];

        int64_t oH = static_cast<int64_t>((H + 2 * padding - kH) / stride) + 1;
        int64_t oW = static_cast<int64_t>((W + 2 * padding - kW) / stride) + 1;

        Tensor<T> output({N, K, static_cast<size_t>(oH), static_cast<size_t>(oW)});

        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < K; ++k)
                for (size_t oh = 0; oh < static_cast<size_t>(oH); ++oh)
                    for (size_t ow = 0; ow < static_cast<size_t>(oW); ++ow) {
                        T val = bias[k];
                        for (size_t c = 0; c < C; ++c)
                            for (size_t kh = 0; kh < kH; ++kh)
                                for (size_t kw = 0; kw < kW; ++kw) {
                                    int64_t ih = static_cast<int64_t>(oh * stride + kh) - padding;
                                    int64_t iw = static_cast<int64_t>(ow * stride + kw) - padding;
                                    if (ih >= 0 && ih < static_cast<int64_t>(H) &&
                                        iw >= 0 && iw < static_cast<int64_t>(W))
                                        val += input[{n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)}]
                                             * weight[{k, c, kh, kw}];
                                }
                        output[{n, k, oh, ow}] = val;
                    }
        return output;
    }

    // ================================================================
    // Transposed 2D Convolution (Native)
    // ================================================================

    template <typename T>
    Tensor<T> conv_transpose2d(const Tensor<T>& input, const Tensor<T>& weight,
                               const Tensor<T>& bias, int stride = 1, int padding = 0)
    {
        if (input.shape().size() != 4 || weight.shape().size() != 4)
            TENSOR_THROW("conv_transpose2d: input and weight must be 4D");
        if (bias.shape().size() != 1)
            TENSOR_THROW("conv_transpose2d: bias must be 1D");

        size_t N = input.shape()[0], C = input.shape()[1];
        size_t H = input.shape()[2], W = input.shape()[3];
        size_t K = weight.shape()[0];
        size_t kH = weight.shape()[2], kW = weight.shape()[3];

        size_t oH = (H - 1) * stride + kH - 2 * padding;
        size_t oW = (W - 1) * stride + kW - 2 * padding;

        Tensor<T> output({N, K, oH, oW});

        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < C; ++c)
                for (size_t h = 0; h < H; ++h)
                    for (size_t w = 0; w < W; ++w) {
                        T in_val = input[{n, c, h, w}];
                        for (size_t k = 0; k < K; ++k)
                            for (size_t kh = 0; kh < kH; ++kh)
                                for (size_t kw = 0; kw < kW; ++kw) {
                                    int64_t oh = static_cast<int64_t>(h * stride + kh) - padding;
                                    int64_t ow = static_cast<int64_t>(w * stride + kw) - padding;
                                    if (oh >= 0 && oh < static_cast<int64_t>(oH) &&
                                        ow >= 0 && ow < static_cast<int64_t>(oW))
                                        output[{n, k, static_cast<size_t>(oh), static_cast<size_t>(ow)}]
                                            += in_val * weight[{k, c, kh, kw}];
                                }
                    }

        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < K; ++k)
                for (size_t oh = 0; oh < oH; ++oh)
                    for (size_t ow = 0; ow < oW; ++ow)
                        output[{n, k, oh, ow}] += bias[k];

        return output;
    }

}

#endif // !__OPERATIONS__H__