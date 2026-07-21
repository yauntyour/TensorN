#pragma once
#ifndef __BLAS_TENSOR_HPP__
#define __BLAS_TENSOR_HPP__

#include "../tensor.hpp"
#include "../einsum.hpp"

#ifndef TENSORN_HAS_OPENBLAS
#if __has_include(<cblas.h>)
#define TENSORN_HAS_OPENBLAS 1
#elif __has_include(<openblas/cblas.h>)
#define TENSORN_HAS_OPENBLAS 1
#else
#define TENSORN_HAS_OPENBLAS 0
#endif
#endif

#if TENSORN_HAS_OPENBLAS
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#endif
#endif

#include <cmath>
#include <algorithm>
#include <type_traits>

namespace TensorN
{
    namespace blas
    {
        namespace detail
        {
            template <typename T>
            struct is_blas_type : std::bool_constant<
                std::is_same_v<T, float> || std::is_same_v<T, double>> {};
        }

        // ================================================================
        // Matrix Multiplication
        // ================================================================

        template <typename T>
        Tensor<T> matmul(const Tensor<T>& A, const Tensor<T>& B)
        {
#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if (A.shape().size() != 2 || B.shape().size() != 2)
                    throw std::invalid_argument("matmul requires 2D tensors");

                int M = static_cast<int>(A.shape()[0]);
                int K = static_cast<int>(A.shape()[1]);
                int N = static_cast<int>(B.shape()[1]);

                if (B.shape()[0] != static_cast<size_t>(K))
                    throw std::invalid_argument("Inner dimensions must match");

                Tensor<T> C({static_cast<size_t>(M), static_cast<size_t>(N)});

                if constexpr (std::is_same_v<T, float>)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A.data.data(), K, B.data.data(), N,
                        0.0f, C.data.data(), N);
                else
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0, A.data.data(), K, B.data.data(), N,
                        0.0, C.data.data(), N);

                return C;
            }
            else
#endif
            {
                return einsum<T>("ij,jk->ik", A, B).tensor;
            }
        }

        // ================================================================
        // Batched Matrix Multiplication
        // ================================================================

        template <typename T>
        Tensor<T> batched_matmul(const Tensor<T>& A, const Tensor<T>& B)
        {
            if (A.shape().size() != 3 || B.shape().size() != 3)
                throw std::invalid_argument("batched_matmul requires 3D tensors");

            size_t batch = A.shape()[0];
            size_t M = A.shape()[1];
            size_t K = A.shape()[2];
            size_t N = B.shape()[2];

            if (B.shape()[0] != batch || B.shape()[1] != K)
                throw std::invalid_argument("Inner dimensions must match");

            Tensor<T> C({batch, M, N});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    if constexpr (std::is_same_v<T, float>)
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                            1.0f, A.data.data() + b * M * K, static_cast<int>(K),
                            B.data.data() + b * K * N, static_cast<int>(N),
                            0.0f, C.data.data() + b * M * N, static_cast<int>(N));
                    else
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                            1.0, A.data.data() + b * M * K, static_cast<int>(K),
                            B.data.data() + b * K * N, static_cast<int>(N),
                            0.0, C.data.data() + b * M * N, static_cast<int>(N));
                }
                return C;
            }
            else
#endif
            {
                return einsum<T>("bij,bjk->bik", A, B).tensor;
            }
        }

        // ================================================================
        // Dot Product
        // ================================================================

        template <typename T>
        T dot(const Tensor<T>& A, const Tensor<T>& B)
        {
#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if (A.shape().size() != 1 || B.shape().size() != 1)
                    throw std::invalid_argument("dot requires 1D tensors");
                if (A.shape()[0] != B.shape()[0])
                    throw std::invalid_argument("Dimension mismatch for dot product");

                if constexpr (std::is_same_v<T, float>)
                    return cblas_sdot(static_cast<int>(A.shape()[0]), A.data.data(), 1, B.data.data(), 1);
                else
                    return cblas_ddot(static_cast<int>(A.shape()[0]), A.data.data(), 1, B.data.data(), 1);
            }
            else
#endif
            {
                return einsum<T>("i,i->", A, B).tensor[0];
            }
        }

        // ================================================================
        // Vector L2 Norm
        // ================================================================

        template <typename T>
        T norm(const Tensor<T>& v)
        {
#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if (v.shape().size() != 1)
                    throw std::invalid_argument("norm requires a 1D tensor");
                if constexpr (std::is_same_v<T, float>)
                    return cblas_snrm2(static_cast<int>(v.size()), v.data.data(), 1);
                else
                    return cblas_dnrm2(static_cast<int>(v.size()), v.data.data(), 1);
            }
            else
#endif
            {
                T sum_sq = T(0);
                for (size_t i = 0; i < v.size(); ++i) sum_sq += v[i] * v[i];
                return std::sqrt(sum_sq);
            }
        }

        // ================================================================
        // Frobenius Norm
        // ================================================================

        template <typename T>
        T frobenius_norm(const Tensor<T>& A)
        {
#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    return cblas_snrm2(static_cast<int>(A.size()), A.data.data(), 1);
                else
                    return cblas_dnrm2(static_cast<int>(A.size()), A.data.data(), 1);
            }
            else
#endif
            {
                T sum_sq = T(0);
                for (size_t i = 0; i < A.size(); ++i) sum_sq += A[i] * A[i];
                return std::sqrt(sum_sq);
            }
        }

        // ================================================================
        // AXPY: y = alpha * x + y
        // ================================================================

        template <typename T>
        void axpy(T alpha, const Tensor<T>& x, Tensor<T>& y)
        {
            if (!x.is_isomorphic(y))
                throw std::invalid_argument("axpy requires matching shapes");

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    cblas_saxpy(static_cast<int>(x.size()), alpha, x.data.data(), 1, y.data.data(), 1);
                else
                    cblas_daxpy(static_cast<int>(x.size()), alpha, x.data.data(), 1, y.data.data(), 1);
                return;
            }
#endif
            for (size_t i = 0; i < x.size(); ++i)
                y[i] += alpha * x[i];
        }

        // ================================================================
        // SCAL: x = alpha * x
        // ================================================================

        template <typename T>
        void scal(T alpha, Tensor<T>& x)
        {
#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    cblas_sscal(static_cast<int>(x.size()), alpha, x.data.data(), 1);
                else
                    cblas_dscal(static_cast<int>(x.size()), alpha, x.data.data(), 1);
                return;
            }
#endif
            for (size_t i = 0; i < x.size(); ++i)
                x[i] *= alpha;
        }

        // ================================================================
        // Outer Product
        // ================================================================

        template <typename T>
        Tensor<T> outer(const Tensor<T>& A, const Tensor<T>& B)
        {
            if (A.shape().size() != 1 || B.shape().size() != 1)
                throw std::invalid_argument("outer requires 1D tensors");

            size_t m = A.shape()[0];
            size_t n = B.shape()[0];
            Tensor<T> C({m, n});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                std::fill(C.data.begin(), C.data.end(), T(0));
                if constexpr (std::is_same_v<T, float>)
                    cblas_sger(CblasRowMajor, static_cast<int>(m), static_cast<int>(n),
                        1.0f, A.data.data(), 1, B.data.data(), 1, C.data.data(), static_cast<int>(n));
                else
                    cblas_dger(CblasRowMajor, static_cast<int>(m), static_cast<int>(n),
                        1.0, A.data.data(), 1, B.data.data(), 1, C.data.data(), static_cast<int>(n));
                return C;
            }
#endif
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    C[{i, j}] = A[i] * B[j];
            return C;
        }

        // ================================================================
        // Transpose
        // ================================================================

        template <typename T>
        Tensor<T> transpose(const Tensor<T>& A)
        {
            auto shape = A.shape();
            if (shape.size() < 2) return A;

            if (shape.size() == 2)
            {
                size_t rows = shape[0], cols = shape[1];
                Tensor<T> result({cols, rows});
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        result[{j, i}] = A[{i, j}];
                return result;
            }

            std::string in, out;
            for (size_t i = 0; i < shape.size(); ++i) in += static_cast<char>('a' + i);
            for (size_t i = shape.size(); i > 0; --i) out += static_cast<char>('a' + i - 1);
            return einsum<T>(in + "->" + out, A).tensor;
        }

        // ================================================================
        // Sum / Mean
        // ================================================================

        template <typename T>
        T sum(const Tensor<T>& A)
        {
            T result = T(0);
            for (size_t i = 0; i < A.size(); ++i) result += A[i];
            return result;
        }

        template <typename T>
        Tensor<T> sum(const Tensor<T>& A, size_t axis)
        {
            const auto& shape = A.shape();
            if (axis >= shape.size())
                throw std::invalid_argument("Axis out of range");

            std::vector<size_t> out_shape;
            for (size_t d = 0; d < shape.size(); ++d)
                if (d != axis) out_shape.push_back(shape[d]);

            Tensor<T> result(out_shape);
            std::fill(result.data.begin(), result.data.end(), T(0));

            size_t outer = 1, reduce_dim = shape[axis], inner = 1;
            for (size_t d = 0; d < axis; ++d) outer *= shape[d];
            for (size_t d = axis + 1; d < shape.size(); ++d) inner *= shape[d];

            for (size_t o = 0; o < outer; ++o)
                for (size_t r = 0; r < reduce_dim; ++r)
                    for (size_t i = 0; i < inner; ++i)
                        result[o * inner + i] += A[o * reduce_dim * inner + r * inner + i];

            return result;
        }

        template <typename T>
        T mean(const Tensor<T>& A) { return blas::sum(A) / static_cast<T>(A.size()); }

        template <typename T>
        Tensor<T> mean(const Tensor<T>& A, size_t axis)
        {
            Tensor<T> s = blas::sum(A, axis);
            scal(T(1) / static_cast<T>(A.shape()[axis]), s);
            return s;
        }

        // ================================================================
        // Max / Min
        // ================================================================

        template <typename T>
        T max(const Tensor<T>& A) { return *std::max_element(A.data.begin(), A.data.end()); }

        template <typename T>
        T min(const Tensor<T>& A) { return *std::min_element(A.data.begin(), A.data.end()); }

        // ================================================================
        // Trace
        // ================================================================

        template <typename T>
        T trace(const Tensor<T>& A)
        {
            if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
                throw std::invalid_argument("trace requires a square matrix");
            T result = T(0);
            for (size_t i = 0; i < A.shape()[0]; ++i) result += A[{i, i}];
            return result;
        }

        // ================================================================
        // Hadamard
        // ================================================================

        template <typename T>
        Tensor<T> hadamard(const Tensor<T>& A, const Tensor<T>& B)
        {
            if (!A.is_isomorphic(B))
                throw std::invalid_argument("Tensors must have same shape for Hadamard product");
            Tensor<T> result(A.shape());
            for (size_t i = 0; i < A.size(); ++i) result[i] = A[i] * B[i];
            return result;
        }

        // ================================================================
        // Variance / Stddev
        // ================================================================

        template <typename T>
        T var(const Tensor<T>& A)
        {
            T m = blas::mean(A);
            T sum_sq = T(0);
            for (size_t i = 0; i < A.size(); ++i) { T d = A[i] - m; sum_sq += d * d; }
            return sum_sq / static_cast<T>(A.size());
        }

        template <typename T>
        T stddev(const Tensor<T>& A) { return std::sqrt(var(A)); }

        // ================================================================
        // Diag / DiagMatrix
        // ================================================================

        template <typename T>
        Tensor<T> diag(const Tensor<T>& A)
        {
            if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
                throw std::invalid_argument("diag requires a square matrix");
            size_t n = A.shape()[0];
            Tensor<T> result({n});
            for (size_t i = 0; i < n; ++i) result[i] = A[{i, i}];
            return result;
        }

        template <typename T>
        Tensor<T> diag_matrix(const Tensor<T>& v)
        {
            if (v.shape().size() != 1)
                throw std::invalid_argument("diag_matrix requires a 1D tensor");
            size_t n = v.shape()[0];
            Tensor<T> result({n, n});
            for (size_t i = 0; i < n; ++i) result[{i, i}] = v[i];
            return result;
        }

        // ================================================================
        // Gram matrix: X * X^T
        // ================================================================

        template <typename T>
        Tensor<T> gram(const Tensor<T>& X)
        {
            if (X.shape().size() != 2)
                throw std::invalid_argument("gram requires 2D tensor");

            size_t M = X.shape()[0];
            size_t N = X.shape()[1];
            Tensor<T> result({M, M});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(M), static_cast<int>(M), static_cast<int>(N),
                        1.0f, X.data.data(), static_cast<int>(N),
                        X.data.data(), static_cast<int>(N),
                        0.0f, result.data.data(), static_cast<int>(M));
                else
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(M), static_cast<int>(M), static_cast<int>(N),
                        1.0, X.data.data(), static_cast<int>(N),
                        X.data.data(), static_cast<int>(N),
                        0.0, result.data.data(), static_cast<int>(M));
                return result;
            }
#endif
            return einsum<T>("ik,jk->ij", X, X).tensor;
        }

        // ================================================================
        // Bilinear form: x^T A y
        // ================================================================

        template <typename T>
        T bilinear(const Tensor<T>& x, const Tensor<T>& A, const Tensor<T>& y)
        {
            if (x.shape().size() != 1 || A.shape().size() != 2 || y.shape().size() != 1)
                throw std::invalid_argument("bilinear: x must be 1D, A must be 2D, y must be 1D");
            if (x.shape()[0] != A.shape()[0] || A.shape()[1] != y.shape()[0])
                throw std::invalid_argument("Dimension mismatch for bilinear form");

            Tensor<T> temp({A.shape()[0]});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(A.shape()[0]), static_cast<int>(A.shape()[1]),
                        1.0f, A.data.data(), static_cast<int>(A.shape()[1]),
                        y.data.data(), 1, 0.0f, temp.data.data(), 1);
                else
                    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(A.shape()[0]), static_cast<int>(A.shape()[1]),
                        1.0, A.data.data(), static_cast<int>(A.shape()[1]),
                        y.data.data(), 1, 0.0, temp.data.data(), 1);
            }
            else
#endif
            {
                for (size_t i = 0; i < A.shape()[0]; ++i) {
                    T s = T(0);
                    for (size_t j = 0; j < A.shape()[1]; ++j) s += A[{i, j}] * y[j];
                    temp[i] = s;
                }
            }

            return blas::dot(x, temp);
        }

        // ================================================================
        // Element-wise math functions
        // ================================================================

        template <typename T, typename Func>
        Tensor<T> apply(const Tensor<T>& A, Func func)
        {
            Tensor<T> result(A.shape());
            for (size_t i = 0; i < A.size(); ++i) result[i] = func(A[i]);
            return result;
        }

        template <typename T>
        Tensor<T> exp(const Tensor<T>& A)  { return apply(A, [](T x) { return std::exp(x); }); }
        template <typename T>
        Tensor<T> log(const Tensor<T>& A)  { return apply(A, [](T x) { return std::log(x); }); }
        template <typename T>
        Tensor<T> sqrt(const Tensor<T>& A) { return apply(A, [](T x) { return std::sqrt(x); }); }
        template <typename T>
        Tensor<T> sin(const Tensor<T>& A)  { return apply(A, [](T x) { return std::sin(x); }); }
        template <typename T>
        Tensor<T> cos(const Tensor<T>& A)  { return apply(A, [](T x) { return std::cos(x); }); }
        template <typename T>
        Tensor<T> abs(const Tensor<T>& A)  { return apply(A, [](T x) { return std::abs(x); }); }
        template <typename T>
        Tensor<T> pow(const Tensor<T>& A, T exponent) { return apply(A, [exponent](T x) { return std::pow(x, exponent); }); }

        // ================================================================
        // Activations
        // ================================================================

        template <typename T>
        Tensor<T> relu(const Tensor<T>& A) {
            return apply(A, [](T x) { return x > T(0) ? x : T(0); });
        }
        template <typename T>
        Tensor<T> leaky_relu(const Tensor<T>& A, T alpha) {
            return apply(A, [alpha](T x) { return x > T(0) ? x : alpha * x; });
        }
        template <typename T>
        Tensor<T> elu(const Tensor<T>& A, T alpha) {
            return apply(A, [alpha](T x) { return x > T(0) ? x : alpha * (std::exp(x) - T(1)); });
        }
        template <typename T>
        Tensor<T> sigmoid(const Tensor<T>& A) {
            return apply(A, [](T x) { return T(1) / (T(1) + std::exp(-x)); });
        }
        template <typename T>
        Tensor<T> tanh(const Tensor<T>& A) {
            return apply(A, [](T x) { return std::tanh(x); });
        }
        template <typename T>
        Tensor<T> gelu(const Tensor<T>& A) {
            return apply(A, [](T x) {
                T a = T(0.7978845608028654) * (x + T(0.044715) * x * x * x);
                return T(0.5) * x * (T(1) + std::tanh(a));
            });
        }

        // ================================================================
        // Element-wise addition
        // ================================================================

        template <typename T>
        Tensor<T> add(const Tensor<T>& A, const Tensor<T>& B)
        {
            if (!A.is_isomorphic(B))
                throw std::invalid_argument("Tensors must have same shape for addition");
            Tensor<T> result(A.shape());
            for (size_t i = 0; i < A.size(); ++i) result[i] = A[i] + B[i];
            return result;
        }

        // ================================================================
        // Softmax
        // ================================================================

        template <typename T>
        Tensor<T> softmax(const Tensor<T>& A, int axis = -1)
        {
            size_t ndim = A.shape().size();
            if (axis < 0) axis = static_cast<int>(ndim) + axis;
            if (axis < 0 || static_cast<size_t>(axis) >= ndim)
                throw std::invalid_argument("Softmax axis out of range");

            Tensor<T> result(A.shape());

            if (ndim == 1) {
                T max_val = blas::max(A);
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

            throw std::runtime_error("Softmax: only 1D/2D supported");
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
                throw std::invalid_argument("Argmax axis out of range");

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
                throw std::invalid_argument("Argmin axis out of range");

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
                throw std::invalid_argument("Tensors must have same shape for equal");
            Tensor<int> result(A.shape());
            for (size_t i = 0; i < A.size(); ++i)
                result[i] = (A[i] == B[i]) ? 1 : 0;
            return result;
        }

        template <typename T>
        Tensor<int> greater(const Tensor<T>& A, const Tensor<T>& B)
        {
            if (!A.is_isomorphic(B))
                throw std::invalid_argument("Tensors must have same shape for greater");
            Tensor<int> result(A.shape());
            for (size_t i = 0; i < A.size(); ++i)
                result[i] = (A[i] > B[i]) ? 1 : 0;
            return result;
        }

        // ================================================================
        // 2D Convolution
        // ================================================================

        template <typename T>
        Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& weight,
                         const Tensor<T>& bias, int stride = 1, int padding = 0)
        {
            if (input.shape().size() != 4 || weight.shape().size() != 4)
                throw std::invalid_argument("conv2d: input and weight must be 4D");
            if (bias.shape().size() != 1)
                throw std::invalid_argument("conv2d: bias must be 1D");

            size_t N = input.shape()[0], C = input.shape()[1];
            size_t H = input.shape()[2], W = input.shape()[3];
            size_t K = weight.shape()[0];
            size_t kH = weight.shape()[2], kW = weight.shape()[3];

            int64_t oH = static_cast<int64_t>((H + 2 * padding - kH) / stride) + 1;
            int64_t oW = static_cast<int64_t>((W + 2 * padding - kW) / stride) + 1;

            if (oH <= 0 || oW <= 0)
                throw std::invalid_argument("conv2d: invalid output dimensions");

            Tensor<T> output({N, K, static_cast<size_t>(oH), static_cast<size_t>(oW)});

            for (size_t n = 0; n < N; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    for (size_t oh = 0; oh < static_cast<size_t>(oH); ++oh) {
                        for (size_t ow = 0; ow < static_cast<size_t>(oW); ++ow) {
                            T val = bias[k];
                            for (size_t c = 0; c < C; ++c) {
                                for (size_t kh = 0; kh < kH; ++kh) {
                                    for (size_t kw = 0; kw < kW; ++kw) {
                                        int64_t ih = static_cast<int64_t>(oh * stride + kh) - padding;
                                        int64_t iw = static_cast<int64_t>(ow * stride + kw) - padding;
                                        if (ih >= 0 && ih < static_cast<int64_t>(H) &&
                                            iw >= 0 && iw < static_cast<int64_t>(W)) {
                                            val += input[{n, c, static_cast<size_t>(ih), static_cast<size_t>(iw)}]
                                                 * weight[{k, c, kh, kw}];
                                        }
                                    }
                                }
                            }
                            output[{n, k, oh, ow}] = val;
                        }
                    }
                }
            }
            return output;
        }

        template <typename T>
        Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& weight,
                         int stride = 1, int padding = 0)
        {
            if (weight.shape().size() != 4)
                throw std::invalid_argument("conv2d: weight must be 4D");
            Tensor<T> bias({weight.shape()[0]});
            return conv2d(input, weight, bias, stride, padding);
        }

        // ================================================================
        // Transposed 2D Convolution
        // ================================================================

        template <typename T>
        Tensor<T> conv_transpose2d(const Tensor<T>& input, const Tensor<T>& weight,
                                   const Tensor<T>& bias, int stride = 1, int padding = 0)
        {
            if (input.shape().size() != 4 || weight.shape().size() != 4)
                throw std::invalid_argument("conv_transpose2d: input and weight must be 4D");
            if (bias.shape().size() != 1)
                throw std::invalid_argument("conv_transpose2d: bias must be 1D");

            size_t N = input.shape()[0], C = input.shape()[1];
            size_t H = input.shape()[2], W = input.shape()[3];
            size_t K = weight.shape()[0];
            size_t kH = weight.shape()[2], kW = weight.shape()[3];

            size_t oH = (H - 1) * stride + kH - 2 * padding;
            size_t oW = (W - 1) * stride + kW - 2 * padding;

            if (oH == 0 || oW == 0)
                throw std::invalid_argument("conv_transpose2d: invalid output dimensions");

            Tensor<T> output({N, K, oH, oW});

            for (size_t n = 0; n < N; ++n) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            T in_val = input[{n, c, h, w}];
                            for (size_t k = 0; k < K; ++k) {
                                for (size_t kh = 0; kh < kH; ++kh) {
                                    for (size_t kw = 0; kw < kW; ++kw) {
                                        int64_t oh = static_cast<int64_t>(h * stride + kh) - padding;
                                        int64_t ow = static_cast<int64_t>(w * stride + kw) - padding;
                                        if (oh >= 0 && oh < static_cast<int64_t>(oH) &&
                                            ow >= 0 && ow < static_cast<int64_t>(oW)) {
                                            output[{n, k, static_cast<size_t>(oh), static_cast<size_t>(ow)}]
                                                += in_val * weight[{k, c, kh, kw}];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (size_t n = 0; n < N; ++n)
                for (size_t k = 0; k < K; ++k)
                    for (size_t oh = 0; oh < oH; ++oh)
                        for (size_t ow = 0; ow < oW; ++ow)
                            output[{n, k, oh, ow}] += bias[k];

            return output;
        }

    } // namespace blas
} // namespace TensorN

#endif // __BLAS_TENSOR_HPP__
