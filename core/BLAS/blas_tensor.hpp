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

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <algorithm>
#include <type_traits>
#include <cstring>

namespace TensorN
{
    namespace blas
    {
        namespace detail
        {
            template <typename T>
            struct is_blas_type : std::bool_constant<
                std::is_same_v<T, float> || std::is_same_v<T, double>> {};

            inline int get_num_threads()
            {
#ifdef _OPENMP
                return omp_get_max_threads();
#else
                return 1;
#endif
            }
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
                    TENSOR_THROW("matmul requires 2D tensors");

                int M = static_cast<int>(A.shape()[0]);
                int K = static_cast<int>(A.shape()[1]);
                int N = static_cast<int>(B.shape()[1]);

                if (B.shape()[0] != static_cast<size_t>(K))
                    TENSOR_THROW("Inner dimensions must match");

                Tensor<T> C({static_cast<size_t>(M), static_cast<size_t>(N)});

                if constexpr (std::is_same_v<T, float>)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0f, A.data->data(), K, B.data->data(), N,
                        0.0f, C.data->data(), N);
                else
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K, 1.0, A.data->data(), K, B.data->data(), N,
                        0.0, C.data->data(), N);

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
                TENSOR_THROW("batched_matmul requires 3D tensors");

            size_t batch = A.shape()[0];
            size_t M = A.shape()[1];
            size_t K = A.shape()[2];
            size_t N = B.shape()[2];

            if (B.shape()[0] != batch || B.shape()[1] != K)
                TENSOR_THROW("Inner dimensions must match");

            Tensor<T> C({batch, M, N});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    if constexpr (std::is_same_v<T, float>)
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                            1.0f, A.data->data() + b * M * K, static_cast<int>(K),
                            B.data->data() + b * K * N, static_cast<int>(N),
                            0.0f, C.data->data() + b * M * N, static_cast<int>(N));
                    else
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                            1.0, A.data->data() + b * M * K, static_cast<int>(K),
                            B.data->data() + b * K * N, static_cast<int>(N),
                            0.0, C.data->data() + b * M * N, static_cast<int>(N));
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
                    TENSOR_THROW("dot requires 1D tensors");
                if (A.shape()[0] != B.shape()[0])
                    TENSOR_THROW("Dimension mismatch for dot product");

                if constexpr (std::is_same_v<T, float>)
                    return cblas_sdot(static_cast<int>(A.shape()[0]), A.data->data(), 1, B.data->data(), 1);
                else
                    return cblas_ddot(static_cast<int>(A.shape()[0]), A.data->data(), 1, B.data->data(), 1);
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
                    TENSOR_THROW("norm requires a 1D tensor");
                if constexpr (std::is_same_v<T, float>)
                    return cblas_snrm2(static_cast<int>(v.size()), v.data->data(), 1);
                else
                    return cblas_dnrm2(static_cast<int>(v.size()), v.data->data(), 1);
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
                    return cblas_snrm2(static_cast<int>(A.size()), A.data->data(), 1);
                else
                    return cblas_dnrm2(static_cast<int>(A.size()), A.data->data(), 1);
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
                TENSOR_THROW("axpy requires matching shapes");

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    cblas_saxpy(static_cast<int>(x.size()), alpha, x.data->data(), 1, y.data->data(), 1);
                else
                    cblas_daxpy(static_cast<int>(x.size()), alpha, x.data->data(), 1, y.data->data(), 1);
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
                    cblas_sscal(static_cast<int>(x.size()), alpha, x.data->data(), 1);
                else
                    cblas_dscal(static_cast<int>(x.size()), alpha, x.data->data(), 1);
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
                TENSOR_THROW("outer requires 1D tensors");

            size_t m = A.shape()[0];
            size_t n = B.shape()[0];
            Tensor<T> C({m, n});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                std::fill(C.data->begin(), C.data->end(), T(0));
                if constexpr (std::is_same_v<T, float>)
                    cblas_sger(CblasRowMajor, static_cast<int>(m), static_cast<int>(n),
                        1.0f, A.data->data(), 1, B.data->data(), 1, C.data->data(), static_cast<int>(n));
                else
                    cblas_dger(CblasRowMajor, static_cast<int>(m), static_cast<int>(n),
                        1.0, A.data->data(), 1, B.data->data(), 1, C.data->data(), static_cast<int>(n));
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
                const T* __restrict src = A.data->data();
                T* __restrict dst = result.data->data();
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(rows); ++i)
                    for (int64_t j = 0; j < static_cast<int64_t>(cols); ++j)
                        dst[j * rows + i] = src[i * cols + j];
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
            const T* __restrict src = A.data->data();
            size_t n = A.size();
            #pragma omp parallel for reduction(+:result) schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) result += src[i];
            return result;
        }

        template <typename T>
        Tensor<T> sum(const Tensor<T>& A, size_t axis)
        {
            const auto& shape = A.shape();
            if (axis >= shape.size())
                TENSOR_THROW("Axis out of range");

            std::vector<size_t> out_shape;
            for (size_t d = 0; d < shape.size(); ++d)
                if (d != axis) out_shape.push_back(shape[d]);

            Tensor<T> result(out_shape);
            std::fill(result.data->begin(), result.data->end(), T(0));

            size_t outer = 1, reduce_dim = shape[axis], inner = 1;
            for (size_t d = 0; d < axis; ++d) outer *= shape[d];
            for (size_t d = axis + 1; d < shape.size(); ++d) inner *= shape[d];

            const T* __restrict src = A.data->data();
            T* __restrict dst = result.data->data();

            #pragma omp parallel for schedule(static)
            for (int64_t oi = 0; oi < static_cast<int64_t>(outer * inner); ++oi)
            {
                size_t o = static_cast<size_t>(oi) / inner;
                size_t i = static_cast<size_t>(oi) % inner;
                T s = T(0);
                for (size_t r = 0; r < reduce_dim; ++r)
                    s += src[o * reduce_dim * inner + r * inner + i];
                dst[o * inner + i] = s;
            }

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
        T max(const Tensor<T>& A) { return *std::max_element(A.data->begin(), A.data->end()); }

        template <typename T>
        T min(const Tensor<T>& A) { return *std::min_element(A.data->begin(), A.data->end()); }

        // ================================================================
        // Trace
        // ================================================================

        template <typename T>
        T trace(const Tensor<T>& A)
        {
            if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
                TENSOR_THROW("trace requires a square matrix");
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
                TENSOR_THROW("Tensors must have same shape for Hadamard product");
            Tensor<T> result(A.shape());
            const T* __restrict a = A.data->data();
            const T* __restrict b = B.data->data();
            T* __restrict c = result.data->data();
            size_t n = A.size();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) c[i] = a[i] * b[i];
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
            const T* __restrict src = A.data->data();
            size_t n = A.size();
            #pragma omp parallel for reduction(+:sum_sq) schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) { T d = src[i] - m; sum_sq += d * d; }
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
                TENSOR_THROW("diag requires a square matrix");
            size_t n = A.shape()[0];
            Tensor<T> result({n});
            for (size_t i = 0; i < n; ++i) result[i] = A[{i, i}];
            return result;
        }

        template <typename T>
        Tensor<T> diag_matrix(const Tensor<T>& v)
        {
            if (v.shape().size() != 1)
                TENSOR_THROW("diag_matrix requires a 1D tensor");
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
                TENSOR_THROW("gram requires 2D tensor");

            size_t M = X.shape()[0];
            size_t N = X.shape()[1];
            Tensor<T> result({M, M});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(M), static_cast<int>(M), static_cast<int>(N),
                        1.0f, X.data->data(), static_cast<int>(N),
                        X.data->data(), static_cast<int>(N),
                        0.0f, result.data->data(), static_cast<int>(M));
                else
                    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        static_cast<int>(M), static_cast<int>(M), static_cast<int>(N),
                        1.0, X.data->data(), static_cast<int>(N),
                        X.data->data(), static_cast<int>(N),
                        0.0, result.data->data(), static_cast<int>(M));
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
                TENSOR_THROW("bilinear: x must be 1D, A must be 2D, y must be 1D");
            if (x.shape()[0] != A.shape()[0] || A.shape()[1] != y.shape()[0])
                TENSOR_THROW("Dimension mismatch for bilinear form");

            Tensor<T> temp({A.shape()[0]});

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                if constexpr (std::is_same_v<T, float>)
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(A.shape()[0]), static_cast<int>(A.shape()[1]),
                        1.0f, A.data->data(), static_cast<int>(A.shape()[1]),
                        y.data->data(), 1, 0.0f, temp.data->data(), 1);
                else
                    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(A.shape()[0]), static_cast<int>(A.shape()[1]),
                        1.0, A.data->data(), static_cast<int>(A.shape()[1]),
                        y.data->data(), 1, 0.0, temp.data->data(), 1);
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
            const T* __restrict src = A.data->data();
            T* __restrict dst = result.data->data();
            size_t n = A.size();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) dst[i] = func(src[i]);
            return result;
        }

        template <typename T, typename Func>
        void apply_inplace(Tensor<T>& A, Func func)
        {
            T* __restrict dst = A.data->data();
            size_t n = A.size();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) dst[i] = func(dst[i]);
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
                TENSOR_THROW("Tensors must have same shape for addition");
            Tensor<T> result(A.shape());
            const T* __restrict a = A.data->data();
            const T* __restrict b = B.data->data();
            T* __restrict c = result.data->data();
            size_t n = A.size();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) c[i] = a[i] + b[i];
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
                TENSOR_THROW("Softmax axis out of range");

            Tensor<T> result(A.shape());

            if (ndim == 1) {
                T max_val = blas::max(A);
                T sum = T(0);
                T* __restrict dst = result.data->data();
                const T* __restrict src = A.data->data();
                size_t n = A.size();
                for (size_t i = 0; i < n; ++i) {
                    dst[i] = std::exp(src[i] - max_val);
                    sum += dst[i];
                }
                #pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) dst[i] /= sum;
                return result;
            }

            if (ndim == 2) {
                size_t rows = A.shape()[0], cols = A.shape()[1];
                const T* __restrict src = A.data->data();
                T* __restrict dst = result.data->data();
                if (axis == 1) {
                    #pragma omp parallel for schedule(static)
                    for (int64_t r = 0; r < static_cast<int64_t>(rows); ++r) {
                        T max_val = src[r * cols];
                        for (size_t c = 1; c < cols; ++c)
                            if (src[r * cols + c] > max_val) max_val = src[r * cols + c];
                        T sum = T(0);
                        for (size_t c = 0; c < cols; ++c) {
                            dst[r * cols + c] = std::exp(src[r * cols + c] - max_val);
                            sum += dst[r * cols + c];
                        }
                        for (size_t c = 0; c < cols; ++c)
                            dst[r * cols + c] /= sum;
                    }
                    return result;
                }
                if (axis == 0) {
                    #pragma omp parallel for schedule(static)
                    for (int64_t c = 0; c < static_cast<int64_t>(cols); ++c) {
                        T max_val = src[c];
                        for (size_t r = 1; r < rows; ++r)
                            if (src[r * cols + c] > max_val) max_val = src[r * cols + c];
                        T sum = T(0);
                        for (size_t r = 0; r < rows; ++r) {
                            dst[r * cols + c] = std::exp(src[r * cols + c] - max_val);
                            sum += dst[r * cols + c];
                        }
                        for (size_t r = 0; r < rows; ++r)
                            dst[r * cols + c] /= sum;
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
            const T* __restrict src = A.data->data();
            int64_t* __restrict dst = result.data->data();

            #pragma omp parallel for schedule(static)
            for (int64_t oi = 0; oi < static_cast<int64_t>(outer * inner); ++oi) {
                size_t o = static_cast<size_t>(oi) / inner;
                size_t i = static_cast<size_t>(oi) % inner;
                int64_t best_idx = 0;
                T best_val = src[o * reduce_dim * inner + i];
                for (size_t r = 1; r < reduce_dim; ++r) {
                    T v = src[o * reduce_dim * inner + r * inner + i];
                    if (v > best_val) { best_val = v; best_idx = static_cast<int64_t>(r); }
                }
                dst[o * inner + i] = best_idx;
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
            const T* __restrict src = A.data->data();
            int64_t* __restrict dst = result.data->data();

            #pragma omp parallel for schedule(static)
            for (int64_t oi = 0; oi < static_cast<int64_t>(outer * inner); ++oi) {
                size_t o = static_cast<size_t>(oi) / inner;
                size_t i = static_cast<size_t>(oi) % inner;
                int64_t best_idx = 0;
                T best_val = src[o * reduce_dim * inner + i];
                for (size_t r = 1; r < reduce_dim; ++r) {
                    T v = src[o * reduce_dim * inner + r * inner + i];
                    if (v < best_val) { best_val = v; best_idx = static_cast<int64_t>(r); }
                }
                dst[o * inner + i] = best_idx;
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
            const T* __restrict a = A.data->data();
            const T* __restrict b = B.data->data();
            int* __restrict c = result.data->data();
            size_t n = A.size();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i)
                c[i] = (a[i] == b[i]) ? 1 : 0;
            return result;
        }

        template <typename T>
        Tensor<int> greater(const Tensor<T>& A, const Tensor<T>& B)
        {
            if (!A.is_isomorphic(B))
                TENSOR_THROW("Tensors must have same shape for greater");
            Tensor<int> result(A.shape());
            const T* __restrict a = A.data->data();
            const T* __restrict b = B.data->data();
            int* __restrict c = result.data->data();
            size_t n = A.size();
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(n); ++i)
                c[i] = (a[i] > b[i]) ? 1 : 0;
            return result;
        }

        // ================================================================
        // 2D Convolution
        // ================================================================

        namespace detail
        {
            template <typename T>
            void im2col(const T* input, size_t C, size_t H, size_t W,
                        size_t kH, size_t kW, int stride, int padding,
                        size_t oH, size_t oW, T* col)
            {
                size_t col_row_size = C * kH * kW;
                #pragma omp parallel for schedule(static)
                for (int64_t oh = 0; oh < static_cast<int64_t>(oH); ++oh)
                {
                    for (size_t ow = 0; ow < oW; ++ow)
                    {
                        T* col_row = col + (oh * oW + ow) * col_row_size;
                        size_t col_idx = 0;
                        for (size_t c = 0; c < C; ++c)
                            for (size_t kh = 0; kh < kH; ++kh)
                                for (size_t kw = 0; kw < kW; ++kw)
                                {
                                    int ih = static_cast<int>(oh * stride + kh) - padding;
                                    int iw = static_cast<int>(ow * stride + kw) - padding;
                                    if (ih >= 0 && ih < static_cast<int>(H) &&
                                        iw >= 0 && iw < static_cast<int>(W))
                                        col_row[col_idx] = input[(c * H + ih) * W + iw];
                                    else
                                        col_row[col_idx] = T(0);
                                    ++col_idx;
                                }
                    }
                }
            }

            template <typename T>
            void col2im(const T* col, size_t C, size_t H, size_t W,
                        size_t kH, size_t kW, int stride, int padding,
                        size_t oH, size_t oW, T* input)
            {
                size_t input_size = C * H * W;
                std::memset(input, 0, input_size * sizeof(T));
                size_t col_row_size = C * kH * kW;

                for (size_t oh = 0; oh < oH; ++oh)
                    for (size_t ow = 0; ow < oW; ++ow)
                    {
                        const T* col_row = col + (oh * oW + ow) * col_row_size;
                        size_t col_idx = 0;
                        for (size_t c = 0; c < C; ++c)
                            for (size_t kh = 0; kh < kH; ++kh)
                                for (size_t kw = 0; kw < kW; ++kw)
                                {
                                    int ih = static_cast<int>(oh * stride + kh) - padding;
                                    int iw = static_cast<int>(ow * stride + kw) - padding;
                                    if (ih >= 0 && ih < static_cast<int>(H) &&
                                        iw >= 0 && iw < static_cast<int>(W))
                                        input[(c * H + ih) * W + iw] += col_row[col_idx];
                                    ++col_idx;
                                }
                    }
            }
        }

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

            if (oH <= 0 || oW <= 0)
                TENSOR_THROW("conv2d: invalid output dimensions");

            Tensor<T> output({N, K, static_cast<size_t>(oH), static_cast<size_t>(oW)});

            size_t col_size = C * kH * kW * oH * oW;

#if TENSORN_HAS_OPENBLAS
            if constexpr (detail::is_blas_type<T>::value)
            {
                std::vector<T> col(col_size);
                const T* weight_ptr = weight.data->data();
                T* output_ptr = output.data->data();
                const T* bias_ptr = bias.data->data();

                int M = static_cast<int>(K);
                int Nn = static_cast<int>(oH * oW);
                int Kk = static_cast<int>(C * kH * kW);

                for (size_t n = 0; n < N; ++n)
                {
                    const T* input_batch = input.data->data() + n * C * H * W;
                    T* output_batch = output_ptr + n * K * oH * oW;

                    detail::im2col(input_batch, C, H, W, kH, kW, stride, padding,
                                   static_cast<size_t>(oH), static_cast<size_t>(oW), col.data());

                    if constexpr (std::is_same_v<T, float>)
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, Nn, Kk, 1.0f, weight_ptr, Kk,
                            col.data(), Nn, 0.0f, output_batch, Nn);
                    else
                        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            M, Nn, Kk, 1.0, weight_ptr, Kk,
                            col.data(), Nn, 0.0, output_batch, Nn);

                    #pragma omp parallel for schedule(static)
                    for (int64_t k = 0; k < static_cast<int64_t>(K); ++k)
                        for (int64_t hw = 0; hw < static_cast<int64_t>(oH * oW); ++hw)
                            output_batch[k * oH * oW + hw] += bias_ptr[k];
                }
                return output;
            }
            else
#endif
            {
                const T* __restrict input_ptr = input.data->data();
                const T* __restrict weight_ptr = weight.data->data();
                const T* __restrict bias_ptr = bias.data->data();
                T* __restrict output_ptr = output.data->data();

                #pragma omp parallel for schedule(static)
                for (int64_t n = 0; n < static_cast<int64_t>(N); ++n)
                    for (size_t k = 0; k < K; ++k)
                        for (size_t oh = 0; oh < static_cast<size_t>(oH); ++oh)
                            for (size_t ow = 0; ow < static_cast<size_t>(oW); ++ow) {
                                T val = bias_ptr[k];
                                for (size_t c = 0; c < C; ++c)
                                    for (size_t kh = 0; kh < kH; ++kh)
                                        for (size_t kw = 0; kw < kW; ++kw) {
                                            int64_t ih = static_cast<int64_t>(oh * stride + kh) - padding;
                                            int64_t iw = static_cast<int64_t>(ow * stride + kw) - padding;
                                            if (ih >= 0 && ih < static_cast<int64_t>(H) &&
                                                iw >= 0 && iw < static_cast<int64_t>(W))
                                                val += input_ptr[((n * C + c) * H + ih) * W + iw]
                                                     * weight_ptr[((k * C + c) * kH + kh) * kW + kw];
                                        }
                                output_ptr[((n * K + k) * oH + oh) * oW + ow] = val;
                            }
                return output;
            }
        }

        template <typename T>
        Tensor<T> conv2d(const Tensor<T>& input, const Tensor<T>& weight,
                         int stride = 1, int padding = 0)
        {
            if (weight.shape().size() != 4)
                TENSOR_THROW("conv2d: weight must be 4D");
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
                TENSOR_THROW("conv_transpose2d: input and weight must be 4D");
            if (bias.shape().size() != 1)
                TENSOR_THROW("conv_transpose2d: bias must be 1D");

            size_t N = input.shape()[0], C = input.shape()[1];
            size_t H = input.shape()[2], W = input.shape()[3];
            size_t K = weight.shape()[0];
            size_t kH = weight.shape()[2], kW = weight.shape()[3];

            size_t oH = (H - 1) * stride + kH - 2 * padding;
            size_t oW = (W - 1) * stride + kW - 2 * padding;

            if (oH == 0 || oW == 0)
                TENSOR_THROW("conv_transpose2d: invalid output dimensions");

            Tensor<T> output({N, K, oH, oW});
            output.zero_();

            const T* __restrict input_ptr = input.data->data();
            const T* __restrict weight_ptr = weight.data->data();
            T* __restrict output_ptr = output.data->data();

            #pragma omp parallel for schedule(static)
            for (int64_t n = 0; n < static_cast<int64_t>(N); ++n)
                for (size_t c = 0; c < C; ++c)
                    for (size_t h = 0; h < H; ++h)
                        for (size_t w = 0; w < W; ++w) {
                            T in_val = input_ptr[((n * C + c) * H + h) * W + w];
                            for (size_t k = 0; k < K; ++k)
                                for (size_t kh = 0; kh < kH; ++kh)
                                    for (size_t kw = 0; kw < kW; ++kw) {
                                        int64_t oh = static_cast<int64_t>(h * stride + kh) - padding;
                                        int64_t ow = static_cast<int64_t>(w * stride + kw) - padding;
                                        if (oh >= 0 && oh < static_cast<int64_t>(oH) &&
                                            ow >= 0 && ow < static_cast<int64_t>(oW))
                                            output_ptr[((n * K + k) * oH + oh) * oW + ow]
                                                += in_val * weight_ptr[((k * C + c) * kH + kh) * kW + kw];
                                    }
                        }

            const T* __restrict bias_ptr = bias.data->data();
            #pragma omp parallel for schedule(static)
            for (int64_t n = 0; n < static_cast<int64_t>(N); ++n)
                for (size_t k = 0; k < K; ++k)
                    for (size_t oh = 0; oh < oH; ++oh)
                        for (size_t ow = 0; ow < oW; ++ow)
                            output_ptr[((n * K + k) * oH + oh) * oW + ow] += bias_ptr[k];

            return output;
        }

    } // namespace blas
} // namespace TensorN

#endif // __BLAS_TENSOR_HPP__
