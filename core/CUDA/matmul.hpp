#pragma once
#ifndef __MATMUL_HPP__
#define __MATMUL_HPP__

#include "cuda_tensor.hpp"

namespace TensorN
{
    namespace cuda
    {
        template <typename T>
        void matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        template <typename T>
        void batched_matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        // cuBLAS-accelerated versions (fall back to hand-written if cuBLAS unavailable)
        template <typename T>
        void matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        template <typename T>
        void batched_matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        // Dot product
        template <typename T>
        T dot(const CudaTensor<T>& A, const CudaTensor<T>& B);

        // Outer product
        template <typename T>
        void outer(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        // Gram matrix: X * X^T
        template <typename T>
        void gram(const CudaTensor<T>& X, CudaTensor<T>& C);

        // Bilinear form: returns scalar x^T A y
        template <typename T>
        T bilinear(const CudaTensor<T>& x, const CudaTensor<T>& A, const CudaTensor<T>& y);

        // AXPY: y = alpha * x + y (in-place on y)
        template <typename T>
        void axpy(T alpha, const CudaTensor<T>& x, CudaTensor<T>& y);

        // Matrix trace
        template <typename T>
        T trace(const CudaTensor<T>& A);

        // Diagonal extraction
        template <typename T>
        void diag(const CudaTensor<T>& A, CudaTensor<T>& C);

        // Diagonal matrix creation
        template <typename T>
        void diag_matrix(const CudaTensor<T>& v, CudaTensor<T>& C);

    } // namespace cuda
} // namespace TensorN

#endif // __MATMUL_HPP__
