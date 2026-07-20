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

    } // namespace cuda
} // namespace TensorN

#endif // __MATMUL_HPP__
