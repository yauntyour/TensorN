#pragma once
#ifndef __FUSED_KERNELS_HPP__
#define __FUSED_KERNELS_HPP__

#include "cuda_tensor.hpp"
#include "cuda_stream.hpp"

namespace TensorN
{
    namespace cuda
    {
        enum class ActivationType
        {
            None,
            ReLU,
            Sigmoid,
            Tanh,
            GELU,
            LeakyReLU,
            ELU
        };

        template <typename T>
        void matmul_activation(const CudaTensor<T>& A, const CudaTensor<T>& B,
                               CudaTensor<T>& C, ActivationType act,
                               T alpha_param = T(0),
                               cudaStream_t stream = nullptr);

        template <typename T>
        void conv2d_activation(const CudaTensor<T>& input, const CudaTensor<T>& weight,
                               const CudaTensor<T>& bias, CudaTensor<T>& output,
                               int stride, int padding, ActivationType act,
                               T alpha_param = T(0),
                               cudaStream_t stream = nullptr);

        template <typename T>
        void add_relu(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C,
                      cudaStream_t stream = nullptr);

        template <typename T>
        void mul_add(const CudaTensor<T>& A, const CudaTensor<T>& B, const CudaTensor<T>& C,
                     CudaTensor<T>& D, cudaStream_t stream = nullptr);

        template <typename T>
        void batchnorm_inference(const CudaTensor<T>& input, const CudaTensor<T>& mean,
                                 const CudaTensor<T>& var, const CudaTensor<T>& gamma,
                                 const CudaTensor<T>& beta, CudaTensor<T>& output,
                                 T eps, size_t channels,
                                 cudaStream_t stream = nullptr);

        template <typename T>
        void residual_block(const CudaTensor<T>& input, const CudaTensor<T>& weight1,
                           const CudaTensor<T>& bias1, const CudaTensor<T>& weight2,
                           const CudaTensor<T>& bias2, CudaTensor<T>& output,
                           int stride, int padding,
                           cudaStream_t stream = nullptr);

    } // namespace cuda
} // namespace TensorN

#endif // __FUSED_KERNELS_HPP__
