#pragma once
#ifndef __CONVOLUTION_HPP__
#define __CONVOLUTION_HPP__

#include "cuda_tensor.hpp"
#include "cuda_stream.hpp"

namespace TensorN
{
    namespace cuda
    {
        template <typename T>
        void conv2d(const CudaTensor<T>& input, 
                   const CudaTensor<T>& weight,
                   const CudaTensor<T>& bias,
                   CudaTensor<T>& output,
                   int stride = 1,
                   int padding = 0);

        template <typename T>
        void conv2d(const CudaTensor<T>& input, 
                   const CudaTensor<T>& weight,
                   const CudaTensor<T>& bias,
                   CudaTensor<T>& output,
                   int stride,
                   int padding,
                   cudaStream_t stream);

        template <typename T>
        void conv2d(const CudaTensor<T>& input,
                   const CudaTensor<T>& weight,
                   CudaTensor<T>& output,
                   int stride = 1,
                   int padding = 0);

        template <typename T>
        void conv_transpose2d(const CudaTensor<T>& input,
                             const CudaTensor<T>& weight,
                             const CudaTensor<T>& bias,
                             CudaTensor<T>& output,
                             int stride = 1,
                             int padding = 0);

        template <typename T>
        void conv_transpose2d(const CudaTensor<T>& input,
                             const CudaTensor<T>& weight,
                             const CudaTensor<T>& bias,
                             CudaTensor<T>& output,
                             int stride,
                             int padding,
                             cudaStream_t stream);

    } // namespace cuda
} // namespace TensorN

#endif // __CONVOLUTION_HPP__