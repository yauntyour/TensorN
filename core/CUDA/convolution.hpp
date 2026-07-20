#pragma once
#ifndef __CONVOLUTION_HPP__
#define __CONVOLUTION_HPP__

#include "cuda_tensor.hpp"

namespace TensorN
{
    namespace cuda
    {
        // 2D Convolution
        // Input: (batch, in_channels, height, width)
        // Weight: (out_channels, in_channels, kernel_h, kernel_w)
        // Bias: (out_channels)
        // Output: (batch, out_channels, out_height, out_width)
        template <typename T>
        void conv2d(const CudaTensor<T>& input, 
                   const CudaTensor<T>& weight,
                   const CudaTensor<T>& bias,
                   CudaTensor<T>& output,
                   int stride = 1,
                   int padding = 0);

        // 2D Convolution without bias
        template <typename T>
        void conv2d(const CudaTensor<T>& input,
                   const CudaTensor<T>& weight,
                   CudaTensor<T>& output,
                   int stride = 1,
                   int padding = 0);

        // 2D Transposed Convolution (Deconvolution)
        template <typename T>
        void conv_transpose2d(const CudaTensor<T>& input,
                             const CudaTensor<T>& weight,
                             const CudaTensor<T>& bias,
                             CudaTensor<T>& output,
                             int stride = 1,
                             int padding = 0);

    } // namespace cuda
} // namespace TensorN

#endif // __CONVOLUTION_HPP__