#pragma once
#ifndef __REDUCTION_HPP__
#define __REDUCTION_HPP__

#include "cuda_tensor.hpp"

namespace TensorN
{
    namespace cuda
    {
        // Sum all elements
        template <typename T>
        T sum(const CudaTensor<T>& A);

        // Sum along axis
        template <typename T>
        CudaTensor<T> sum_axis(const CudaTensor<T>& A, int axis);

        // Mean of all elements
        template <typename T>
        T mean(const CudaTensor<T>& A);

        // Mean along axis
        template <typename T>
        CudaTensor<T> mean_axis(const CudaTensor<T>& A, int axis);

        // Max of all elements
        template <typename T>
        T max(const CudaTensor<T>& A);

        // Max along axis
        template <typename T>
        CudaTensor<T> max_axis(const CudaTensor<T>& A, int axis);

        // Min of all elements
        template <typename T>
        T min(const CudaTensor<T>& A);

        // Min along axis
        template <typename T>
        CudaTensor<T> min_axis(const CudaTensor<T>& A, int axis);

        // Argmax along axis
        template <typename T>
        CudaTensor<int64_t> argmax(const CudaTensor<T>& A, int axis);

        // Argmin along axis
        template <typename T>
        CudaTensor<int64_t> argmin(const CudaTensor<T>& A, int axis);

        // Vector L2 norm
        template <typename T>
        T norm(const CudaTensor<T>& A);

        // Frobenius norm
        template <typename T>
        T frobenius_norm(const CudaTensor<T>& A);

        // Variance
        template <typename T>
        T var(const CudaTensor<T>& A);

        // Standard deviation
        template <typename T>
        T stddev(const CudaTensor<T>& A);

    } // namespace cuda
} // namespace TensorN

#endif // __REDUCTION_HPP__