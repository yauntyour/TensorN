#pragma once
#ifndef __ELEMENTWISE_HPP__
#define __ELEMENTWISE_HPP__

#include "cuda_tensor.hpp"

namespace TensorN
{
    namespace cuda
    {
        template <typename T>
        void add(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        template <typename T>
        void subtract(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        template <typename T>
        void multiply(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        template <typename T>
        void divide(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C);

        template <typename T>
        void add_scalar(const CudaTensor<T>& A, T scalar, CudaTensor<T>& C);

        template <typename T>
        void multiply_scalar(const CudaTensor<T>& A, T scalar, CudaTensor<T>& C);

        template <typename T>
        void subtract_scalar(const CudaTensor<T>& A, T scalar, CudaTensor<T>& C);

        template <typename T>
        void divide_scalar(const CudaTensor<T>& A, T scalar, CudaTensor<T>& C);

        // Unary element-wise
        template <typename T>
        void negate(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void abs(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void sqrt(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void exp(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void log(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void sin(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void cos(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void pow(const CudaTensor<T>& A, T exponent, CudaTensor<T>& C);

        // Clipping
        template <typename T>
        void clip(const CudaTensor<T>& A, T min_val, T max_val, CudaTensor<T>& C);

        // Activation functions
        template <typename T>
        void relu(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void leaky_relu(const CudaTensor<T>& A, T alpha, CudaTensor<T>& C);

        template <typename T>
        void elu(const CudaTensor<T>& A, T alpha, CudaTensor<T>& C);

        template <typename T>
        void gelu(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void sigmoid(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void tanh(const CudaTensor<T>& A, CudaTensor<T>& C);

        template <typename T>
        void softmax(const CudaTensor<T>& A, CudaTensor<T>& C, int axis = -1);

        // Comparison operations (returns int tensor: 1 if true, 0 if false)
        template <typename T>
        void equal(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C);

        template <typename T>
        void not_equal(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C);

        template <typename T>
        void greater(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C);

        template <typename T>
        void less(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C);

        template <typename T>
        void greater_equal(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C);

        template <typename T>
        void less_equal(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C);

    } // namespace cuda
} // namespace TensorN

#endif // __ELEMENTWISE_HPP__
