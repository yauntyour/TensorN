#pragma once
#ifndef __CUBLAS_EX_HPP__
#define __CUBLAS_EX_HPP__

// ============================================================================
// Internal helpers for cuBLAS low-precision GEMM dispatch (GemmEx).
// Included only from .cu translation units.
//
// The low-precision types (half / bfloat16 / tf32 / fp8_e4m3 / fp8_e5m2) are
// bit-layout-compatible with the CUDA types __half / __nv_bfloat16 /
// __nv_fp8_e4m3 / __nv_fp8_e5m2, so their device pointers can be passed to
// cuBLAS directly.
// ============================================================================

#include "../dtypes.hpp"
#include "../tensor.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#include <string>
#include <type_traits>

namespace TensorN
{
    namespace cuda
    {
        namespace detail
        {
            // Runtime TF32 toggle for float GEMMs (thread-safe singleton)
            inline bool& tf32_flag_impl()
            {
                static bool enabled = false;
                return enabled;
            }

            // Types that are dispatched through cublasGemmEx
            template <typename T>
            struct is_gemmex_type : std::bool_constant<
                std::is_same_v<T, TensorN::half> ||
                std::is_same_v<T, TensorN::bfloat16> ||
                std::is_same_v<T, TensorN::tf32>
#if CUDART_VERSION >= 12000
                || std::is_same_v<T, TensorN::fp8_e4m3> ||
                std::is_same_v<T, TensorN::fp8_e5m2>
#endif
                > {};

            template <typename T>
            cudaDataType_t cublas_data_type()
            {
                if constexpr (std::is_same_v<T, float>)              return CUDA_R_32F;
                else if constexpr (std::is_same_v<T, double>)        return CUDA_R_64F;
                else if constexpr (std::is_same_v<T, TensorN::half>) return CUDA_R_16F;
#if CUDART_VERSION >= 11000
                else if constexpr (std::is_same_v<T, TensorN::bfloat16>) return CUDA_R_16BF;
#endif
                else if constexpr (std::is_same_v<T, TensorN::tf32>) return CUDA_R_32F;
#if CUDART_VERSION >= 12000
                else if constexpr (std::is_same_v<T, TensorN::fp8_e4m3>) return CUDA_R_8F_E4M3;
                else if constexpr (std::is_same_v<T, TensorN::fp8_e5m2>) return CUDA_R_8F_E5M2;
#endif
                else return CUDA_R_32F;
            }

            // Compute type: FP32 accumulation for all low-precision inputs;
            // TF32 inputs use the TF32 tensor-core math mode.
            template <typename T>
            cublasComputeType_t cublas_compute_type()
            {
                if constexpr (std::is_same_v<T, TensorN::tf32>)
                    return CUBLAS_COMPUTE_32F_FAST_TF32;
                else
                    return CUBLAS_COMPUTE_32F;
            }

            template <typename T>
            cublasGemmAlgo_t cublas_gemm_algo()
            {
                if constexpr (std::is_same_v<T, TensorN::fp8_e4m3> ||
                              std::is_same_v<T, TensorN::fp8_e5m2>)
                    return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
                else
                    return CUBLAS_GEMM_DEFAULT;
            }

            // FP8 tensor-core GEMM requires the leading dimensions to be
            // multiples of 16 elements (16 bytes for 1-byte FP8).
            template <typename T>
            void check_gemmex_leading_dims(size_t M, size_t N, size_t K)
            {
                if constexpr (std::is_same_v<T, TensorN::fp8_e4m3> ||
                              std::is_same_v<T, TensorN::fp8_e5m2>)
                {
                    if (N % 16 != 0 || K % 16 != 0 || M % 16 != 0)
                    {
                        TENSOR_THROW(
                            "FP8 GEMM requires matrix dimensions (M, N, K) to be "
                            "multiples of 16. Got M=" + std::to_string(M) +
                            ", N=" + std::to_string(N) + ", K=" + std::to_string(K) +
                            ". Pad the matrices to the nearest multiple of 16.");
                    }
                }
            }

            // Row-major C = A * B emulated as column-major
            // C(N x M) = B(N x K) * A(K x M).
            template <typename T>
            void gemm_ex(cublasHandle_t handle, size_t M, size_t N, size_t K,
                         const T* A, const T* B, T* C)
            {
                check_gemmex_leading_dims<T>(M, N, K);

                const float alpha = 1.0f, beta = 0.0f;
                const cudaDataType_t type = cublas_data_type<T>();
                const cublasComputeType_t compute = cublas_compute_type<T>();
                const cublasGemmAlgo_t algo = cublas_gemm_algo<T>();

                cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha,
                    B, type, static_cast<int>(N),
                    A, type, static_cast<int>(K),
                    &beta,
                    C, type, static_cast<int>(N),
                    compute, algo);

                if (stat != CUBLAS_STATUS_SUCCESS)
                {
                    TENSOR_THROW(
                        "cuBLAS GemmEx failed (status " +
                        std::to_string(static_cast<int>(stat)) +
                        "). Low-precision GEMM requires CUDA 12+ (FP8 needs "
                        "compute capability >= 8.9); half/bf16/tf32 tensor cores "
                        "need compute capability >= 8.0.");
                }
            }

            // Row-major C = A^T * B emulated as column-major
            // C(M x M) = X(M x N) * X(M x N)^T.
            template <typename T>
            void gemm_ex_trans(cublasHandle_t handle, size_t M, size_t N,
                               const T* A, T* C)
            {
                check_gemmex_leading_dims<T>(M, M, N);

                const float alpha = 1.0f, beta = 0.0f;
                const cudaDataType_t type = cublas_data_type<T>();
                const cublasComputeType_t compute = cublas_compute_type<T>();
                const cublasGemmAlgo_t algo = cublas_gemm_algo<T>();

                cublasStatus_t stat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(M), static_cast<int>(M), static_cast<int>(N),
                    &alpha,
                    A, type, static_cast<int>(N),
                    A, type, static_cast<int>(N),
                    &beta,
                    C, type, static_cast<int>(M),
                    compute, algo);

                if (stat != CUBLAS_STATUS_SUCCESS)
                {
                    TENSOR_THROW(
                        "cuBLAS GemmEx (transposed) failed (status " +
                        std::to_string(static_cast<int>(stat)) +
                        "). Low-precision GEMM requires CUDA 12+ (FP8 needs "
                        "compute capability >= 8.9); half/bf16/tf32 tensor cores "
                        "need compute capability >= 8.0.");
                }
            }

            // cublasDotEx with FP32 result accumulation, cast back to T
            template <typename T>
            T dot_ex(cublasHandle_t handle, size_t n, const T* x, const T* y)
            {
                const cudaDataType_t type = cublas_data_type<T>();
                float result = 0.0f;
                cublasStatus_t stat = cublasDotEx(handle, static_cast<int>(n),
                    x, type, 1, y, type, 1, &result, CUDA_R_32F, CUDA_R_32F);
                if (stat != CUBLAS_STATUS_SUCCESS)
                {
                    TENSOR_THROW(
                        "cuBLAS DotEx failed (status " +
                        std::to_string(static_cast<int>(stat)) + ").");
                }
                return T(result);
            }
        } // namespace detail
    } // namespace cuda
} // namespace TensorN

#endif // __CUBLAS_EX_HPP__
