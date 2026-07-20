#include "matmul.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace TensorN
{
    namespace cuda
    {
        // ---- Hand-written kernels (fallback) ----

        template <typename T>
        __global__ void matmul_kernel(const T* A, const T* B, T* C,
                                     size_t M, size_t N, size_t K)
        {
            size_t row = blockIdx.y * blockDim.y + threadIdx.y;
            size_t col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < M && col < N)
            {
                T sum = 0;
                for (size_t k = 0; k < K; ++k)
                    sum += A[row * K + k] * B[k * N + col];
                C[row * N + col] = sum;
            }
        }

        template <typename T>
        __global__ void matmul_shared_kernel(const T* A, const T* B, T* C,
                                            size_t M, size_t N, size_t K)
        {
            const size_t BLOCK_SIZE = 16;
            __shared__ T As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ T Bs[BLOCK_SIZE][BLOCK_SIZE];

            size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
            size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

            T sum = 0;

            for (size_t t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t)
            {
                if (row < M && t * BLOCK_SIZE + threadIdx.x < K)
                    As[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
                else
                    As[threadIdx.y][threadIdx.x] = 0;

                if (t * BLOCK_SIZE + threadIdx.y < K && col < N)
                    Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
                else
                    Bs[threadIdx.y][threadIdx.x] = 0;

                __syncthreads();

                for (size_t k = 0; k < BLOCK_SIZE; ++k)
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

                __syncthreads();
            }

            if (row < M && col < N)
                C[row * N + col] = sum;
        }

        template <typename T>
        __global__ void batched_matmul_kernel(const T* A, const T* B, T* C,
                                              size_t M, size_t N, size_t K,
                                              size_t batch_size)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total = batch_size * M * N;
            if (idx >= total) return;

            size_t mn = idx % (M * N);
            size_t b = idx / (M * N);
            size_t row = mn / N;
            size_t col = mn % N;

            const T* A_batch = A + b * M * K;
            const T* B_batch = B + b * K * N;
            T* C_batch = C + b * M * N;

            T sum = 0;
            for (size_t k = 0; k < K; ++k)
                sum += A_batch[row * K + k] * B_batch[k * N + col];
            C_batch[mn] = sum;
        }

        // ---- cuBLAS helpers ----

        namespace
        {
            struct CublasHandle
            {
                cublasHandle_t handle;
                CublasHandle() { cublasCreate(&handle); }
                ~CublasHandle() { cublasDestroy(handle); }
            };

            inline cublasHandle_t get_handle()
            {
                static CublasHandle h;
                return h.handle;
            }
        }

        template <typename T>
        void matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            if (A.shape().size() != 2 || B.shape().size() != 2 || C.shape().size() != 2)
                throw std::invalid_argument("matmul_cublas requires 2D tensors");

            size_t M = A.shape()[0];
            size_t K = A.shape()[1];
            size_t N = B.shape()[1];

            if (B.shape()[0] != K)
                throw std::invalid_argument("Inner dimensions must match");
            if (C.shape()[0] != M || C.shape()[1] != N)
                throw std::invalid_argument("Output tensor has wrong shape");

            T alpha = T(1), beta = T(0);
            cublasHandle_t handle = get_handle();
            cublasStatus_t stat;

            // cuBLAS uses column-major, so we compute C^T = B^T * A^T
            // i.e., we swap A<->B and treat M<->N
            if constexpr (std::is_same_v<T, float>)
                stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N),
                    A.device_ptr(), static_cast<int>(K),
                    &beta, C.device_ptr(), static_cast<int>(N));
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N),
                    A.device_ptr(), static_cast<int>(K),
                    &beta, C.device_ptr(), static_cast<int>(N));
            else
                throw std::runtime_error("cuBLAS matmul only supports float/double");

            if (stat != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS gemm failed");
        }

        template <typename T>
        void batched_matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            if (A.shape().size() != 3 || B.shape().size() != 3 || C.shape().size() != 3)
                throw std::invalid_argument("batched_matmul_cublas requires 3D tensors");

            size_t batch_size = A.shape()[0];
            size_t M = A.shape()[1];
            size_t K = A.shape()[2];
            size_t N = B.shape()[2];

            if (B.shape()[0] != batch_size || B.shape()[1] != K)
                throw std::invalid_argument("Inner dimensions must match");
            if (C.shape()[0] != batch_size || C.shape()[1] != M || C.shape()[2] != N)
                throw std::invalid_argument("Output tensor has wrong shape");

            T alpha = T(1), beta = T(0);
            cublasHandle_t handle = get_handle();
            cublasStatus_t stat;

            long long strideA = static_cast<long long>(M * K);
            long long strideB = static_cast<long long>(K * N);
            long long strideC = static_cast<long long>(M * N);

            if constexpr (std::is_same_v<T, float>)
                stat = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N), strideB,
                    A.device_ptr(), static_cast<int>(K), strideA,
                    &beta, C.device_ptr(), static_cast<int>(N), strideC,
                    static_cast<int>(batch_size));
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N), strideB,
                    A.device_ptr(), static_cast<int>(K), strideA,
                    &beta, C.device_ptr(), static_cast<int>(N), strideC,
                    static_cast<int>(batch_size));
            else
                throw std::runtime_error("cuBLAS batched matmul only supports float/double");

            if (stat != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS strided batched gemm failed");
        }

        template <typename T>
        T dot(const CudaTensor<T>& A, const CudaTensor<T>& B)
        {
            if (A.shape().size() != 1 || B.shape().size() != 1)
                throw std::invalid_argument("dot requires 1D tensors");
            if (A.shape()[0] != B.shape()[0])
                throw std::invalid_argument("Dimension mismatch for dot product");

            size_t n = A.shape()[0];
            T result = T(0);
            cublasHandle_t handle = get_handle();
            cublasStatus_t stat;

            if constexpr (std::is_same_v<T, float>)
                stat = cublasSdot(handle, static_cast<int>(n),
                    A.device_ptr(), 1, B.device_ptr(), 1, &result);
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDdot(handle, static_cast<int>(n),
                    A.device_ptr(), 1, B.device_ptr(), 1, &result);
            else
                throw std::runtime_error("cuBLAS dot only supports float/double");

            if (stat != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS dot failed");

            return result;
        }

        // ---- Main matmul (uses cuBLAS) ----

        template <typename T>
        void matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            matmul_cublas(A, B, C);
        }

        template <typename T>
        void batched_matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            batched_matmul_cublas(A, B, C);
        }

        template void matmul<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void matmul<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void batched_matmul<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void batched_matmul<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void matmul_cublas<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void matmul_cublas<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void batched_matmul_cublas<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void batched_matmul_cublas<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template float dot<float>(const CudaTensor<float>&, const CudaTensor<float>&);
        template double dot<double>(const CudaTensor<double>&, const CudaTensor<double>&);

    } // namespace cuda
} // namespace TensorN
