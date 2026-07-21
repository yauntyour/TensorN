#include "matmul.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>

namespace TensorN
{
    namespace cuda
    {
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
        void matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C, cudaStream_t stream)
        {
            if (A.shape().size() != 2 || B.shape().size() != 2 || C.shape().size() != 2)
                TENSOR_THROW("matmul_cublas requires 2D tensors");

            size_t M = A.shape()[0];
            size_t K = A.shape()[1];
            size_t N = B.shape()[1];

            if (B.shape()[0] != K)
                TENSOR_THROW("Inner dimensions must match");
            if (C.shape()[0] != M || C.shape()[1] != N)
                TENSOR_THROW("Output tensor has wrong shape");

            T alpha = T(1), beta = T(0);

            auto& blas_handle = get_stream_blas_handle();
            blas_handle.set_stream(stream);

            cublasStatus_t stat;
            if constexpr (std::is_same_v<T, float>)
                stat = cublasSgemm(blas_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N),
                    A.device_ptr(), static_cast<int>(K),
                    &beta, C.device_ptr(), static_cast<int>(N));
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDgemm(blas_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N),
                    A.device_ptr(), static_cast<int>(K),
                    &beta, C.device_ptr(), static_cast<int>(N));
            else
                TENSOR_THROW("cuBLAS matmul only supports float/double");

            if (stat != CUBLAS_STATUS_SUCCESS)
                TENSOR_THROW("cuBLAS gemm failed");
        }

        template <typename T>
        void matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            matmul_cublas(A, B, C, nullptr);
        }

        template <typename T>
        void batched_matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C, cudaStream_t stream)
        {
            if (A.shape().size() != 3 || B.shape().size() != 3 || C.shape().size() != 3)
                TENSOR_THROW("batched_matmul_cublas requires 3D tensors");

            size_t batch_size = A.shape()[0];
            size_t M = A.shape()[1];
            size_t K = A.shape()[2];
            size_t N = B.shape()[2];

            if (B.shape()[0] != batch_size || B.shape()[1] != K)
                TENSOR_THROW("Inner dimensions must match");
            if (C.shape()[0] != batch_size || C.shape()[1] != M || C.shape()[2] != N)
                TENSOR_THROW("Output tensor has wrong shape");

            T alpha = T(1), beta = T(0);

            auto& blas_handle = get_stream_blas_handle();
            blas_handle.set_stream(stream);

            long long strideA = static_cast<long long>(M * K);
            long long strideB = static_cast<long long>(K * N);
            long long strideC = static_cast<long long>(M * N);

            cublasStatus_t stat;
            if constexpr (std::is_same_v<T, float>)
                stat = cublasSgemmStridedBatched(blas_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N), strideB,
                    A.device_ptr(), static_cast<int>(K), strideA,
                    &beta, C.device_ptr(), static_cast<int>(N), strideC,
                    static_cast<int>(batch_size));
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDgemmStridedBatched(blas_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
                    &alpha, B.device_ptr(), static_cast<int>(N), strideB,
                    A.device_ptr(), static_cast<int>(K), strideA,
                    &beta, C.device_ptr(), static_cast<int>(N), strideC,
                    static_cast<int>(batch_size));
            else
                TENSOR_THROW("cuBLAS batched matmul only supports float/double");

            if (stat != CUBLAS_STATUS_SUCCESS)
                TENSOR_THROW("cuBLAS strided batched gemm failed");
        }

        template <typename T>
        void batched_matmul_cublas(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            batched_matmul_cublas(A, B, C, nullptr);
        }

        template <typename T>
        T dot(const CudaTensor<T>& A, const CudaTensor<T>& B, cudaStream_t stream)
        {
            if (A.shape().size() != 1 || B.shape().size() != 1)
                TENSOR_THROW("dot requires 1D tensors");
            if (A.shape()[0] != B.shape()[0])
                TENSOR_THROW("Dimension mismatch for dot product");

            size_t n = A.shape()[0];
            T result = T(0);

            auto& blas_handle = get_stream_blas_handle();
            blas_handle.set_stream(stream);

            cublasStatus_t stat;
            if constexpr (std::is_same_v<T, float>)
                stat = cublasSdot(blas_handle.get(), static_cast<int>(n),
                    A.device_ptr(), 1, B.device_ptr(), 1, &result);
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDdot(blas_handle.get(), static_cast<int>(n),
                    A.device_ptr(), 1, B.device_ptr(), 1, &result);
            else
                TENSOR_THROW("cuBLAS dot only supports float/double");

            if (stat != CUBLAS_STATUS_SUCCESS)
                TENSOR_THROW("cuBLAS dot failed");

            return result;
        }

        template <typename T>
        T dot(const CudaTensor<T>& A, const CudaTensor<T>& B)
        {
            return dot(A, B, nullptr);
        }

        template <typename T>
        void matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            matmul_cublas(A, B, C, nullptr);
        }

        template <typename T>
        void matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C, cudaStream_t stream)
        {
            matmul_cublas(A, B, C, stream);
        }

        template <typename T>
        void batched_matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            batched_matmul_cublas(A, B, C, nullptr);
        }

        template <typename T>
        void batched_matmul(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C, cudaStream_t stream)
        {
            batched_matmul_cublas(A, B, C, stream);
        }

        template void matmul<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void matmul<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void matmul<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void matmul<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template void batched_matmul<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void batched_matmul<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void batched_matmul<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void batched_matmul<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template void matmul_cublas<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void matmul_cublas<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void matmul_cublas<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void matmul_cublas<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template void batched_matmul_cublas<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void batched_matmul_cublas<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void batched_matmul_cublas<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void batched_matmul_cublas<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template float dot<float>(const CudaTensor<float>&, const CudaTensor<float>&);
        template double dot<double>(const CudaTensor<double>&, const CudaTensor<double>&);
        template float dot<float>(const CudaTensor<float>&, const CudaTensor<float>&, cudaStream_t);
        template double dot<double>(const CudaTensor<double>&, const CudaTensor<double>&, cudaStream_t);

        template <typename T>
        __global__ void outer_kernel(const T* A, const T* B, T* C, size_t m, size_t n)
        {
            size_t row = blockIdx.y * blockDim.y + threadIdx.y;
            size_t col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row < m && col < n)
                C[row * n + col] = A[row] * B[col];
        }

        template <typename T>
        void outer(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C, cudaStream_t stream)
        {
            if (A.shape().size() != 1 || B.shape().size() != 1)
                TENSOR_THROW("outer requires 1D tensors");
            size_t m = A.shape()[0], n = B.shape()[0];
            if (C.shape()[0] != m || C.shape()[1] != n)
                TENSOR_THROW("Output shape mismatch for outer");
            dim3 block(16, 16);
            dim3 grid((n + 15) / 16, (m + 15) / 16);
            cudaMemsetAsync(C.device_ptr(), 0, m * n * sizeof(T), stream);
            outer_kernel<<<grid, block, 0, stream>>>(A.device_ptr(), B.device_ptr(), C.device_ptr(), m, n);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        template <typename T>
        void outer(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C)
        {
            outer(A, B, C, nullptr);
        }

        template <typename T>
        void gram(const CudaTensor<T>& X, CudaTensor<T>& C, cudaStream_t stream)
        {
            if (X.shape().size() != 2)
                TENSOR_THROW("gram requires 2D tensor");
            size_t M = X.shape()[0], N = X.shape()[1];
            if (C.shape()[0] != M || C.shape()[1] != M)
                TENSOR_THROW("Output shape mismatch for gram");

            T alpha = T(1), beta = T(0);
            auto& blas_handle = get_stream_blas_handle();
            blas_handle.set_stream(stream);

            cublasStatus_t stat;
            if constexpr (std::is_same_v<T, float>)
                stat = cublasSgemm(blas_handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(M), static_cast<int>(M), static_cast<int>(N),
                    &alpha, X.device_ptr(), static_cast<int>(N),
                    X.device_ptr(), static_cast<int>(N),
                    &beta, C.device_ptr(), static_cast<int>(M));
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDgemm(blas_handle.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                    static_cast<int>(M), static_cast<int>(M), static_cast<int>(N),
                    &alpha, X.device_ptr(), static_cast<int>(N),
                    X.device_ptr(), static_cast<int>(N),
                    &beta, C.device_ptr(), static_cast<int>(M));
            if (stat != CUBLAS_STATUS_SUCCESS)
                TENSOR_THROW("cuBLAS gram failed");
        }

        template <typename T>
        void gram(const CudaTensor<T>& X, CudaTensor<T>& C)
        {
            gram(X, C, nullptr);
        }

        template <typename T>
        T bilinear(const CudaTensor<T>& x, const CudaTensor<T>& A, const CudaTensor<T>& y)
        {
            if (x.shape().size() != 1 || A.shape().size() != 2 || y.shape().size() != 1)
                TENSOR_THROW("bilinear: wrong dimensions");
            size_t M = A.shape()[0], N = A.shape()[1];

            T alpha = T(1), beta = T(0);
            auto& blas_handle = get_stream_blas_handle();
            blas_handle.set_stream(nullptr);

            cublasStatus_t stat;
            CudaTensor<T> temp({M});
            if constexpr (std::is_same_v<T, float>)
                stat = cublasSgemv(blas_handle.get(), CUBLAS_OP_N,
                    static_cast<int>(M), static_cast<int>(N),
                    &alpha, A.device_ptr(), static_cast<int>(N),
                    y.device_ptr(), 1, &beta, temp.device_ptr(), 1);
            else if constexpr (std::is_same_v<T, double>)
                stat = cublasDgemv(blas_handle.get(), CUBLAS_OP_N,
                    static_cast<int>(M), static_cast<int>(N),
                    &alpha, A.device_ptr(), static_cast<int>(N),
                    y.device_ptr(), 1, &beta, temp.device_ptr(), 1);
            if (stat != CUBLAS_STATUS_SUCCESS)
                TENSOR_THROW("cuBLAS gemv failed");

            return dot(x, temp);
        }

        template <typename T>
        __global__ void axpy_kernel(const T* x, T* y, T alpha, size_t n)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) y[idx] = alpha * x[idx] + y[idx];
        }

        template <typename T>
        void axpy(T alpha, const CudaTensor<T>& x, CudaTensor<T>& y, cudaStream_t stream)
        {
            if (x.size() != y.size())
                TENSOR_THROW("axpy: size mismatch");
            size_t n = x.size(), bs = 256, gs = (n + bs - 1) / bs;
            axpy_kernel<<<gs, bs, 0, stream>>>(x.device_ptr(), y.device_ptr(), alpha, n);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        template <typename T>
        void axpy(T alpha, const CudaTensor<T>& x, CudaTensor<T>& y)
        {
            axpy(alpha, x, y, nullptr);
        }

        template <typename T>
        __global__ void trace_sum_kernel(const T* A, T* partial, size_t n, size_t stride)
        {
            extern __shared__ char smem[];
            T* sdata = reinterpret_cast<T*>(smem);
            size_t tid = threadIdx.x;
            size_t idx = blockIdx.x * blockDim.x + tid;
            T val = T(0);
            if (idx < n) val = A[idx * (stride + 1)];
            sdata[tid] = val;
            __syncthreads();
            for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            if (tid == 0) partial[blockIdx.x] = sdata[0];
        }

        template <typename T>
        T trace(const CudaTensor<T>& A)
        {
            if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
                TENSOR_THROW("trace requires a square matrix");
            size_t n = A.shape()[0];
            if (n == 0) return T(0);
            size_t bs = 256, gs = (n + bs - 1) / bs;
            T* d_partial;
            cudaMalloc(reinterpret_cast<void**>(&d_partial), gs * sizeof(T));
            trace_sum_kernel<<<gs, bs, bs * sizeof(T)>>>(A.device_ptr(), d_partial, n, A.shape()[1]);
            CHECK_CUDA_ERROR(cudaGetLastError());
            std::vector<T> h(gs);
            cudaMemcpy(h.data(), d_partial, gs * sizeof(T), cudaMemcpyDeviceToHost);
            cudaFree(d_partial);
            T result = T(0);
            for (size_t i = 0; i < gs; ++i) result += h[i];
            return result;
        }

        template <typename T>
        __global__ void diag_kernel(const T* A, T* C, size_t n, size_t stride)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) C[idx] = A[idx * (stride + 1)];
        }

        template <typename T>
        __global__ void diag_matrix_kernel(const T* v, T* C, size_t n, size_t stride)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n * n) {
                size_t r = idx / n, c = idx % n;
                C[idx] = (r == c) ? v[r] : T(0);
            }
        }

        template <typename T>
        void diag(const CudaTensor<T>& A, CudaTensor<T>& C, cudaStream_t stream)
        {
            if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
                TENSOR_THROW("diag requires a square matrix");
            size_t n = A.shape()[0];
            size_t bs = 256, gs = (n + bs - 1) / bs;
            diag_kernel<<<gs, bs, 0, stream>>>(A.device_ptr(), C.device_ptr(), n, A.shape()[1]);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        template <typename T>
        void diag(const CudaTensor<T>& A, CudaTensor<T>& C)
        {
            diag(A, C, nullptr);
        }

        template <typename T>
        void diag_matrix(const CudaTensor<T>& v, CudaTensor<T>& C, cudaStream_t stream)
        {
            if (v.shape().size() != 1)
                TENSOR_THROW("diag_matrix requires a 1D tensor");
            size_t n = v.shape()[0];
            size_t total = n * n, bs = 256, gs = (total + bs - 1) / bs;
            diag_matrix_kernel<<<gs, bs, 0, stream>>>(v.device_ptr(), C.device_ptr(), n, n);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        template <typename T>
        void diag_matrix(const CudaTensor<T>& v, CudaTensor<T>& C)
        {
            diag_matrix(v, C, nullptr);
        }

        template void outer<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&);
        template void outer<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&);
        template void outer<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void outer<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template void gram<float>(const CudaTensor<float>&, CudaTensor<float>&);
        template void gram<double>(const CudaTensor<double>&, CudaTensor<double>&);
        template void gram<float>(const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void gram<double>(const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template float bilinear<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&);
        template double bilinear<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&);
        template void axpy<float>(float, const CudaTensor<float>&, CudaTensor<float>&);
        template void axpy<double>(double, const CudaTensor<double>&, CudaTensor<double>&);
        template void axpy<float>(float, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void axpy<double>(double, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template float trace<float>(const CudaTensor<float>&);
        template double trace<double>(const CudaTensor<double>&);
        template void diag<float>(const CudaTensor<float>&, CudaTensor<float>&);
        template void diag<double>(const CudaTensor<double>&, CudaTensor<double>&);
        template void diag<float>(const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void diag<double>(const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
        template void diag_matrix<float>(const CudaTensor<float>&, CudaTensor<float>&);
        template void diag_matrix<double>(const CudaTensor<double>&, CudaTensor<double>&);
        template void diag_matrix<float>(const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
        template void diag_matrix<double>(const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);

    } // namespace cuda
} // namespace TensorN
