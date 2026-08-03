#include "convolution.hpp"
#include "cublas_ex.hpp"
#include "cuda_stream.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <type_traits>

namespace TensorN
{
    namespace cuda
    {
        inline size_t get_optimal_block_size(size_t n) {
            if (n <= 0) return 256;
            if (n < 1024) return std::min(n, size_t(256));
            if (n < 1024 * 1024) return 512;
            return 1024;
        }

        inline size_t get_grid_size(size_t n, size_t block_size) {
            if (n == 0 || block_size == 0) return 0;
            size_t grid_size = (n + block_size - 1) / block_size;
            const size_t MAX_GRID_SIZE = 2147483647;
            return std::min(grid_size, MAX_GRID_SIZE);
        }

        // GPU im2col kernel: transforms input patches into column matrix
        template <typename T>
        __global__ void im2col_kernel(const T* input,
            size_t C, size_t H, size_t W,
            size_t kH, size_t kW,
            int stride, int padding,
            size_t oH, size_t oW,
            T* col)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total = oH * oW;
            if (idx >= total) return;

            size_t oh = idx / oW;
            size_t ow = idx % oW;

            size_t col_row_size = C * kH * kW;
            T* col_row = col + idx * col_row_size;

            size_t col_idx = 0;
            for (size_t c = 0; c < C; ++c) {
                for (size_t kh = 0; kh < kH; ++kh) {
                    for (size_t kw = 0; kw < kW; ++kw) {
                        int ih = static_cast<int>(oh * stride + kh) - padding;
                        int iw = static_cast<int>(ow * stride + kw) - padding;
                        if (ih >= 0 && static_cast<size_t>(ih) < H &&
                            iw >= 0 && static_cast<size_t>(iw) < W)
                            col_row[col_idx] = input[(c * H + ih) * W + iw];
                        else
                            col_row[col_idx] = T(0);
                        ++col_idx;
                    }
                }
            }
        }

        // Bias add kernel
        template <typename T>
        __global__ void bias_add_kernel(T* output, const T* bias,
            size_t batch, size_t out_channels,
            size_t out_height, size_t out_width) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total = batch * out_channels * out_height * out_width;
            if (idx < total) {
                size_t oc = (idx / (out_width * out_height)) % out_channels;
                output[idx] += bias[oc];
            }
        }

        template <typename T>
        void conv2d(const CudaTensor<T>& input,
                   const CudaTensor<T>& weight,
                   const CudaTensor<T>& bias,
                   CudaTensor<T>& output,
                   int stride,
                   int padding,
                   cudaStream_t stream)
        {
            if (input.shape().size() != 4 || weight.shape().size() != 4 || output.shape().size() != 4)
                TENSOR_THROW("conv2d requires 4D tensors");

            size_t batch = input.shape()[0];
            size_t in_channels = input.shape()[1];
            size_t height = input.shape()[2];
            size_t width = input.shape()[3];

            size_t out_channels = weight.shape()[0];
            size_t kernel_h = weight.shape()[2];
            size_t kernel_w = weight.shape()[3];

            size_t out_height = (height + 2 * padding - kernel_h) / stride + 1;
            size_t out_width = (width + 2 * padding - kernel_w) / stride + 1;

            if (output.shape()[0] != batch || output.shape()[1] != out_channels ||
                output.shape()[2] != out_height || output.shape()[3] != out_width)
                TENSOR_THROW("Output tensor has wrong shape");

            size_t col_size = in_channels * kernel_h * kernel_w * out_height * out_width;
            auto& pool = CudaMemoryPool::instance();

            int M = static_cast<int>(out_channels);
            int Nn = static_cast<int>(out_height * out_width);
            int Kk = static_cast<int>(in_channels * kernel_h * kernel_w);

            auto& blas_handle = get_stream_blas_handle();
            blas_handle.set_stream(stream);

            const T* weight_ptr = weight.device_ptr();

            for (size_t n = 0; n < batch; ++n)
            {
                const T* input_batch = input.device_ptr() + n * in_channels * height * width;
                T* output_batch = output.device_ptr() + n * out_channels * out_height * out_width;

                T* d_col = static_cast<T*>(pool.acquire(col_size * sizeof(T)));

                size_t col_grid_size = get_grid_size(out_height * out_width,
                    get_optimal_block_size(out_height * out_width));
                im2col_kernel<<<col_grid_size, get_optimal_block_size(out_height * out_width), 0, stream>>>(
                    input_batch, in_channels, height, width, kernel_h, kernel_w,
                    stride, padding, out_height, out_width, d_col);
                CHECK_CUDA_ERROR(cudaGetLastError());

                cublasStatus_t stat;
                if constexpr (std::is_same_v<T, float>)
                {
                    float alpha = 1.0f, beta = 0.0f;
                    stat = cublasSgemm(blas_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                        Nn, M, Kk, &alpha, d_col, Nn,
                        weight_ptr, Kk, &beta, output_batch, Nn);
                }
                else if constexpr (std::is_same_v<T, double>)
                {
                    double alpha = 1.0, beta = 0.0;
                    stat = cublasDgemm(blas_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                        Nn, M, Kk, &alpha, d_col, Nn,
                        weight_ptr, Kk, &beta, output_batch, Nn);
                }
                else if constexpr (detail::is_gemmex_type<T>::value)
                {
                    detail::gemm_ex<T>(blas_handle.get(),
                        static_cast<size_t>(M), static_cast<size_t>(Nn), static_cast<size_t>(Kk),
                        weight_ptr, d_col, output_batch);
                    stat = CUBLAS_STATUS_SUCCESS;
                }
                else
                {
                    pool.release(d_col);
                    TENSOR_THROW("cuBLAS conv2d only supports float/double");
                }

                if (stat != CUBLAS_STATUS_SUCCESS)
                {
                    pool.release(d_col);
                    TENSOR_THROW("cuBLAS gemm failed in conv2d");
                }

                pool.release(d_col);
            }

            if (bias.size() > 0)
            {
                size_t total = batch * out_channels * out_height * out_width;
                size_t bs = get_optimal_block_size(total);
                size_t gs = get_grid_size(total, bs);
                bias_add_kernel<<<gs, bs, 0, stream>>>(output.device_ptr(), bias.device_ptr(),
                    batch, out_channels, out_height, out_width);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }
        }

        template <typename T>
        void conv2d(const CudaTensor<T>& input,
                   const CudaTensor<T>& weight,
                   const CudaTensor<T>& bias,
                   CudaTensor<T>& output,
                   int stride,
                   int padding)
        {
            conv2d(input, weight, bias, output, stride, padding, nullptr);
        }

        template <typename T>
        void conv2d(const CudaTensor<T>& input,
                   const CudaTensor<T>& weight,
                   CudaTensor<T>& output,
                   int stride,
                   int padding)
        {
            CudaTensor<T> empty_bias;
            conv2d(input, weight, empty_bias, output, stride, padding, nullptr);
        }

        template <typename T>
        __global__ void conv_transpose2d_kernel(const T* input, const T* weight, T* output,
                                               size_t batch, size_t in_channels, size_t out_channels,
                                               size_t in_height, size_t in_width,
                                               size_t kernel_h, size_t kernel_w,
                                               size_t out_height, size_t out_width,
                                               int stride, int padding)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total = batch * out_channels * out_height * out_width;

            if (idx < total)
            {
                size_t ow = idx % out_width;
                size_t oh = (idx / out_width) % out_height;
                size_t oc = (idx / (out_width * out_height)) % out_channels;
                size_t b = idx / (out_width * out_height * out_channels);

                T sum = 0;

                for (size_t ic = 0; ic < in_channels; ++ic)
                {
                    for (size_t kh = 0; kh < kernel_h; ++kh)
                    {
                        for (size_t kw = 0; kw < kernel_w; ++kw)
                        {
                            int ih = static_cast<int>(oh) + static_cast<int>(padding) - static_cast<int>(kh);
                            int iw = static_cast<int>(ow) + static_cast<int>(padding) - static_cast<int>(kw);

                            if (ih >= 0 && ih % stride == 0 &&
                                iw >= 0 && iw % stride == 0)
                            {
                                ih /= stride;
                                iw /= stride;
                                if (static_cast<size_t>(ih) < in_height && static_cast<size_t>(iw) < in_width)
                                {
                                    size_t input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                                    size_t weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                                    sum += input[input_idx] * weight[weight_idx];
                                }
                            }
                        }
                    }
                }

                output[idx] = sum;
            }
        }

        template <typename T>
        void conv_transpose2d(const CudaTensor<T>& input,
                             const CudaTensor<T>& weight,
                             const CudaTensor<T>& bias,
                             CudaTensor<T>& output,
                             int stride,
                             int padding,
                             cudaStream_t stream)
        {
            if (input.shape().size() != 4 || weight.shape().size() != 4 || output.shape().size() != 4)
                TENSOR_THROW("conv_transpose2d requires 4D tensors");

            size_t batch = input.shape()[0];
            size_t in_channels = input.shape()[1];
            size_t in_height = input.shape()[2];
            size_t in_width = input.shape()[3];

            size_t out_channels = weight.shape()[0];
            size_t kernel_h = weight.shape()[2];
            size_t kernel_w = weight.shape()[3];

            size_t out_height = (in_height - 1) * stride - 2 * padding + kernel_h;
            size_t out_width = (in_width - 1) * stride - 2 * padding + kernel_w;

            if (output.shape()[0] != batch || output.shape()[1] != out_channels ||
                output.shape()[2] != out_height || output.shape()[3] != out_width)
                TENSOR_THROW("Output tensor has wrong shape");

            size_t total = batch * out_channels * out_height * out_width;
            size_t block_size = get_optimal_block_size(total);
            size_t grid_size = get_grid_size(total, block_size);

            conv_transpose2d_kernel<<<grid_size, block_size, 0, stream>>>(
                input.device_ptr(), weight.device_ptr(), output.device_ptr(),
                batch, in_channels, out_channels, in_height, in_width,
                kernel_h, kernel_w, out_height, out_width, stride, padding);

            CHECK_CUDA_ERROR(cudaGetLastError());

            if (bias.size() > 0)
            {
                bias_add_kernel<<<grid_size, block_size, 0, stream>>>(output.device_ptr(), bias.device_ptr(),
                    batch, out_channels, out_height, out_width);
                CHECK_CUDA_ERROR(cudaGetLastError());
            }
        }

        template <typename T>
        void conv_transpose2d(const CudaTensor<T>& input,
                             const CudaTensor<T>& weight,
                             const CudaTensor<T>& bias,
                             CudaTensor<T>& output,
                             int stride,
                             int padding)
        {
            conv_transpose2d(input, weight, bias, output, stride, padding, nullptr);
        }

        template void conv2d<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, int, int);
        template void conv2d<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, int, int);
        template void conv2d<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, int, int, cudaStream_t);
        template void conv2d<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, int, int, cudaStream_t);
        template void conv2d<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, int, int);
        template void conv2d<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, int, int);
        template void conv_transpose2d<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, int, int);
        template void conv_transpose2d<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, int, int);
        template void conv_transpose2d<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, int, int, cudaStream_t);
        template void conv_transpose2d<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, int, int, cudaStream_t);

#define INST_LOWP(T) \
        template void conv2d<T>(const CudaTensor<T>&, const CudaTensor<T>&, const CudaTensor<T>&, CudaTensor<T>&, int, int); \
        template void conv2d<T>(const CudaTensor<T>&, const CudaTensor<T>&, const CudaTensor<T>&, CudaTensor<T>&, int, int, cudaStream_t); \
        template void conv2d<T>(const CudaTensor<T>&, const CudaTensor<T>&, CudaTensor<T>&, int, int); \
        template void conv_transpose2d<T>(const CudaTensor<T>&, const CudaTensor<T>&, const CudaTensor<T>&, CudaTensor<T>&, int, int); \
        template void conv_transpose2d<T>(const CudaTensor<T>&, const CudaTensor<T>&, const CudaTensor<T>&, CudaTensor<T>&, int, int, cudaStream_t);

        INST_LOWP(TensorN::half)
        INST_LOWP(TensorN::bfloat16)
        INST_LOWP(TensorN::tf32)
#if CUDART_VERSION >= 12000
        INST_LOWP(TensorN::fp8_e4m3)
        INST_LOWP(TensorN::fp8_e5m2)
#endif

    } // namespace cuda
} // namespace TensorN
