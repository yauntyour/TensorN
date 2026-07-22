#include "fused_kernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <cmath>

namespace TensorN { namespace cuda {

template <typename T>
__device__ __forceinline__ T apply_activation(T x, ActivationType act, T alpha)
{
    switch (act)
    {
    case ActivationType::ReLU:
        return x > T(0) ? x : T(0);
    case ActivationType::Sigmoid:
        return T(1) / (T(1) + ::exp(-x));
    case ActivationType::Tanh:
        return ::tanh(x);
    case ActivationType::GELU:
    {
        T cdf = T(0.5) * (T(1) + ::tanh(T(0.7978845608028654) * (x + T(0.044715) * x * x * x)));
        return x * cdf;
    }
    case ActivationType::LeakyReLU:
        return x > T(0) ? x : alpha * x;
    case ActivationType::ELU:
        return x > T(0) ? x : alpha * (::exp(x) - T(1));
    default:
        return x;
    }
}

template <typename T>
__global__ void fused_activation_kernel(T* data, size_t n, ActivationType act, T alpha)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = apply_activation(data[idx], act, alpha);
}

template <typename T>
__global__ void fused_add_relu_kernel(const T* A, const T* B, T* C, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        T val = A[idx] + B[idx];
        C[idx] = val > T(0) ? val : T(0);
    }
}

template <typename T>
__global__ void fused_mul_add_kernel(const T* A, const T* B, const T* C, T* D, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        D[idx] = A[idx] * B[idx] + C[idx];
}

template <typename T>
__global__ void fused_conv2d_activation_kernel(
    const T* input, const T* weight, const T* bias, T* output,
    size_t batch, size_t in_channels, size_t out_channels,
    size_t height, size_t width,
    size_t kernel_h, size_t kernel_w,
    size_t out_height, size_t out_width,
    int stride, int padding, size_t bias_size,
    ActivationType act, T alpha)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * out_channels * out_height * out_width;
    if (idx >= total) return;

    size_t ow = idx % out_width;
    size_t oh = (idx / out_width) % out_height;
    size_t oc = (idx / (out_width * out_height)) % out_channels;
    size_t b = idx / (out_width * out_height * out_channels);

    T sum = T(0);
    for (size_t ic = 0; ic < in_channels; ++ic)
        for (size_t kh = 0; kh < kernel_h; ++kh)
            for (size_t kw = 0; kw < kernel_w; ++kw)
            {
                int ih = static_cast<int>(oh) * stride - padding + static_cast<int>(kh);
                int iw = static_cast<int>(ow) * stride - padding + static_cast<int>(kw);
                if (ih >= 0 && static_cast<size_t>(ih) < height &&
                    iw >= 0 && static_cast<size_t>(iw) < width)
                {
                    sum += input[((b * in_channels + ic) * height + ih) * width + iw]
                         * weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                }
            }

    if (bias != nullptr && bias_size > 0)
        sum += bias[oc];

    output[idx] = apply_activation(sum, act, alpha);
}

template <typename T>
__global__ void fused_batchnorm_inference_kernel(
    const T* input, const T* mean, const T* var,
    const T* gamma, const T* beta, T* output,
    T eps, size_t channels, size_t spatial)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = channels * spatial;
    if (idx >= total) return;

    size_t c = idx / spatial;
    T inv_std = T(1) / ::sqrt(var[c] + eps);
    output[idx] = gamma[c] * (input[idx] - mean[c]) * inv_std + beta[c];
}

template <typename T>
void matmul_activation(const CudaTensor<T>& A, const CudaTensor<T>& B,
                       CudaTensor<T>& C, ActivationType act, T alpha_param,
                       cudaStream_t stream)
{
    if (A.shape().size() != 2 || B.shape().size() != 2 || C.shape().size() != 2)
        TENSOR_THROW("matmul_activation requires 2D tensors");

    size_t M = A.shape()[0], K = A.shape()[1], N = B.shape()[1];
    if (B.shape()[0] != K || C.shape()[0] != M || C.shape()[1] != N)
        TENSOR_THROW("matmul_activation dimension mismatch");

    auto& blas_handle = get_stream_blas_handle();
    blas_handle.set_stream(stream);

    T alpha = T(1), beta = T(0);
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
        TENSOR_THROW("matmul_activation only supports float/double");

    if (stat != CUBLAS_STATUS_SUCCESS)
        TENSOR_THROW("cuBLAS gemm failed in matmul_activation");

    if (act != ActivationType::None)
    {
        size_t n = M * N;
        size_t bs = 256, gs = (n + bs - 1) / bs;
        fused_activation_kernel<<<gs, bs, 0, stream>>>(C.device_ptr(), n, act, alpha_param);
    }
}

template <typename T>
void conv2d_activation(const CudaTensor<T>& input, const CudaTensor<T>& weight,
                       const CudaTensor<T>& bias, CudaTensor<T>& output,
                       int stride, int padding, ActivationType act, T alpha_param,
                       cudaStream_t stream)
{
    size_t batch = input.shape()[0], in_channels = input.shape()[1];
    size_t height = input.shape()[2], width = input.shape()[3];
    size_t out_channels = weight.shape()[0];
    size_t kernel_h = weight.shape()[2], kernel_w = weight.shape()[3];
    size_t out_height = (height + 2 * padding - kernel_h) / stride + 1;
    size_t out_width = (width + 2 * padding - kernel_w) / stride + 1;

    size_t total = batch * out_channels * out_height * out_width;
    size_t bs = 256, gs = (total + bs - 1) / bs;
    size_t bias_size = bias.size();

    fused_conv2d_activation_kernel<<<gs, bs, 0, stream>>>(
        input.device_ptr(), weight.device_ptr(),
        bias_size > 0 ? bias.device_ptr() : nullptr,
        output.device_ptr(),
        batch, in_channels, out_channels, height, width,
        kernel_h, kernel_w, out_height, out_width,
        stride, padding, bias_size, act, alpha_param);
}

template <typename T>
void add_relu(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C,
              cudaStream_t stream)
{
    size_t n = A.size();
    size_t bs = 256, gs = (n + bs - 1) / bs;
    fused_add_relu_kernel<<<gs, bs, 0, stream>>>(A.device_ptr(), B.device_ptr(), C.device_ptr(), n);
}

template <typename T>
void mul_add(const CudaTensor<T>& A, const CudaTensor<T>& B, const CudaTensor<T>& C,
             CudaTensor<T>& D, cudaStream_t stream)
{
    size_t n = A.size();
    size_t bs = 256, gs = (n + bs - 1) / bs;
    fused_mul_add_kernel<<<gs, bs, 0, stream>>>(A.device_ptr(), B.device_ptr(), C.device_ptr(), D.device_ptr(), n);
}

template <typename T>
void batchnorm_inference(const CudaTensor<T>& input, const CudaTensor<T>& mean,
                         const CudaTensor<T>& var, const CudaTensor<T>& gamma,
                         const CudaTensor<T>& beta, CudaTensor<T>& output,
                         T eps, size_t channels, cudaStream_t stream)
{
    size_t total = input.size();
    size_t spatial = total / channels;
    size_t bs = 256, gs = (total + bs - 1) / bs;
    fused_batchnorm_inference_kernel<<<gs, bs, 0, stream>>>(
        input.device_ptr(), mean.device_ptr(), var.device_ptr(),
        gamma.device_ptr(), beta.device_ptr(), output.device_ptr(),
        eps, channels, spatial);
}

template <typename T>
void residual_block(const CudaTensor<T>& input, const CudaTensor<T>& weight1,
                   const CudaTensor<T>& bias1, const CudaTensor<T>& weight2,
                   const CudaTensor<T>& bias2, CudaTensor<T>& output,
                   int stride, int padding, cudaStream_t stream)
{
    size_t batch = input.shape()[0], in_ch = input.shape()[1];
    size_t H = input.shape()[2], W = input.shape()[3];
    size_t mid_ch = weight1.shape()[0];
    size_t kH1 = weight1.shape()[2], kW1 = weight1.shape()[3];
    size_t oH1 = (H + 2 * padding - kH1) / stride + 1;
    size_t oW1 = (W + 2 * padding - kW1) / stride + 1;

    CudaTensor<T> temp1({batch, mid_ch, oH1, oW1});
    conv2d_activation(input, weight1, bias1, temp1, stride, padding,
                      ActivationType::ReLU, T(0), stream);

    size_t kH2 = weight2.shape()[2], kW2 = weight2.shape()[3];
    size_t out_ch = weight2.shape()[0];
    size_t oH2 = (oH1 + 2 * padding - kH2) / stride + 1;
    size_t oW2 = (oW1 + 2 * padding - kW2) / stride + 1;

    CudaTensor<T> temp2({batch, out_ch, oH2, oW2});
    conv2d_activation(temp1, weight2, bias2, temp2, stride, padding,
                      ActivationType::None, T(0), stream);

    if (input.shape() == temp2.shape())
    {
        size_t n = temp2.size();
        size_t bs = 256, gs = (n + bs - 1) / bs;
        fused_add_relu_kernel<<<gs, bs, 0, stream>>>(
            temp2.device_ptr(), input.device_ptr(), output.device_ptr(), n);
    }
    else
    {
        size_t n = temp2.size();
        cudaMemcpyAsync(output.device_ptr(), temp2.device_ptr(), n * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream);
    }
}

template void matmul_activation<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, ActivationType, float, cudaStream_t);
template void matmul_activation<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, ActivationType, double, cudaStream_t);
template void conv2d_activation<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, int, int, ActivationType, float, cudaStream_t);
template void conv2d_activation<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, int, int, ActivationType, double, cudaStream_t);
template void add_relu<float>(const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
template void add_relu<double>(const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
template void mul_add<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, cudaStream_t);
template void mul_add<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, cudaStream_t);
template void batchnorm_inference<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, float, size_t, cudaStream_t);
template void batchnorm_inference<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, double, size_t, cudaStream_t);
template void residual_block<float>(const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, const CudaTensor<float>&, CudaTensor<float>&, int, int, cudaStream_t);
template void residual_block<double>(const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, const CudaTensor<double>&, CudaTensor<double>&, int, int, cudaStream_t);

}} // namespace TensorN::cuda
