#include "convolution.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace TensorN
{
    namespace cuda
    {
        template <typename T>
        __global__ void conv2d_kernel(const T* input, const T* weight, const T* bias, T* output,
                                     size_t batch, size_t in_channels, size_t out_channels,
                                     size_t height, size_t width,
                                     size_t kernel_h, size_t kernel_w,
                                     size_t out_height, size_t out_width,
                                     int stride, int padding, size_t bias_size)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total = batch * out_channels * out_height * out_width;

            if (idx < total)
            {
                size_t w = idx % out_width;
                size_t h = (idx / out_width) % out_height;
                size_t oc = (idx / (out_width * out_height)) % out_channels;
                size_t b = idx / (out_width * out_height * out_channels);

                T sum = 0;

                for (size_t ic = 0; ic < in_channels; ++ic)
                {
                    for (size_t kh = 0; kh < kernel_h; ++kh)
                    {
                        for (size_t kw = 0; kw < kernel_w; ++kw)
                        {
                            int ih = static_cast<int>(h) * stride - padding + static_cast<int>(kh);
                            int iw = static_cast<int>(w) * stride - padding + static_cast<int>(kw);

                            if (ih >= 0 && static_cast<size_t>(ih) < height &&
                                iw >= 0 && static_cast<size_t>(iw) < width)
                            {
                                size_t input_idx = ((b * in_channels + ic) * height + ih) * width + iw;
                                size_t weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }

                if (bias != nullptr && bias_size > 0)
                    sum += bias[oc];

                output[idx] = sum;
            }
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
        void conv2d(const CudaTensor<T>& input,
                   const CudaTensor<T>& weight,
                   const CudaTensor<T>& bias,
                   CudaTensor<T>& output,
                   int stride,
                   int padding,
                   cudaStream_t stream)
        {
            if (input.shape().size() != 4 || weight.shape().size() != 4 || output.shape().size() != 4)
                throw std::invalid_argument("conv2d requires 4D tensors");

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
                throw std::invalid_argument("Output tensor has wrong shape");

            size_t total = batch * out_channels * out_height * out_width;
            size_t block_size = 256;
            size_t grid_size = (total + block_size - 1) / block_size;

            size_t bias_size = bias.size();

            conv2d_kernel<<<grid_size, block_size, 0, stream>>>(
                input.device_ptr(), weight.device_ptr(),
                bias_size > 0 ? bias.device_ptr() : nullptr,
                output.device_ptr(),
                batch, in_channels, out_channels, height, width,
                kernel_h, kernel_w, out_height, out_width,
                stride, padding, bias_size);

            CHECK_CUDA_ERROR(cudaGetLastError());
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
        __global__ void bias_add_kernel(T* output, const T* bias, size_t batch, size_t out_channels,
                                        size_t out_height, size_t out_width) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total = batch * out_channels * out_height * out_width;
            if (idx < total) {
                size_t oc = (idx / (out_width * out_height)) % out_channels;
                output[idx] += bias[oc];
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
                throw std::invalid_argument("conv_transpose2d requires 4D tensors");

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
                throw std::invalid_argument("Output tensor has wrong shape");

            size_t total = batch * out_channels * out_height * out_width;
            size_t block_size = 256;
            size_t grid_size = (total + block_size - 1) / block_size;

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

    } // namespace cuda
} // namespace TensorN
