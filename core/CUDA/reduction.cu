#include "reduction.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <cmath>

namespace TensorN
{
    namespace cuda
    {
        template <typename T, typename Op>
        __device__ inline T device_op(T a, T b, Op) { return Op{}(a, b); }

        template <typename T>
        __global__ void reduce_kernel_sum(const T* input, T* output, size_t n)
        {
            extern __shared__ char shared_mem[];
            T* sdata = reinterpret_cast<T*>(shared_mem);

            size_t tid = threadIdx.x;
            size_t i = blockIdx.x * blockDim.x * 2 + tid;

            T val = T(0);
            if (i < n) val += input[i];
            if (i + blockDim.x < n) val += input[i + blockDim.x];

            sdata[tid] = val;
            __syncthreads();

            for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                    sdata[tid] += sdata[tid + s];
                __syncthreads();
            }

            if (tid == 0) output[blockIdx.x] = sdata[0];
        }

        template <typename T>
        __global__ void reduce_kernel_max(const T* input, T* output, size_t n)
        {
            extern __shared__ char shared_mem[];
            T* sdata = reinterpret_cast<T*>(shared_mem);

            size_t tid = threadIdx.x;
            size_t i = blockIdx.x * blockDim.x * 2 + tid;

            T val = std::numeric_limits<T>::lowest();
            if (i < n) val = val > input[i] ? val : input[i];
            if (i + blockDim.x < n) val = val > input[i + blockDim.x] ? val : input[i + blockDim.x];

            sdata[tid] = val;
            __syncthreads();

            for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                    sdata[tid] = sdata[tid] > sdata[tid + s] ? sdata[tid] : sdata[tid + s];
                __syncthreads();
            }

            if (tid == 0) output[blockIdx.x] = sdata[0];
        }

        template <typename T>
        __global__ void reduce_kernel_min(const T* input, T* output, size_t n)
        {
            extern __shared__ char shared_mem[];
            T* sdata = reinterpret_cast<T*>(shared_mem);

            size_t tid = threadIdx.x;
            size_t i = blockIdx.x * blockDim.x * 2 + tid;

            T val = std::numeric_limits<T>::max();
            if (i < n) val = val < input[i] ? val : input[i];
            if (i + blockDim.x < n) val = val < input[i + blockDim.x] ? val : input[i + blockDim.x];

            sdata[tid] = val;
            __syncthreads();

            for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                    sdata[tid] = sdata[tid] < sdata[tid + s] ? sdata[tid] : sdata[tid + s];
                __syncthreads();
            }

            if (tid == 0) output[blockIdx.x] = sdata[0];
        }

        template <typename T>
        T reduce_global_sum(const CudaTensor<T>& A)
        {
            size_t n = A.size();
            if (n == 0) return T(0);

            size_t block_size = 256;
            size_t grid_size = (n + block_size * 2 - 1) / (block_size * 2);

            T* d_intermediate;
            cudaMalloc(&d_intermediate, grid_size * sizeof(T));

            reduce_kernel_sum<<<grid_size, block_size, block_size * sizeof(T)>>>(A.device_ptr(), d_intermediate, n);
            CHECK_CUDA_ERROR(cudaGetLastError());

            std::vector<T> h_intermediate(grid_size);
            cudaMemcpy(h_intermediate.data(), d_intermediate, grid_size * sizeof(T), cudaMemcpyDeviceToHost);

            T result = T(0);
            for (size_t i = 0; i < grid_size; ++i) result += h_intermediate[i];

            cudaFree(d_intermediate);
            return result;
        }

        template <typename T>
        T reduce_global_max(const CudaTensor<T>& A)
        {
            size_t n = A.size();
            if (n == 0) return std::numeric_limits<T>::lowest();

            size_t block_size = 256;
            size_t grid_size = (n + block_size * 2 - 1) / (block_size * 2);

            T* d_intermediate;
            cudaMalloc(&d_intermediate, grid_size * sizeof(T));

            reduce_kernel_max<<<grid_size, block_size, block_size * sizeof(T)>>>(A.device_ptr(), d_intermediate, n);
            CHECK_CUDA_ERROR(cudaGetLastError());

            std::vector<T> h_intermediate(grid_size);
            cudaMemcpy(h_intermediate.data(), d_intermediate, grid_size * sizeof(T), cudaMemcpyDeviceToHost);

            T result = std::numeric_limits<T>::lowest();
            for (size_t i = 0; i < grid_size; ++i)
                if (h_intermediate[i] > result) result = h_intermediate[i];

            cudaFree(d_intermediate);
            return result;
        }

        template <typename T>
        T reduce_global_min(const CudaTensor<T>& A)
        {
            size_t n = A.size();
            if (n == 0) return std::numeric_limits<T>::max();

            size_t block_size = 256;
            size_t grid_size = (n + block_size * 2 - 1) / (block_size * 2);

            T* d_intermediate;
            cudaMalloc(&d_intermediate, grid_size * sizeof(T));

            reduce_kernel_min<<<grid_size, block_size, block_size * sizeof(T)>>>(A.device_ptr(), d_intermediate, n);
            CHECK_CUDA_ERROR(cudaGetLastError());

            std::vector<T> h_intermediate(grid_size);
            cudaMemcpy(h_intermediate.data(), d_intermediate, grid_size * sizeof(T), cudaMemcpyDeviceToHost);

            T result = std::numeric_limits<T>::max();
            for (size_t i = 0; i < grid_size; ++i)
                if (h_intermediate[i] < result) result = h_intermediate[i];

            cudaFree(d_intermediate);
            return result;
        }

        template <typename T>
        T sum(const CudaTensor<T>& A) { return reduce_global_sum(A); }

        template <typename T>
        T mean(const CudaTensor<T>& A)
        {
            T total = sum(A);
            return total / static_cast<T>(A.size());
        }

        template <typename T>
        T max(const CudaTensor<T>& A) { return reduce_global_max(A); }

        template <typename T>
        T min(const CudaTensor<T>& A) { return reduce_global_min(A); }

        // ---- Axis reductions ----

        template <typename T>
        __global__ void sum_axis_kernel(const T* input, T* output, size_t outer, size_t inner, size_t reduce_dim)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= outer * inner) return;

            size_t o = idx / inner;
            size_t i = idx % inner;

            T sum_val = T(0);
            for (size_t r = 0; r < reduce_dim; ++r)
                sum_val += input[o * reduce_dim * inner + r * inner + i];

            output[idx] = sum_val;
        }

        template <typename T>
        __global__ void max_axis_kernel(const T* input, T* output, size_t outer, size_t inner, size_t reduce_dim)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= outer * inner) return;

            size_t o = idx / inner;
            size_t i = idx % inner;

            T max_val = std::numeric_limits<T>::lowest();
            for (size_t r = 0; r < reduce_dim; ++r)
            {
                T v = input[o * reduce_dim * inner + r * inner + i];
                if (v > max_val) max_val = v;
            }
            output[idx] = max_val;
        }

        template <typename T>
        __global__ void min_axis_kernel(const T* input, T* output, size_t outer, size_t inner, size_t reduce_dim)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= outer * inner) return;

            size_t o = idx / inner;
            size_t i = idx % inner;

            T min_val = std::numeric_limits<T>::max();
            for (size_t r = 0; r < reduce_dim; ++r)
            {
                T v = input[o * reduce_dim * inner + r * inner + i];
                if (v < min_val) min_val = v;
            }
            output[idx] = min_val;
        }

        template <typename T>
        CudaTensor<T> reduce_axis_impl(const CudaTensor<T>& A, int axis,
            void(*kernel)(const T*, T*, size_t, size_t, size_t))
        {
            const auto& shape = A.shape();
            size_t ndim = shape.size();

            if (axis < 0) axis = static_cast<int>(ndim) + axis;
            if (axis < 0 || static_cast<size_t>(axis) >= ndim)
                throw std::invalid_argument("Axis out of range");

            size_t outer = 1, reduce_dim = shape[axis], inner = 1;
            for (size_t d = 0; d < static_cast<size_t>(axis); ++d) outer *= shape[d];
            for (size_t d = static_cast<size_t>(axis) + 1; d < ndim; ++d) inner *= shape[d];

            std::vector<size_t> out_shape;
            for (size_t d = 0; d < ndim; ++d)
                if (d != static_cast<size_t>(axis)) out_shape.push_back(shape[d]);

            CudaTensor<T> C(out_shape);

            size_t total = outer * inner;
            size_t block_size = 256;
            size_t grid_size = (total + block_size - 1) / block_size;

            kernel<<<grid_size, block_size>>>(A.device_ptr(), C.device_ptr(), outer, inner, reduce_dim);
            CHECK_CUDA_ERROR(cudaGetLastError());

            return C;
        }

        template <typename T>
        __global__ void scale_kernel(T* data, T factor, size_t n)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) data[idx] *= factor;
        }

        template <typename T>
        CudaTensor<T> sum_axis(const CudaTensor<T>& A, int axis)
        {
            return reduce_axis_impl(A, axis, sum_axis_kernel<T>);
        }

        template <typename T>
        CudaTensor<T> mean_axis(const CudaTensor<T>& A, int axis)
        {
            if (axis < 0) axis = static_cast<int>(A.shape().size()) + axis;
            CudaTensor<T> C = sum_axis(A, axis);
            T factor = T(1) / static_cast<T>(A.shape()[axis]);

            size_t n = C.size();
            size_t block_size = 256;
            size_t grid_size = (n + block_size - 1) / block_size;
            scale_kernel<<<grid_size, block_size>>>(C.device_ptr(), factor, n);
            CHECK_CUDA_ERROR(cudaGetLastError());

            return C;
        }

        template <typename T>
        CudaTensor<T> max_axis(const CudaTensor<T>& A, int axis)
        {
            return reduce_axis_impl(A, axis, max_axis_kernel<T>);
        }

        template <typename T>
        CudaTensor<T> min_axis(const CudaTensor<T>& A, int axis)
        {
            return reduce_axis_impl(A, axis, min_axis_kernel<T>);
        }

        // ---- Argmax / Argmin ----

        template <typename T>
        __global__ void argmax_kernel(const T* input, int64_t* output,
            size_t outer, size_t inner, size_t reduce_dim)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= outer * inner) return;

            size_t o = idx / inner;
            size_t i = idx % inner;

            T max_val = std::numeric_limits<T>::lowest();
            int64_t max_idx = 0;
            for (size_t r = 0; r < reduce_dim; ++r)
            {
                T v = input[o * reduce_dim * inner + r * inner + i];
                if (v > max_val) { max_val = v; max_idx = static_cast<int64_t>(r); }
            }
            output[idx] = max_idx;
        }

        template <typename T>
        __global__ void argmin_kernel(const T* input, int64_t* output,
            size_t outer, size_t inner, size_t reduce_dim)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= outer * inner) return;

            size_t o = idx / inner;
            size_t i = idx % inner;

            T min_val = std::numeric_limits<T>::max();
            int64_t min_idx = 0;
            for (size_t r = 0; r < reduce_dim; ++r)
            {
                T v = input[o * reduce_dim * inner + r * inner + i];
                if (v < min_val) { min_val = v; min_idx = static_cast<int64_t>(r); }
            }
            output[idx] = min_idx;
        }

        template <typename T>
        CudaTensor<int64_t> argmax(const CudaTensor<T>& A, int axis)
        {
            const auto& shape = A.shape();
            size_t ndim = shape.size();
            if (axis < 0) axis = static_cast<int>(ndim) + axis;
            if (axis < 0 || static_cast<size_t>(axis) >= ndim)
                throw std::invalid_argument("Axis out of range");

            size_t outer = 1, reduce_dim = shape[axis], inner = 1;
            for (size_t d = 0; d < static_cast<size_t>(axis); ++d) outer *= shape[d];
            for (size_t d = static_cast<size_t>(axis) + 1; d < ndim; ++d) inner *= shape[d];

            std::vector<size_t> out_shape;
            for (size_t d = 0; d < ndim; ++d)
                if (d != static_cast<size_t>(axis)) out_shape.push_back(shape[d]);

            CudaTensor<int64_t> C(out_shape);

            size_t total = outer * inner;
            size_t block_size = 256;
            size_t grid_size = (total + block_size - 1) / block_size;

            argmax_kernel<<<grid_size, block_size>>>(A.device_ptr(), C.device_ptr(), outer, inner, reduce_dim);
            CHECK_CUDA_ERROR(cudaGetLastError());

            return C;
        }

        template <typename T>
        CudaTensor<int64_t> argmin(const CudaTensor<T>& A, int axis)
        {
            const auto& shape = A.shape();
            size_t ndim = shape.size();
            if (axis < 0) axis = static_cast<int>(ndim) + axis;
            if (axis < 0 || static_cast<size_t>(axis) >= ndim)
                throw std::invalid_argument("Axis out of range");

            size_t outer = 1, reduce_dim = shape[axis], inner = 1;
            for (size_t d = 0; d < static_cast<size_t>(axis); ++d) outer *= shape[d];
            for (size_t d = static_cast<size_t>(axis) + 1; d < ndim; ++d) inner *= shape[d];

            std::vector<size_t> out_shape;
            for (size_t d = 0; d < ndim; ++d)
                if (d != static_cast<size_t>(axis)) out_shape.push_back(shape[d]);

            CudaTensor<int64_t> C(out_shape);

            size_t total = outer * inner;
            size_t block_size = 256;
            size_t grid_size = (total + block_size - 1) / block_size;

            argmin_kernel<<<grid_size, block_size>>>(A.device_ptr(), C.device_ptr(), outer, inner, reduce_dim);
            CHECK_CUDA_ERROR(cudaGetLastError());

            return C;
        }

        template float sum<float>(const CudaTensor<float>&);
        template double sum<double>(const CudaTensor<double>&);
        template float mean<float>(const CudaTensor<float>&);
        template double mean<double>(const CudaTensor<double>&);
        template float max<float>(const CudaTensor<float>&);
        template double max<double>(const CudaTensor<double>&);
        template float min<float>(const CudaTensor<float>&);
        template double min<double>(const CudaTensor<double>&);

        template CudaTensor<float> sum_axis<float>(const CudaTensor<float>&, int);
        template CudaTensor<double> sum_axis<double>(const CudaTensor<double>&, int);
        template CudaTensor<float> mean_axis<float>(const CudaTensor<float>&, int);
        template CudaTensor<double> mean_axis<double>(const CudaTensor<double>&, int);
        template CudaTensor<float> max_axis<float>(const CudaTensor<float>&, int);
        template CudaTensor<double> max_axis<double>(const CudaTensor<double>&, int);
        template CudaTensor<float> min_axis<float>(const CudaTensor<float>&, int);
        template CudaTensor<double> min_axis<double>(const CudaTensor<double>&, int);

        template CudaTensor<int64_t> argmax<float>(const CudaTensor<float>&, int);
        template CudaTensor<int64_t> argmax<double>(const CudaTensor<double>&, int);
        template CudaTensor<int64_t> argmin<float>(const CudaTensor<float>&, int);
        template CudaTensor<int64_t> argmin<double>(const CudaTensor<double>&, int);

    } // namespace cuda
} // namespace TensorN
