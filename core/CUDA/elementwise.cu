#include "elementwise.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace TensorN { namespace cuda { namespace kernels {

template <typename T>
__global__ void add(const T* A, const T* B, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}
template <typename T>
__global__ void subtract(const T* A, const T* B, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] - B[idx];
}
template <typename T>
__global__ void multiply(const T* A, const T* B, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] * B[idx];
}
template <typename T>
__global__ void divide(const T* A, const T* B, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] / B[idx];
}
template <typename T>
__global__ void add_scalar(const T* A, T scalar, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + scalar;
}
template <typename T>
__global__ void multiply_scalar(const T* A, T scalar, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] * scalar;
}
template <typename T>
__global__ void subtract_scalar(const T* A, T scalar, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] - scalar;
}
template <typename T>
__global__ void divide_scalar(const T* A, T scalar, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] / scalar;
}
template <typename T>
__global__ void negate(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = -A[idx];
}
template <typename T>
__global__ void abs(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::abs(A[idx]);
}
template <typename T>
__global__ void sqrt(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::sqrt(A[idx]);
}
template <typename T>
__global__ void exp(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::exp(A[idx]);
}
template <typename T>
__global__ void log(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::log(A[idx]);
}
template <typename T>
__global__ void sin(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::sin(A[idx]);
}
template <typename T>
__global__ void cos(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::cos(A[idx]);
}
template <typename T>
__global__ void pow(const T* A, T exponent, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::pow(A[idx], exponent);
}
template <typename T>
__global__ void clip(const T* A, T min_val, T max_val, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::min(::max(A[idx], min_val), max_val);
}
template <typename T>
__global__ void relu(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] > T(0) ? A[idx] : T(0);
}
template <typename T>
__global__ void leaky_relu(const T* A, T alpha, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] > T(0) ? A[idx] : alpha * A[idx];
}
template <typename T>
__global__ void elu(const T* A, T alpha, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] > T(0) ? A[idx] : alpha * (::exp(A[idx]) - T(1));
}
template <typename T>
__global__ void gelu(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = A[idx];
        T cdf = T(0.5) * (T(1) + ::tanh(T(0.7978845608028654) * (x + T(0.044715) * x * x * x)));
        C[idx] = x * cdf;
    }
}
template <typename T>
__global__ void sigmoid(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = T(1) / (T(1) + ::exp(-A[idx]));
}
template <typename T>
__global__ void tanh(const T* A, T* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = ::tanh(A[idx]);
}

template <typename T>
__global__ void equal(const T* A, const T* B, int* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = (A[idx] == B[idx]) ? 1 : 0;
}
template <typename T>
__global__ void not_equal(const T* A, const T* B, int* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = (A[idx] != B[idx]) ? 1 : 0;
}
template <typename T>
__global__ void greater(const T* A, const T* B, int* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = (A[idx] > B[idx]) ? 1 : 0;
}
template <typename T>
__global__ void less(const T* A, const T* B, int* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = (A[idx] < B[idx]) ? 1 : 0;
}
template <typename T>
__global__ void greater_equal(const T* A, const T* B, int* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = (A[idx] >= B[idx]) ? 1 : 0;
}
template <typename T>
__global__ void less_equal(const T* A, const T* B, int* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = (A[idx] <= B[idx]) ? 1 : 0;
}

template <typename T>
__global__ void transpose(const T* in, T* out, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    size_t r = idx / cols, c = idx % cols;
    out[c * rows + r] = in[idx];
}
template <typename T>
__global__ void transpose_back(const T* in, T* out, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    size_t r = idx / rows, c = idx % rows;
    out[c * cols + r] = in[idx];
}

template <typename T>
__global__ void softmax_max(const T* input, T* row_max, size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    if (row >= rows) return;
    const T* row_ptr = input + row * cols;
    T mx = row_ptr[0];
    for (size_t j = 1; j < cols; ++j)
        if (row_ptr[j] > mx) mx = row_ptr[j];
    row_max[row] = mx;
}
template <typename T>
__global__ void softmax_exp_sum(const T* input, const T* row_max, T* output, T* row_sum, size_t rows, size_t cols) {
    size_t row = blockIdx.x, col = threadIdx.x;
    if (row >= rows || col >= cols) return;
    size_t idx = row * cols + col;
    T v = ::exp(input[idx] - row_max[row]);
    output[idx] = v;
    atomicAdd(&row_sum[row], v);
}
template <typename T>
__global__ void softmax_normalize(T* output, const T* row_sum, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    output[idx] /= row_sum[idx / cols];
}

}}} // namespace TensorN::cuda::kernels

namespace TensorN { namespace cuda {

using namespace kernels;

template <typename T>
void softmax_2d_axis1(const CudaTensor<T>& A, CudaTensor<T>& C) {
    size_t rows = A.shape()[0], cols = A.shape()[1];
    T* d_row_max; T* d_row_sum;
    cudaMalloc(&d_row_max, rows * sizeof(T));
    cudaMalloc(&d_row_sum, rows * sizeof(T));
    cudaMemset(d_row_sum, 0, rows * sizeof(T));
    CudaTensor<T> temp({rows, cols});

    softmax_max<<<rows, 1>>>(A.device_ptr(), d_row_max, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());
    softmax_exp_sum<<<rows, cols>>>(A.device_ptr(), d_row_max, temp.device_ptr(), d_row_sum, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());

    size_t total = rows * cols;
    size_t bs = 256, gs = (total + bs - 1) / bs;
    softmax_normalize<<<gs, bs>>>(temp.device_ptr(), d_row_sum, rows, cols);
    CHECK_CUDA_ERROR(cudaGetLastError());

    C = std::move(temp);
    cudaFree(d_row_max); cudaFree(d_row_sum);
}

template <typename T>
void softmax(const CudaTensor<T>& A, CudaTensor<T>& C, int axis) {
    if (A.shape().empty()) throw std::invalid_argument("Softmax requires non-empty tensor");
    size_t ndim = A.shape().size();
    if (axis < 0) axis = static_cast<int>(ndim) + axis;
    if (axis < 0 || static_cast<size_t>(axis) >= ndim)
        throw std::invalid_argument("Softmax axis out of range");
    if (A.shape().size() > 2) throw std::runtime_error("Softmax: only 1D/2D currently");

    if (A.shape().size() == 1) {
        C = CudaTensor<T>(A.shape());
        size_t n = A.size();
        T *d_max, *d_sum;
        cudaMalloc(&d_max, sizeof(T)); cudaMalloc(&d_sum, sizeof(T));
        cudaMemset(d_sum, 0, sizeof(T));
        CudaTensor<T> temp(A.shape());
        softmax_max<<<1,1>>>(A.device_ptr(), d_max, 1, n);
        softmax_exp_sum<<<1,n>>>(A.device_ptr(), d_max, temp.device_ptr(), d_sum, 1, n);
        size_t bs=256, gs=(n+bs-1)/bs;
        softmax_normalize<<<gs,bs>>>(temp.device_ptr(), d_sum, 1, n);
        C = std::move(temp); cudaFree(d_max); cudaFree(d_sum);
        return;
    }
    if (axis == 1) { softmax_2d_axis1(A, C); return; }
    if (axis == 0 && A.shape().size() == 2) {
        size_t rows = A.shape()[0], cols = A.shape()[1];
        CudaTensor<T> At({cols, rows}), Ct({cols, rows});
        size_t total = rows * cols, bs = 256, gs = (total + bs - 1) / bs;
        transpose<<<gs,bs>>>(A.device_ptr(), At.device_ptr(), rows, cols);
        softmax_2d_axis1(At, Ct);
        transpose_back<<<gs,bs>>>(Ct.device_ptr(), C.device_ptr(), cols, rows);
        return;
    }
    throw std::runtime_error("Softmax for this axis configuration not implemented");
}

// -- launch helpers --
template <typename T>
void launch_unary(void(*k)(const T*,T*,size_t), const CudaTensor<T>& A, CudaTensor<T>& C) {
    if (A.size() != C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(), bs=256, gs=(n+bs-1)/bs;
    k<<<gs,bs>>>(A.device_ptr(),C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template <typename T>
void launch_binary(void(*k)(const T*,const T*,T*,size_t), const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C) {
    if (A.size()!=B.size()||A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(), bs=256, gs=(n+bs-1)/bs;
    k<<<gs,bs>>>(A.device_ptr(),B.device_ptr(),C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template <typename T>
void launch_binary_int(void(*k)(const T*,const T*,int*,size_t), const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C) {
    if (A.size()!=B.size()||A.size()!=static_cast<size_t>(C.size())) throw std::invalid_argument("size mismatch");
    size_t n=A.size(), bs=256, gs=(n+bs-1)/bs;
    k<<<gs,bs>>>(A.device_ptr(),B.device_ptr(),C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// -- impl --
#define IMPL_UNARY(NAME) template<typename T> void NAME(const CudaTensor<T>& A, CudaTensor<T>& C) { launch_unary(kernels::NAME<T>,A,C); }
#define IMPL_BINARY(NAME) template<typename T> void NAME(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<T>& C) { launch_binary(kernels::NAME<T>,A,B,C); }
#define IMPL_BINARY_INT(NAME) template<typename T> void NAME(const CudaTensor<T>& A, const CudaTensor<T>& B, CudaTensor<int>& C) { launch_binary_int(kernels::NAME<T>,A,B,C); }

IMPL_BINARY(add)
IMPL_BINARY(subtract)
IMPL_BINARY(multiply)
IMPL_BINARY(divide)
IMPL_UNARY(negate)
IMPL_UNARY(abs)
IMPL_UNARY(sqrt)
IMPL_UNARY(exp)
IMPL_UNARY(log)
IMPL_UNARY(sin)
IMPL_UNARY(cos)
IMPL_UNARY(relu)
IMPL_UNARY(gelu)
IMPL_UNARY(sigmoid)
IMPL_UNARY(tanh)
IMPL_BINARY_INT(equal)
IMPL_BINARY_INT(not_equal)
IMPL_BINARY_INT(greater)
IMPL_BINARY_INT(less)
IMPL_BINARY_INT(greater_equal)
IMPL_BINARY_INT(less_equal)

template<typename T> void add_scalar(const CudaTensor<T>& A, T s, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::add_scalar<<<gs,bs>>>(A.device_ptr(),s,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template<typename T> void multiply_scalar(const CudaTensor<T>& A, T s, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::multiply_scalar<<<gs,bs>>>(A.device_ptr(),s,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template<typename T> void subtract_scalar(const CudaTensor<T>& A, T s, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::subtract_scalar<<<gs,bs>>>(A.device_ptr(),s,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template<typename T> void divide_scalar(const CudaTensor<T>& A, T s, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::divide_scalar<<<gs,bs>>>(A.device_ptr(),s,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template<typename T> void pow(const CudaTensor<T>& A, T exp, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::pow<<<gs,bs>>>(A.device_ptr(),exp,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template<typename T> void clip(const CudaTensor<T>& A, T lo, T hi, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::clip<<<gs,bs>>>(A.device_ptr(),lo,hi,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template<typename T> void leaky_relu(const CudaTensor<T>& A, T alpha, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::leaky_relu<<<gs,bs>>>(A.device_ptr(),alpha,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
template<typename T> void elu(const CudaTensor<T>& A, T alpha, CudaTensor<T>& C) {
    if(A.size()!=C.size()) throw std::invalid_argument("size mismatch");
    size_t n=A.size(),bs=256,gs=(n+bs-1)/bs;
    kernels::elu<<<gs,bs>>>(A.device_ptr(),alpha,C.device_ptr(),n);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

#define INST(T) \
    template void add<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<T>&); \
    template void subtract<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<T>&); \
    template void multiply<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<T>&); \
    template void divide<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<T>&); \
    template void add_scalar<T>(const CudaTensor<T>&,T,CudaTensor<T>&); \
    template void multiply_scalar<T>(const CudaTensor<T>&,T,CudaTensor<T>&); \
    template void subtract_scalar<T>(const CudaTensor<T>&,T,CudaTensor<T>&); \
    template void divide_scalar<T>(const CudaTensor<T>&,T,CudaTensor<T>&); \
    template void negate<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void abs<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void sqrt<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void exp<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void log<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void sin<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void cos<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void pow<T>(const CudaTensor<T>&,T,CudaTensor<T>&); \
    template void clip<T>(const CudaTensor<T>&,T,T,CudaTensor<T>&); \
    template void relu<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void leaky_relu<T>(const CudaTensor<T>&,T,CudaTensor<T>&); \
    template void elu<T>(const CudaTensor<T>&,T,CudaTensor<T>&); \
    template void gelu<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void sigmoid<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void tanh<T>(const CudaTensor<T>&,CudaTensor<T>&); \
    template void softmax<T>(const CudaTensor<T>&,CudaTensor<T>&,int); \
    template void equal<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<int>&); \
    template void not_equal<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<int>&); \
    template void greater<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<int>&); \
    template void less<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<int>&); \
    template void greater_equal<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<int>&); \
    template void less_equal<T>(const CudaTensor<T>&,const CudaTensor<T>&,CudaTensor<int>&);

INST(float)
INST(double)

}} // namespace TensorN::cuda
