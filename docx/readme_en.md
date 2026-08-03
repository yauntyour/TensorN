<p align="center">
  <h1 align="center">TensorN</h1>
  <p align="center">
    <em>A C++17 header-only tensor library · OpenBLAS & CUDA/cuBLAS accelerated</em>
  </p>
  <p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/c%2B%2B-17-00599C.svg" alt="C++17">
    <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="Version 1.0.0">
    <img src="https://img.shields.io/badge/header--only-✔-brightgreen.svg" alt="Header-only">
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> ·
    <a href="#-build">Build</a> ·
    <a href="#-architecture">Architecture</a> ·
    <a href="#-operations">Operations</a> ·
    <a href="#-in-place-operations">In-place</a> ·
    <a href="#-zero-copy-views--memory-pool">Zero-Copy</a> ·
    <a href="#cuda-streams--async">CUDA Streams</a> ·
    <a href="#-fused-kernels">Fused Kernels</a> ·
    <a href="#-benchmark">Benchmark</a> ·
    <a href="#-dependencies">Dependencies</a>
  </p>
  <p align="center">
    English | <a href="../readme.md">中文</a>
  </p>
</p>

---

## ✨ Features

- **Header-only** — single `#include "TensorN.hpp"` to use
- **Three acceleration backends** — Native C++, OpenBLAS, CUDA/cuBLAS
- **Einstein summation** — `einsum("ij,jk->ik", A, B)` for flexible tensor operations
- **Rich operation set** — linear algebra, element-wise math, activations, reductions, convolution
- **Data I/O** — CSV, NumPy `.npy`/`.npz`, JSON, PyTorch `.pt` formats, with TensorN↔PyTorch bridge tool
- **OpenCV interop** — optional `cv::Mat` conversion
- **In-place operations** — `add_()`, `sub_()`, `mul_()`, `div_()`, `apply_()`, `fill_()`, `zero_()` for zero-allocation transforms
- **Zero-copy views** — `view()`, `reshape()` share underlying data, no copy
- **CUDA streams & async** — stream-aware cuBLAS, async transfers, memory pools, and fused kernels
- **OpenBLAS multi-core** — OpenMP parallelism across all non-BLAS loops, im2col+GEMM convolution

---

## 🚀 Quick Start

```cpp
#include "TensorN.hpp"
using namespace TensorN;

int main()
{
    // Create tensors
    Tensor<float> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> B({3, 2}, {7, 8, 9, 10, 11, 12});

    // Matrix multiplication
    auto C = matmul(A, B);

    // Einstein summation
    auto D = einsum<float>("ij,jk->ik", A, B);

    // Factory functions
    auto Z = zeros<float>({3, 3});
    auto I = eye<double>(4);
    auto R = arange(0.0f, 10.0f, 0.5f);

    // Save / Load
    C.tensor.save("result.npy");
    auto loaded = load<float>("result.npy");
}
```

---

## 🛠 Build

Requires **CMake 3.18+**, **C++17 compiler**. CUDA and OpenBLAS are optional.

```bash
cmake -B build -DTENSORN_ENABLE_CUDA=ON -DTENSORN_ENABLE_OPENBLAS=ON
cmake --build build --config Release
```

| Option | Default | Description |
|---|---|---|
| `TENSORN_ENABLE_CUDA` | ON | Enable CUDA/cuBLAS backend |
| `TENSORN_ENABLE_OPENBLAS` | ON | Enable OpenBLAS backend |
| `TENSORN_BUILD_EXAMPLES` | ON | Build example programs |
| `TENSORN_BUILD_BENCHMARKS` | ON | Build benchmark programs |

---

## 🏗 Architecture

```
TensorN
├── Tensor<T>          Core tensor class (N-dimensional, row-major)
├── opt<T>             Lazy evaluation wrapper for chained operations
├── einsum()           Einstein summation engine
├── operations.hpp     High-level ops (matmul, dot, outer, gram, ...)
├── static.hpp         Data I/O (csv, npy, npz, json, pt)
├── memory_pool.hpp    CPU memory pool (bucket allocator, PooledAllocator, PooledVector)
├── BLAS/              OpenBLAS accelerated backend (OpenMP multi-core, im2col+GEMM conv)
│   └── blas_tensor.hpp
└── CUDA/              CUDA/cuBLAS accelerated backend
    ├── cuda_tensor.hpp    CudaTensor<T> (device memory, async transfers, zero-copy views)
    ├── cuda_stream.hpp    CudaStream, CudaEvent, stream pool, device/pinned memory pools
    ├── fused_kernels.hpp  Fused kernels (matmul+activation, conv+activation, add_relu, etc.)
    ├── matmul.cu          Matrix multiplication (cuBLAS, stream-aware)
    ├── elementwise.cu     Element-wise & activation kernels
    ├── reduction.cu       Reduction kernels (sum, mean, max, ...)
    └── convolution.cu     Conv2d / ConvTranspose2d kernels
```

### Backends

| Backend | Namespace | Description |
|---|---|---|
| Native C++ | `TensorN::` | einsum-based, no external dependencies |
| OpenBLAS | `TensorN::blas::` | Uses cblas_sgemm/cblas_dgemm |
| cuBLAS | `TensorN::cuda::` | Uses cublasSgemm/cublasDgemm + custom CUDA kernels |

> All three backends share the same API pattern — pass `Tensor<T>` for native/OpenBLAS, `CudaTensor<T>` for CUDA.

---

## 🔧 Operations

### Linear Algebra

| Operation | Native | OpenBLAS | cuBLAS |
|---|---|---|---|
| `matmul(A, B)` | `einsum` | `cblas_sgemm` | `cublasSgemm` |
| `dot(v1, v2)` | `einsum` | `cblas_sdot` | `cublasSdot` |
| `outer(a, b)` | `einsum` | `cblas_sger` | custom kernel |
| `gram(X)` | `einsum` | `cblas_sgemm(T)` | `cublasSgemm(T)` |
| `bilinear(x, A, y)` | native | `cblas_sgemv` | `cublasSgemv` |
| `batched_matmul(A, B)` | `einsum` | loop+sgemm | `cublasSgemmStridedBatched` |
| `trace(A)` | `einsum` | manual loop | custom kernel |
| `transpose(A)` | `einsum` | manual loop | custom kernel |

### Element-wise

`add`, `subtract`, `multiply`, `divide`, `scalar ops`, `exp`, `log`, `sqrt`, `sin`, `cos`, `pow`, `abs`, `clip`, `negate`

### Activations

`relu`, `leaky_relu`, `elu`, `gelu`, `sigmoid`, `tanh`, `softmax`

### Reductions

`sum`, `mean`, `max`, `min`, `norm`, `frobenius_norm`, `var`, `stddev`, `argmax`, `argmin`

### Convolution

`conv2d`, `conv_transpose2d` (with stride and padding)

### Other

`hadamard` (element-wise multiply), `equal`, `greater`, `contract`, `diag`, `diag_matrix`

### Data I/O

```cpp
tensor.save("data.csv");   // CSV (1D/2D only)
tensor.save("data.npy");   // NumPy format
tensor.save("data.npz");   // NumPy compressed
tensor.save("data.json");  // JSON with shape + data
tensor.save("data.pt");    // TensorN .pt binary format (also .pth)

auto t = load<float>("data.pt");  // auto-detect by extension
```

**Supported types:** `float`, `double`, `int32_t`, `int64_t`, `uint8_t`, `int16_t`

**PyTorch interop:** use `tools/pt_converter.py` to convert between TensorN `.pt` and PyTorch `.pth`:

```bash
# PyTorch .pth → TensorN .pt
python tools/pt_converter.py torch2pt model.pth data.pt

# TensorN .pt → PyTorch .pth
python tools/pt_converter.py pt2torch data.pt model.pth

# also supports .npy as intermediate
python tools/pt_converter.py np2pt data.npy data.pt
python tools/pt_converter.py pt2np data.pt data.npy
```

---

## ⚡ In-place Operations

Zero-allocation in-place transforms on both `Tensor` and `CudaTensor`:

```cpp
Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
t.add_(2.0f);          // add 2 to every element
t.mul_(0.5f);          // multiply every element by 0.5
t.apply_([](float x) { return x * x; });  // custom element-wise transform
t.zero_();             // fill with zeros
```

## 🔄 Zero-Copy Views & Memory Pool

- **`view(shape)` / `reshape(shape)`** — returns a new tensor sharing underlying data, no allocation
- **`memory_pool.hpp`** — CPU bucket allocator providing `PooledAllocator<T>` and `PooledVector<T>`
- **`from_pool(shape, pool)`** — allocate a tensor from a memory pool

## 🌊 CUDA Streams & Async

All CUDA operations provide `cudaStream_t` overloads for efficient pipelining with stream pools and async memory pools:

```cpp
auto stream = CudaStreamPool::acquire();
auto a_dev = CudaTensor<float>::fromPinned(a_host, stream);
auto b_dev = CudaTensor<float>::fromPinned(b_host, stream);
auto c_dev = matmul(a_dev, b_dev, stream);        // stream-aware cuBLAS
c_dev.copyToHostAsync(result, stream);             // async transfer back
stream.sync();
```

- **`CudaStreamPool`** — pre-created CUDA stream reuse
- **`CudaMemoryPool` / `PinnedMemoryPool`** — device and pinned host memory pools
- **`copyFromHostAsync()` / `copyToHostAsync()` / `copyFromDeviceAsync()`** — async data transfers
- **`memset_zero_async()`** — async zero initialization
- **`view()` / `reshape()`** — device-side zero-copy views

## 🔥 Fused Kernels

Eliminate intermediate buffers by combining operations in a single kernel:

| Fused Op | Description |
|---|---|
| `fused_matmul_relu(A, B)` | Matrix multiply + ReLU activation |
| `fused_conv_relu(input, kernel, bias)` | Conv2d + Bias + ReLU |
| `fused_add_relu(A, B)` | Element-wise add + ReLU |
| `fused_mul_add(A, B, C)` | Element-wise multiply + add |
| `fused_batchnorm_inference(x, gamma, beta, mean, var)` | Inference batchnorm |
| `fused_residual_block(x, w1, w2, ...)` | Residual block (MLP/conv) |

---

## 📊 Benchmark

Build and run the benchmark to compare backend performance on your hardware:

```bash
cmake --build build --config Release --target TensorN_Benchmark
./build/bin/benchmarks/Release/TensorN_Benchmark.exe
```

Benchmark covers: matrix multiplication, element-wise operations, activations, reductions, convolution, and comparison operations across all three backends.

### Sample Results

> **Test environment:** NVIDIA GeForce RTX 5060 Ti (SM 12.0)
> **Matmul:** 512×512 | **Element-wise:** 65536 elements | **Vector:** 4096 | **Warmup:** 2 | **Repeats:** 5

#### Linear Algebra (Matrix 512×512)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| matmul | 9101.528 | 64.036 | 0.033 | 142.1× | 279229.1× |
| gram (X·Xᵀ) | 9561.571 | 64.693 | 0.037 | 147.8× | 260459.5× |
| dot (vec 4096) | 0.177 | 0.002 | 0.084 | 105.3× | 2.1× |
| outer (1024) | 70.760 | 0.884 | 0.017 | 80.1× | 4282.1× |
| bilinear (xᵀAy) | 16.983 | 0.112 | 0.054 | 151.2× | 315.2× |
| axpy (65536) | 0.033 | 0.018 | 0.010 | 1.8× | 3.3× |
| trace | 0.012 | 0.010 | 0.050 | 1.1× | 0.2× |

#### Element-wise (65536 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| add (A+B) | 0.034 | 0.014 | 0.006 | 2.5× | 5.8× |
| hadamard (A*B) | 7.185 | 0.015 | 0.006 | 490.8× | 1131.8× |
| scalar_mul (A×3.14) | 0.037 | 0.014 | 0.006 | 2.7× | 6.7× |
| exp | 0.138 | 0.071 | 0.006 | 1.9× | 22.9× |
| log | 0.204 | 1.280 | 0.006 | 0.2× | 34.6× |
| sqrt | 0.018 | 0.447 | 0.006 | 0.0× | 3.2× |
| sin | 0.255 | 0.368 | 0.010 | 0.7× | 24.7× |
| cos | 0.256 | 3.761 | 0.007 | 0.1× | 39.3× |
| pow (x²) | 0.437 | 0.855 | 0.010 | 0.5× | 44.8× |
| abs | 0.008 | 0.032 | 0.006 | 0.3× | 1.4× |

#### Activation Functions (65536 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| relu | 0.035 | 0.179 | 0.009 | 0.2× | 3.8× |
| sigmoid | 0.180 | 1.361 | 0.014 | 0.1× | 12.4× |
| tanh | 0.321 | 0.387 | 0.009 | 0.8× | 36.3× |
| gelu | 0.369 | 0.509 | 0.006 | 0.7× | 64.6× |
| softmax (axis=1) | 11.981 | 0.049 | 0.048 | 242.1× | 251.7× |

#### Reductions (65536 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| sum | 2.554 | 0.012 | 0.064 | 212.8× | 40.0× |
| mean | 2.541 | 0.011 | 0.055 | 228.5× | 46.0× |
| max | 0.004 | 0.004 | 0.045 | 1.0× | 0.1× |
| min | 0.004 | 0.004 | 0.045 | 1.0× | 0.1× |
| L2 norm (vec 4096) | 0.302 | 0.003 | 0.044 | 92.0× | 6.9× |
| frobenius_norm | 9.765 | 0.051 | 0.054 | 192.8× | 181.6× |
| variance | 7.879 | 0.020 | 0.098 | 385.9× | 80.2× |
| stddev | 12.446 | 0.026 | 0.096 | 473.9× | 130.2× |
| argmax (axis=1) | 0.034 | 0.054 | 0.035 | 0.6× | 1.0× |
| argmin (axis=1) | 0.038 | 0.018 | 0.035 | 2.1× | 1.1× |

#### Transpose (512×512)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| transpose | 17.296 | 0.375 | 0.011 | 46.1× | 1515.7× |

#### Conv2d (input: 1×3×64×64, kernel: 32×3×3×3, stride=1, pad=1)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| conv2d | 172.840 | 1.679 | 0.032 | 103.0× | 5343.5× |
| conv_transpose2d | 180.763 | 5.396 | 0.060 | 33.5× | 3038.0× |

#### Comparison (65536 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| greater (A>B) | 0.066 | 0.023 | 0.006 | 2.8× | 10.7× |
| equal (A==B) | 0.068 | 0.023 | 0.005 | 3.0× | 12.4× |

> Speedup columns show how many times faster the backend is vs Native C++.

---

## 📦 Dependencies

| Library | Required | Purpose |
|---|---|---|
| C++17 compiler | ✅ | Core language features |
| nlohmann/json | 🔽 Auto-fetched | JSON serialization |
| zlib | 🔽 Auto-fetched | npz compression (via cnpy) |
| OpenBLAS | ⬜ Optional | CPU BLAS acceleration |
| CUDA Toolkit | ⬜ Optional | GPU acceleration |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](../LICENSE) file for details.
