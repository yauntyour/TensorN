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
├── BLAS/              OpenBLAS accelerated backend
│   └── blas_tensor.hpp
└── CUDA/              CUDA/cuBLAS accelerated backend
    ├── cuda_tensor.hpp   CudaTensor<T> (device memory management)
    ├── matmul.cu         Matrix multiplication (cuBLAS)
    ├── elementwise.cu    Element-wise & activation kernels
    ├── reduction.cu      Reduction kernels (sum, mean, max, ...)
    └── convolution.cu    Conv2d / ConvTranspose2d kernels
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

## 📊 Benchmark

Build and run the benchmark to compare backend performance on your hardware:

```bash
cmake --build build --config Release --target TensorN_Benchmark
./build/bin/benchmarks/Release/TensorN_Benchmark.exe
```

Benchmark covers: matrix multiplication, element-wise operations, activations, reductions, convolution, and comparison operations across all three backends.

### Sample Results

> **Test environment:** NVIDIA GeForce RTX 5060 Ti (SM 12.0)
> **Matmul:** 64×64 | **Element-wise:** 4096 elements | **Vector:** 1024 | **Warmup:** 2 | **Repeats:** 5

#### Linear Algebra (Matrix 64×64)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| matmul | 17.976 | 0.144 | 0.019 | 124.9× | 930.0× |
| gram (X·Xᵀ) | 17.767 | 0.113 | 0.022 | 156.8× | 803.7× |
| dot (vec 1024) | 0.065 | <0.001 | 0.064 | 147.9× | 1.0× |
| outer (256) | 4.706 | 0.019 | 0.030 | 252.5× | 154.9× |
| bilinear (xᵀAy) | 0.301 | 0.001 | 0.079 | 301.3× | 3.8× |
| axpy (4096) | 0.002 | 0.001 | 0.011 | 1.4× | 0.2× |
| trace | 0.004 | 0.002 | 0.054 | 1.9× | 0.1× |

#### Element-wise (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| add (A+B) | 0.002 | 0.002 | 0.008 | 0.8× | 0.2× |
| hadamard (A*B) | 0.404 | 0.001 | 0.005 | 311.0× | 84.8× |
| scalar_mul (A×3.14) | 0.001 | <0.001 | 0.006 | 1.6× | 0.3× |
| exp | 0.009 | 0.009 | 0.006 | 1.0× | 1.6× |
| log | 0.012 | 0.013 | 0.011 | 1.0× | 1.1× |
| sqrt | 0.001 | 0.004 | 0.011 | 0.4× | 0.1× |
| sin | 0.013 | 0.013 | 0.005 | 1.0× | 2.5× |
| cos | 0.013 | 0.013 | 0.006 | 1.0× | 2.4× |
| pow (x²) | 0.027 | 0.026 | 0.006 | 1.0× | 4.8× |
| abs | <0.001 | 0.001 | 0.009 | 0.4× | 0.1× |

#### Activation Functions (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| relu | 0.002 | 0.002 | 0.006 | 1.4× | 0.3× |
| sigmoid | 0.010 | 0.010 | 0.009 | 1.0× | 1.2× |
| tanh | 0.018 | 0.018 | 0.010 | 1.0× | 1.7× |
| gelu | 0.020 | 0.020 | 0.006 | 1.0× | 3.4× |
| softmax (axis=1) | 0.465 | 0.425 | 0.033 | 1.1× | 13.9× |

#### Reductions (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| sum | 0.094 | <0.001 | 0.047 | 4682.0× | 2.0× |
| mean | 0.094 | N/A | 0.042 | — | 2.2× |
| max | <0.001 | <0.001 | 0.054 | 1.1× | 0.0× |
| min | <0.001 | <0.001 | 0.063 | 1.0× | 0.0× |
| L2 norm (vec 1024) | 0.068 | <0.001 | 0.068 | 91.9× | 1.0× |
| frobenius_norm | 0.611 | 0.003 | 0.153 | 210.7× | 4.0× |
| variance | 0.604 | N/A | 0.100 | — | 6.0× |
| stddev | 0.457 | <0.001 | 0.085 | 22867.0× | 5.4× |
| argmax (axis=1) | 0.001 | 0.001 | 0.012 | 1.0× | 0.1× |
| argmin (axis=1) | 0.001 | 0.001 | 0.013 | 1.0× | 0.1× |

#### Transpose (64×64)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| transpose | 0.217 | 0.231 | 0.012 | 0.9× | 18.5× |

#### Conv2d (input: 1×3×32×32, kernel: 16×3×3×3, stride=1, pad=1)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| conv2d | 20.329 | 19.764 | 0.009 | 1.0× | 2183.1× |
| conv_transpose2d | 21.714 | 28.664 | 0.016 | 0.8× | 1382.0× |

#### Comparison (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| greater (A>B) | 0.002 | 0.002 | 0.005 | 1.0× | 0.4× |
| equal (A==B) | 0.003 | 0.003 | 0.005 | 1.0× | 0.6× |

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
