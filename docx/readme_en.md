# TensorN

English | [中文](../readme.md)

A C++17 header-only tensor library with **OpenBLAS** and **CUDA/cuBLAS** backend support. Provides a Torch-like API with Einstein summation, multi-backend acceleration, and seamless data serialization.

## Features

- **Header-only** - single `#include "TensorN.hpp"` to use
- **Three acceleration backends** - Native C++, OpenBLAS, CUDA/cuBLAS
- **Einstein summation** - `einsum("ij,jk->ik", A, B)` for flexible tensor operations
- **Rich operation set** - linear algebra, element-wise math, activations, reductions, convolution
- **Data I/O** - CSV, NumPy `.npy`/`.npz`, JSON, PyTorch `.pt` formats, with TensorN↔PyTorch bridge tool
- **OpenCV interop** - optional `cv::Mat` conversion

## Quick Start

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

## Build

Requires CMake 3.18+, C++17 compiler. CUDA and OpenBLAS are optional.

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

## Architecture

```
TensorN
├── Tensor<T>           Core tensor class (N-dimensional, row-major)
├── opt<T>              Lazy evaluation wrapper for chained operations
├── einsum()            Einstein summation engine
├── operations.hpp      High-level ops (matmul, dot, outer, gram, ...)
├── static.hpp          Data I/O (csv, npy, npz, json, pt)
├── BLAS/               OpenBLAS accelerated backend
│   └── blas_tensor.hpp
└── CUDA/               CUDA/cuBLAS accelerated backend
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
| cuBLAS | `TensorN::cuda::` | Uses cublasSgemm/cblasDgemm + custom CUDA kernels |

All three backends share the same API pattern - pass `Tensor<T>` for native/OpenBLAS, `CudaTensor<T>` for CUDA.

## Operations

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

### Reduction

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

## Benchmark

Build and run the benchmark to compare backend performance on your hardware:

```bash
cmake --build build --config Release --target TensorN_Benchmark
./build/bin/benchmarks/Release/TensorN_Benchmark.exe
```

Benchmark covers: matrix multiplication, element-wise operations, activations, reductions, convolution, and comparison operations across all three backends.

### Sample Results

> Test environment: NVIDIA GeForce RTX 5060 Ti (SM 12.0)
> Matmul: 64x64 | Element-wise: 4096 elements | Vector: 1024 | Warmup: 2 | Repeats: 5

#### Linear Algebra (Matrix 64x64)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| matmul | 17.976 | 0.144 | 0.019 | 124.9x | 930.0x |
| gram (X*X^T) | 17.767 | 0.113 | 0.022 | 156.8x | 803.7x |
| dot (vec 1024) | 0.065 | <0.001 | 0.064 | 147.9x | 1.0x |
| outer (256) | 4.706 | 0.019 | 0.030 | 252.5x | 154.9x |
| bilinear (x^T A y) | 0.301 | 0.001 | 0.079 | 301.3x | 3.8x |
| axpy (4096) | 0.002 | 0.001 | 0.011 | 1.4x | 0.2x |
| trace | 0.004 | 0.002 | 0.054 | 1.9x | 0.1x |

#### Element-wise (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| add (A+B) | 0.002 | 0.002 | 0.008 | 0.8x | 0.2x |
| hadamard (A*B) | 0.404 | 0.001 | 0.005 | 311.0x | 84.8x |
| scalar_mul (A*3.14) | 0.001 | <0.001 | 0.006 | 1.6x | 0.3x |
| exp | 0.009 | 0.009 | 0.006 | 1.0x | 1.6x |
| log | 0.012 | 0.013 | 0.011 | 1.0x | 1.1x |
| sqrt | 0.001 | 0.004 | 0.011 | 0.4x | 0.1x |
| sin | 0.013 | 0.013 | 0.005 | 1.0x | 2.5x |
| cos | 0.013 | 0.013 | 0.006 | 1.0x | 2.4x |
| pow (x^2) | 0.027 | 0.026 | 0.006 | 1.0x | 4.8x |
| abs | <0.001 | 0.001 | 0.009 | 0.4x | 0.1x |

#### Activation Functions (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| relu | 0.002 | 0.002 | 0.006 | 1.4x | 0.3x |
| sigmoid | 0.010 | 0.010 | 0.009 | 1.0x | 1.2x |
| tanh | 0.018 | 0.018 | 0.010 | 1.0x | 1.7x |
| gelu | 0.020 | 0.020 | 0.006 | 1.0x | 3.4x |
| softmax (axis=1) | 0.465 | 0.425 | 0.033 | 1.1x | 13.9x |

#### Reduction (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| sum | 0.094 | <0.001 | 0.047 | 4682.0x | 2.0x |
| mean | 0.094 | N/A | 0.042 | - | 2.2x |
| max | <0.001 | <0.001 | 0.054 | 1.1x | 0.0x |
| min | <0.001 | <0.001 | 0.063 | 1.0x | 0.0x |
| L2 norm (vec 1024) | 0.068 | <0.001 | 0.068 | 91.9x | 1.0x |
| frobenius_norm | 0.611 | 0.003 | 0.153 | 210.7x | 4.0x |
| variance | 0.604 | N/A | 0.100 | - | 6.0x |
| stddev | 0.457 | <0.001 | 0.085 | 22867.0x | 5.4x |
| argmax (axis=1) | 0.001 | 0.001 | 0.012 | 1.0x | 0.1x |
| argmin (axis=1) | 0.001 | 0.001 | 0.013 | 1.0x | 0.1x |

#### Transpose (64x64)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| transpose | 0.217 | 0.231 | 0.012 | 0.9x | 18.5x |

#### Conv2d (input: 1x3x32x32, kernel: 16x3x3x3, stride=1, pad=1)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| conv2d | 20.329 | 19.764 | 0.009 | 1.0x | 2183.1x |
| conv_transpose2d | 21.714 | 28.664 | 0.016 | 0.8x | 1382.0x |

#### Comparison (4096 elements)

| Operation | Native(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/Native | CUDA/Native |
|---|---|---|---|---|---|
| greater (A>B) | 0.002 | 0.002 | 0.005 | 1.0x | 0.4x |
| equal (A==B) | 0.003 | 0.003 | 0.005 | 1.0x | 0.6x |

> Speedup columns show how many times faster the backend is vs Native C++.

## Dependencies

| Library | Required | Purpose |
|---|---|---|
| C++17 compiler | Yes | Core language features |
| nlohmann/json | Auto-fetched | JSON serialization |
| zlib | Auto-fetched | npz compression (via cnpy) |
| OpenBLAS | Optional | CPU BLAS acceleration |
| CUDA Toolkit | Optional | GPU acceleration |

## License

See [LICENSE](LICENSE).
