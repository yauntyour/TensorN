# TensorN

A C++17 header-only tensor library with **OpenBLAS** and **CUDA/cuBLAS** backend support. Provides a NumPy-like API with Einstein summation, multi-backend acceleration, and seamless data serialization.

## Features

- **Header-only** - single `#include "TensorN.hpp"` to use
- **Three acceleration backends** - Native C++, OpenBLAS, CUDA/cuBLAS
- **Einstein summation** - `einsum("ij,jk->ik", A, B)` for flexible tensor operations
- **Rich operation set** - linear algebra, element-wise math, activations, reductions, convolution
- **Data I/O** - CSV, NumPy `.npy`/`.npz`, JSON formats
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
├── static.hpp          Data I/O (csv, npy, npz, json)
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

auto t = load<float>("data.npy");  // auto-detect by extension
```

## Benchmark

Build and run the benchmark to compare backend performance on your hardware:

```bash
cmake --build build --config Release --target TensorN_Benchmark
./build/bin/benchmarks/Release/TensorN_Benchmark.exe
```

Benchmark covers: matrix multiplication, element-wise operations, activations, reductions, convolution, and comparison operations across all three backends.

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
