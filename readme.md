# TensorN

[English](docx/readme_en.md) | 中文

一个 C++17 纯头文件张量库，支持 **OpenBLAS** 和 **CUDA/cuBLAS** 后端。提供类似 Torch 的 API，支持爱因斯坦求和、多后端加速以及无缝数据序列化。

## 特性

- **纯头文件** - 仅需 `#include "TensorN.hpp"` 即可使用
- **三种加速后端** - 原生 C++、OpenBLAS、CUDA/cuBLAS
- **爱因斯坦求和** - `einsum("ij,jk->ik", A, B)` 实现灵活的张量运算
- **丰富的运算集** - 线性代数、逐元素数学运算、激活函数、规约、卷积
- **数据 I/O** - CSV、NumPy `.npy`/`.npz`、JSON 格式
- **OpenCV 互操作** - 可选的 `cv::Mat` 转换

## 快速开始

```cpp
#include "TensorN.hpp"
using namespace TensorN;

int main()
{
    // 创建张量
    Tensor<float> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> B({3, 2}, {7, 8, 9, 10, 11, 12});

    // 矩阵乘法
    auto C = matmul(A, B);

    // 爱因斯坦求和
    auto D = einsum<float>("ij,jk->ik", A, B);

    // 工厂函数
    auto Z = zeros<float>({3, 3});
    auto I = eye<double>(4);
    auto R = arange(0.0f, 10.0f, 0.5f);

    // 保存 / 加载
    C.tensor.save("result.npy");
    auto loaded = load<float>("result.npy");
}
```

## 构建

需要 CMake 3.18+、C++17 编译器。CUDA 和 OpenBLAS 为可选依赖。

```bash
cmake -B build -DTENSORN_ENABLE_CUDA=ON -DTENSORN_ENABLE_OPENBLAS=ON
cmake --build build --config Release
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `TENSORN_ENABLE_CUDA` | ON | 启用 CUDA/cuBLAS 后端 |
| `TENSORN_ENABLE_OPENBLAS` | ON | 启用 OpenBLAS 后端 |
| `TENSORN_BUILD_EXAMPLES` | ON | 构建示例程序 |
| `TENSORN_BUILD_BENCHMARKS` | ON | 构建基准测试程序 |

## 架构

```
TensorN
├── Tensor<T>           核心张量类（N 维，行主序）
├── opt<T>              链式操作的惰性求值包装器
├── einsum()            爱因斯坦求和引擎
├── operations.hpp      高级运算（matmul, dot, outer, gram, ...）
├── static.hpp          数据 I/O（csv, npy, npz, json）
├── BLAS/               OpenBLAS 加速后端
│   └── blas_tensor.hpp
└── CUDA/               CUDA/cuBLAS 加速后端
    ├── cuda_tensor.hpp   CudaTensor<T>（设备内存管理）
    ├── matmul.cu         矩阵乘法（cuBLAS）
    ├── elementwise.cu    逐元素运算与激活函数内核
    ├── reduction.cu      规约内核（sum, mean, max, ...）
    └── convolution.cu    Conv2d / ConvTranspose2d 内核
```

### 后端

| 后端 | 命名空间 | 说明 |
|---|---|---|
| 原生 C++ | `TensorN::` | 基于 einsum，无外部依赖 |
| OpenBLAS | `TensorN::blas::` | 使用 cblas_sgemm/cblas_dgemm |
| cuBLAS | `TensorN::cuda::` | 使用 cublasSgemm/cublasDgemm + 自定义 CUDA 内核 |

三个后端共享相同的 API 模式——原生/OpenBLAS 传入 `Tensor<T>`，CUDA 传入 `CudaTensor<T>`。

## 运算

### 线性代数

| 运算 | 原生 | OpenBLAS | cuBLAS |
|---|---|---|---|
| `matmul(A, B)` | `einsum` | `cblas_sgemm` | `cublasSgemm` |
| `dot(v1, v2)` | `einsum` | `cblas_sdot` | `cublasSdot` |
| `outer(a, b)` | `einsum` | `cblas_sger` | 自定义内核 |
| `gram(X)` | `einsum` | `cblas_sgemm(T)` | `cublasSgemm(T)` |
| `bilinear(x, A, y)` | 原生 | `cblas_sgemv` | `cublasSgemv` |
| `batched_matmul(A, B)` | `einsum` | 循环+sgemm | `cublasSgemmStridedBatched` |
| `trace(A)` | `einsum` | 手动循环 | 自定义内核 |
| `transpose(A)` | `einsum` | 手动循环 | 自定义内核 |

### 逐元素运算

`add`, `subtract`, `multiply`, `divide`, `scalar ops`, `exp`, `log`, `sqrt`, `sin`, `cos`, `pow`, `abs`, `clip`, `negate`

### 激活函数

`relu`, `leaky_relu`, `elu`, `gelu`, `sigmoid`, `tanh`, `softmax`

### 规约

`sum`, `mean`, `max`, `min`, `norm`, `frobenius_norm`, `var`, `stddev`, `argmax`, `argmin`

### 卷积

`conv2d`, `conv_transpose2d`（支持步长和填充）

### 其他

`hadamard`（逐元素乘法）, `equal`, `greater`, `contract`, `diag`, `diag_matrix`

### 数据 I/O

```cpp
tensor.save("data.csv");   // CSV（仅 1D/2D）
tensor.save("data.npy");   // NumPy 格式
tensor.save("data.npz");   // NumPy 压缩格式
tensor.save("data.json");  // JSON（包含形状和数据）

auto t = load<float>("data.npy");  // 根据扩展名自动检测
```

## 基准测试

构建并运行基准测试，在您的硬件上比较各后端的性能：

```bash
cmake --build build --config Release --target TensorN_Benchmark
./build/bin/benchmarks/Release/TensorN_Benchmark.exe
```

基准测试涵盖：矩阵乘法、逐元素运算、激活函数、规约、卷积以及三个后端之间的比较运算。

### 示例结果

> 测试环境：NVIDIA GeForce RTX 5060 Ti (SM 12.0)
> 矩阵乘法：64x64 | 逐元素：4096 元素 | 向量：1024 | 预热：2 | 重复：5

#### 线性代数（矩阵 64x64）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| matmul | 17.976 | 0.144 | 0.019 | 124.9x | 930.0x |
| gram (X*X^T) | 17.767 | 0.113 | 0.022 | 156.8x | 803.7x |
| dot (vec 1024) | 0.065 | <0.001 | 0.064 | 147.9x | 1.0x |
| outer (256) | 4.706 | 0.019 | 0.030 | 252.5x | 154.9x |
| bilinear (x^T A y) | 0.301 | 0.001 | 0.079 | 301.3x | 3.8x |
| axpy (4096) | 0.002 | 0.001 | 0.011 | 1.4x | 0.2x |
| trace | 0.004 | 0.002 | 0.054 | 1.9x | 0.1x |

#### 逐元素运算（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
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

#### 激活函数（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| relu | 0.002 | 0.002 | 0.006 | 1.4x | 0.3x |
| sigmoid | 0.010 | 0.010 | 0.009 | 1.0x | 1.2x |
| tanh | 0.018 | 0.018 | 0.010 | 1.0x | 1.7x |
| gelu | 0.020 | 0.020 | 0.006 | 1.0x | 3.4x |
| softmax (axis=1) | 0.465 | 0.425 | 0.033 | 1.1x | 13.9x |

#### 规约（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
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

#### 转置（64x64）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| transpose | 0.217 | 0.231 | 0.012 | 0.9x | 18.5x |

#### Conv2d（输入：1x3x32x32，卷积核：16x3x3x3，步长=1，填充=1）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| conv2d | 20.329 | 19.764 | 0.009 | 1.0x | 2183.1x |
| conv_transpose2d | 21.714 | 28.664 | 0.016 | 0.8x | 1382.0x |

#### 比较运算（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| greater (A>B) | 0.002 | 0.002 | 0.005 | 1.0x | 0.4x |
| equal (A==B) | 0.003 | 0.003 | 0.005 | 1.0x | 0.6x |

> 加速比列表示后端相比原生 C++ 的加速倍数。

## 依赖

| 库 | 必需 | 用途 |
|---|---|---|
| C++17 编译器 | 是 | 核心语言特性 |
| nlohmann/json | 自动获取 | JSON 序列化 |
| zlib | 自动获取 | npz 压缩（通过 cnpy） |
| OpenBLAS | 可选 | CPU BLAS 加速 |
| CUDA Toolkit | 可选 | GPU 加速 |

## 许可证

参见 [LICENSE](LICENSE)。
