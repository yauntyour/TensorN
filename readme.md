<p align="center">
  <h1 align="center">TensorN</h1>
  <p align="center">
    <em>一个 C++17 纯头文件张量库 · 支持 OpenBLAS & CUDA/cuBLAS 加速</em>
  </p>
  <p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/c%2B%2B-17-00599C.svg" alt="C++17">
    <img src="https://img.shields.io/badge/version-1.0.0-green.svg" alt="Version 1.0.0">
    <img src="https://img.shields.io/badge/header--only-✔-brightgreen.svg" alt="Header-only">
  </p>
  <p align="center">
    <a href="#-快速开始">快速开始</a> ·
    <a href="#-构建">构建</a> ·
    <a href="#-架构">架构</a> ·
    <a href="#-运算">运算</a> ·
    <a href="#-原地操作">原地操作</a> ·
    <a href="#-零拷贝视图--内存池">零拷贝</a> ·
    <a href="#cuda-流与异步">CUDA流</a> ·
    <a href="#-融合内核">融合内核</a> ·
    <a href="#-基准测试">基准测试</a> ·
    <a href="#-依赖">依赖</a>
  </p>
  <p align="center">
    <a href="./docx/readme_en.md">English</a> | 中文
  </p>
</p>

---

## ✨ 特性

- **纯头文件** — 仅需 `#include "TensorN.hpp"` 即可使用
- **三种加速后端** — 原生 C++、OpenBLAS、CUDA/cuBLAS
- **爱因斯坦求和** — `einsum("ij,jk->ik", A, B)` 实现灵活的张量运算
- **丰富的运算集** — 线性代数、逐元素数学运算、激活函数、规约、卷积
- **数据 I/O** — CSV、NumPy `.npy`/`.npz`、JSON、PyTorch `.pt` 格式，附带 TensorN↔PyTorch 桥接工具
- **OpenCV 互操作** — 可选的 `cv::Mat` 转换
- **原地操作** — `add_()`, `sub_()`, `mul_()`, `div_()`, `apply_()`, `fill_()`, `zero_()` 等零分配原地变换
- **零拷贝视图** — `view()`, `reshape()` 共享底层数据，无需复制
- **CUDA 流与异步** — 流感知 cuBLAS、异步传输、内存池与融合内核
- **OpenBLAS 多核加速** — OpenMP 并行化所有非 BLAS 循环，im2col+GEMM 卷积

---

## 🚀 快速开始

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

---

## 🛠 构建

需要 **CMake 3.18+**、**C++17 编译器**。CUDA 和 OpenBLAS 为可选依赖。

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

---

## 🏗 架构

```
TensorN
├── Tensor<T>          核心张量类（N 维，行主序）
├── opt<T>             链式操作的惰性求值包装器
├── einsum()           爱因斯坦求和引擎
├── operations.hpp     高级运算（matmul, dot, outer, gram, ...）
├── static.hpp         数据 I/O（csv, npy, npz, json, pt）
├── memory_pool.hpp    CPU 内存池（桶分配器、PooledAllocator、PooledVector）
├── BLAS/              OpenBLAS 加速后端（OpenMP 多核并行、im2col+GEMM 卷积）
│   └── blas_tensor.hpp
└── CUDA/              CUDA/cuBLAS 加速后端
    ├── cuda_tensor.hpp    CudaTensor<T>（设备内存管理、异步传输、零拷贝视图）
    ├── cuda_stream.hpp    CudaStream、CudaEvent、流池、设备/页锁定内存池
    ├── fused_kernels.hpp  融合内核（matmul+activation、conv+activation、add_relu 等）
    ├── matmul.cu          矩阵乘法（cuBLAS，流感知）
    ├── elementwise.cu     逐元素运算与激活函数内核
    ├── reduction.cu       规约内核（sum, mean, max, ...）
    └── convolution.cu     Conv2d / ConvTranspose2d 内核
```

### 后端

| 后端 | 命名空间 | 说明 |
|---|---|---|
| 原生 C++ | `TensorN::` | 基于 einsum，无外部依赖 |
| OpenBLAS | `TensorN::blas::` | 使用 cblas_sgemm/cblas_dgemm |
| cuBLAS | `TensorN::cuda::` | 使用 cublasSgemm/cublasDgemm + 自定义 CUDA 内核 |

> 三个后端共享相同的 API 模式——原生/OpenBLAS 传入 `Tensor<T>`，CUDA 传入 `CudaTensor<T>`。

---

## 🔧 运算

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
tensor.save("data.pt");    // TensorN .pt 二进制格式（支持 .pt / .pth 扩展名）

auto t = load<float>("data.pt");  // 根据扩展名自动检测
```

**支持类型：** `float`, `double`, `int32_t`, `int64_t`, `uint8_t`, `int16_t`

**与 PyTorch 互操作：** 使用 `tools/pt_converter.py` 可在 TensorN `.pt` 和 PyTorch `.pth` 之间相互转换：

```bash
# PyTorch .pth → TensorN .pt
python tools/pt_converter.py torch2pt model.pth data.pt

# TensorN .pt → PyTorch .pth
python tools/pt_converter.py pt2torch data.pt model.pth

# 也支持 .npy 中转
python tools/pt_converter.py np2pt data.npy data.pt
python tools/pt_converter.py pt2np data.pt data.npy
```

---

## ⚡ 原地操作

对 Tensor 和 CudaTensor 均支持的零分配原地变换：

```cpp
Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
t.add_(2.0f);          // 张量每个元素加 2
t.mul_(0.5f);          // 张量每个元素乘 0.5
t.apply_([](float x) { return x * x; });  // 逐元素自定义变换
t.zero_();             // 全零填充
```

## 🔄 零拷贝视图 & 内存池

- **`view(shape)` / `reshape(shape)`** — 返回共享底层数据的新张量，不分配内存
- **`memory_pool.hpp`** — CPU 桶分配器，提供 `PooledAllocator<T>` 和 `PooledVector<T>`
- **`from_pool(shape, pool)`** — 从内存池分配张量

## 🌊 CUDA 流与异步

所有 CUDA 运算均提供 `cudaStream_t` 重载，配合流池与异步内存池实现高效流水线：

```cpp
auto stream = CudaStreamPool::acquire();
auto a_dev = CudaTensor<float>::fromPinned(a_host, stream);
auto b_dev = CudaTensor<float>::fromPinned(b_host, stream);
auto c_dev = matmul(a_dev, b_dev, stream);        // 流感知 cuBLAS
c_dev.copyToHostAsync(result, stream);             // 异步回传
stream.sync();
```

- **`CudaStreamPool`** — 预创建 CUDA 流复用
- **`CudaMemoryPool` / `PinnedMemoryPool`** — 设备与页锁定内存池
- **`copyFromHostAsync()` / `copyToHostAsync()` / `copyFromDeviceAsync()`** — 异步数据传输
- **`memset_zero_async()`** — 异步零初始化
- **`view()` / `reshape()`** — 设备端零拷贝视图

## 🔥 融合内核

消除中间缓冲区，单次内核完成多重运算：

| 融合运算 | 描述 |
|---|---|
| `fused_matmul_relu(A, B)` | 矩阵乘法 + ReLU 激活 |
| `fused_conv_relu(input, kernel, bias)` | Conv2d + Bias + ReLU |
| `fused_add_relu(A, B)` | 逐元素加法 + ReLU |
| `fused_mul_add(A, B, C)` | 逐元素乘法 + 加法 |
| `fused_batchnorm_inference(x, gamma, beta, mean, var)` | 推理批归一化 |
| `fused_residual_block(x, w1, w2, ...)` | 残差块（MLP/卷积） |

---

## 📊 基准测试

构建并运行基准测试，在您的硬件上比较各后端的性能：

```bash
cmake --build build --config Release --target TensorN_Benchmark
./build/bin/benchmarks/Release/TensorN_Benchmark.exe
```

基准测试涵盖：矩阵乘法、逐元素运算、激活函数、规约、卷积以及三个后端之间的比较运算。

### 示例结果

> **测试环境：** NVIDIA GeForce RTX 5060 Ti (SM 12.0)
> **矩阵乘法：** 64×64 | **逐元素：** 4096 元素 | **向量：** 1024 | **预热：** 2 | **重复：** 5

#### 线性代数（矩阵 64×64）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| matmul | 17.976 | 0.144 | 0.019 | 124.9× | 930.0× |
| gram (X·Xᵀ) | 17.767 | 0.113 | 0.022 | 156.8× | 803.7× |
| dot (vec 1024) | 0.065 | <0.001 | 0.064 | 147.9× | 1.0× |
| outer (256) | 4.706 | 0.019 | 0.030 | 252.5× | 154.9× |
| bilinear (xᵀAy) | 0.301 | 0.001 | 0.079 | 301.3× | 3.8× |
| axpy (4096) | 0.002 | 0.001 | 0.011 | 1.4× | 0.2× |
| trace | 0.004 | 0.002 | 0.054 | 1.9× | 0.1× |

#### 逐元素运算（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
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

#### 激活函数（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| relu | 0.002 | 0.002 | 0.006 | 1.4× | 0.3× |
| sigmoid | 0.010 | 0.010 | 0.009 | 1.0× | 1.2× |
| tanh | 0.018 | 0.018 | 0.010 | 1.0× | 1.7× |
| gelu | 0.020 | 0.020 | 0.006 | 1.0× | 3.4× |
| softmax (axis=1) | 0.465 | 0.425 | 0.033 | 1.1× | 13.9× |

#### 规约（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
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

#### 转置（64×64）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| transpose | 0.217 | 0.231 | 0.012 | 0.9× | 18.5× |

#### Conv2d（输入：1×3×32×32，卷积核：16×3×3×3，步长=1，填充=1）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| conv2d | 20.329 | 19.764 | 0.009 | 1.0× | 2183.1× |
| conv_transpose2d | 21.714 | 28.664 | 0.016 | 0.8× | 1382.0× |

#### 比较运算（4096 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| greater (A>B) | 0.002 | 0.002 | 0.005 | 1.0× | 0.4× |
| equal (A==B) | 0.003 | 0.003 | 0.005 | 1.0× | 0.6× |

> 加速比列表示后端相比原生 C++ 的加速倍数。

---

## 📦 依赖

| 库 | 必需 | 用途 |
|---|---|---|
| C++17 编译器 | ✅ | 核心语言特性 |
| nlohmann/json | 🔽 自动获取 | JSON 序列化 |
| zlib | 🔽 自动获取 | npz 压缩（通过 cnpy） |
| OpenBLAS | ⬜ 可选 | CPU BLAS 加速 |
| CUDA Toolkit | ⬜ 可选 | GPU 加速 |

---

## 📄 许可证

本项目采用 **MIT 许可证** —— 参见 [LICENSE](LICENSE) 文件。
