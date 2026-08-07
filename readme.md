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
    <a href="#-特性">特性</a> ·
    <a href="#-快速开始">快速开始</a> ·
    <a href="#-构建">构建</a> ·
    <a href="#-架构">架构</a> ·
    <a href="#-运算">运算</a> ·
    <a href="#-数据-io">数据 I/O</a> ·
    <a href="#-原地操作">原地操作</a> ·
    <a href="#-零拷贝视图--内存池">零拷贝</a> ·
    <a href="#cuda-流与异步">CUDA 流</a> ·
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
- **三种加速后端** — 原生 C++、OpenBLAS、CUDA/cuBLAS，共享统一 API 模式
- **低精度数据类型** — `half`(FP16)、`bfloat16`(BF16)、`tf32`、`fp8_e4m3`、`fp8_e5m2`，CPU 与 GPU 张量核心全链路支持
- **爱因斯坦求和** — `einsum("ij,jk->ik", A, B)` 实现灵活的张量运算
- **丰富的运算集** — 线性代数、逐元素数学运算、激活函数、规约、卷积、比较运算
- **数据 I/O** — CSV、NumPy `.npy`/`.npz`、JSON、PyTorch `.pt`、GGUF 格式，附带 TensorN↔PyTorch 桥接工具
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

    // 保存 / 加载（根据扩展名自动检测格式）
    C.tensor.save("result.npy");
    auto loaded = load<float>("result.npy");
}
```

---

## 🛠 构建

需要 **CMake 3.18+**、**C++17 以上编译器**。CUDA 和 OpenBLAS 为可选依赖。

```bash
# g++
cmake -B build -DTENSORN_ENABLE_CUDA=ON -DTENSORN_ENABLE_OPENBLAS=ON
cmake --build build --config Release

# MSVC（CUDA 在 Windows 上需要 MSVC 编译器）
cmake -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="D:/x64/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --config Release
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `TENSORN_ENABLE_CUDA` | ON | 启用 CUDA/cuBLAS 后端 |
| `TENSORN_ENABLE_OPENBLAS` | ON | 启用 OpenBLAS 后端 |
| `TENSORN_ENABLE_OPENMP` | ON | 启用 OpenMP 多核并行 |
| `TENSORN_BUILD_EXAMPLES` | ON | 构建示例程序 |
| `TENSORN_BUILD_BENCHMARKS` | ON | 构建基准测试程序 |

> 在 Windows + MSVC 下 CUDA 自动启用；若使用 MinGW 则自动禁用。OpenBLAS 未找到时会自动降级为原生后端。

---

## 🏗 架构

```
TensorN
├── TensorN.hpp          总入口，包含 core/core.hpp
├── core/
│   ├── core.hpp         统一头文件聚合
│   ├── dtypes.hpp       低精度数据类型（half / bfloat16 / tf32 / fp8_e4m3 / fp8_e5m2）
│   ├── tensor.hpp       核心张量类（N 维，行主序）
│   ├── einsum.hpp       爱因斯坦求和引擎
│   ├── operations.hpp   高级运算（matmul, dot, outer, gram, ...）
│   ├── static.hpp       数据 I/O（csv, npy, npz, json, pt, gguf, safetensors）
│   ├── memory_pool.hpp  CPU 内存池（桶分配器、PooledAllocator、PooledVector）
│   ├── BLAS/            OpenBLAS 加速后端（OpenMP 多核并行、im2col+GEMM 卷积）
│   │   └── blas_tensor.hpp
│   ├── CUDA/            CUDA/cuBLAS 加速后端
│   │   ├── cuda_tensor.hpp    CudaTensor<T>（设备内存管理、异步传输、零拷贝视图）
│   │   ├── cublas_ex.hpp      cuBLAS GemmEx 低精度 GEMM 分发（FP16/BF16/TF32/FP8）
│   │   ├── cuda_stream.hpp    CudaStream、CudaEvent、流池、设备/页锁定内存池
│   │   ├── fused_kernels.hpp  融合内核（matmul+activation、conv+activation、add_relu 等）
│   │   ├── matmul.cu          矩阵乘法（cuBLAS，流感知）
│   │   ├── elementwise.cu     逐元素运算与激活函数内核
│   │   ├── reduction.cu       规约内核（sum, mean, max, ...）
│   │   └── convolution.cu     Conv2d / ConvTranspose2d 内核
│   ├── GGUF/            GGUF 格式读写
│   ├── HF/              HuggingFace 格式读写
│   │   └── safetensors.hpp  safetensors 格式读写（含分片 model.safetensors-00001-of-00001.safetensors）
│   └── cnpy/            NumPy .npy/.npz 格式支持
├── example/             示例程序（exp1 ~ exp10）
├── benchmark/           基准测试
└── tools/               辅助工具（pt_converter.py）
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
| `axpy(alpha, x, y)` | 原生 | `cblas_saxpy` | `cublasSaxpy` |

### 逐元素运算

`add`, `subtract`, `multiply`, `divide`, 标量运算, `exp`, `log`, `sqrt`, `sin`, `cos`, `pow`, `abs`, `clip`, `negate`

### 激活函数

`relu`, `leaky_relu`, `elu`, `gelu`, `sigmoid`, `tanh`, `softmax`

### 规约

`sum`, `mean`, `max`, `min`, `norm`, `frobenius_norm`, `var`, `stddev`, `argmax`, `argmin`

### 卷积

`conv2d`, `conv_transpose2d`（支持步长和填充）

### 其他

`hadamard`, `equal`, `greater`, `contract`, `diag`, `diag_matrix`

---

## 🧮 低精度数据类型（加速计算）

为 GPU 张量核心与推理部署提供的高效数据类型，均实现为纯 C++17 类型（位操作转换，舍入采用 round-to-nearest-even），**同时支持 CPU 运算与 CUDA 内核**：

| 类型 | 别名 | 存储 | 指数/尾数 | 动态范围 | 硬件加速 |
|---|---|---|---|---|---|
| `TensorN::half` | `fp16` | 2B | 5/10 (bias 15) | ±65504 | FP16 张量核心 (sm80+) |
| `TensorN::bfloat16` | `bf16` | 2B | 8/7 (bias 127) | ±3.4e38 | BF16 张量核心 (sm80+) |
| `TensorN::tf32` | — | 4B | 8/10 (截断) | 同 FP32 | TF32 张量核心 (sm80+) |
| `TensorN::fp8_e4m3` | — | 1B | 4/3 (bias 7) | ±448 | FP8 张量核心 (sm89+) |
| `TensorN::fp8_e5m2` | — | 1B | 5/2 (bias 15) | ±57344 | FP8 张量核心 (sm89+) |

### 基本用法

```cpp
#include "TensorN.hpp"
using namespace TensorN;
using TensorN::fp16;  // CUDA/OpenBLAS 头文件在全局命名空间声明了 half/bfloat16，
using TensorN::bf16;  // 使用别名或全限定名 TensorN::half 避免二义

Tensor<fp16> A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
auto C = matmul(A, A.tensor);            // CPU：einsum 自动支持
fp16 s = blas::sum(A);                   // 规约、逐元素运算均可用

std::cout << dtype_of<fp16>() << "\n";   // 类型自省：dtype_t::Float16
std::cout << name_of<bf16>() << "\n";    // "bfloat16"
A.save("w.pt");                          // .pt 序列化支持全部新类型
auto W = load<fp16>("w.pt");
```

### CUDA 张量核心

```cpp
auto a_dev = CudaTensor<fp16>(A);        // 布局与 __half 位兼容，零转换
auto b_dev = CudaTensor<fp16>(B);
CudaTensor<fp16> c_dev({M, N});
cuda::matmul(a_dev, b_dev, c_dev);       // cublasGemmEx，FP32 累加

CudaTensor<bf16> dbf(...);               // BF16 同理
CudaTensor<tf32> dtf(...);               // TF32 存储 + CUBLAS_COMPUTE_32F_FAST_TF32

cuda::set_tf32(true);                    // 全局开关：float matmul 也走 TF32 张量核心
cuda::matmul(a_f32, b_f32, c_f32);
cuda::set_tf32(false);
```

- FP16/BF16/TF32 需要 compute capability ≥ 8.0（Ampere+），FP8 GEMM 需要 ≥ 8.9（Ada/Hopper/Blackwell）且 M/N/K 为 16 的倍数
- FP8 GEMM 取决于 cuBLAS 对具体硬件的支持（如消费级 Blackwell sm120 目前返回 `CUBLAS_STATUS_NOT_SUPPORTED`，会抛出带说明的异常）；FP8 的存储与逐元素运算在所有平台可用
- 低精度类型的运算在 float 中执行、每次运算按 RNE 舍入一次；CUDA GEMM 使用 FP32 累加，精度优于 CPU 端逐 T 累加

---

## 💾 数据 I/O

```cpp
tensor.save("data.csv");   // CSV（仅 1D/2D）
tensor.save("data.npy");   // NumPy 格式
tensor.save("data.npz");   // NumPy 压缩格式
tensor.save("data.json");  // JSON（包含形状和数据）
tensor.save("data.pt");    // TensorN .pt 二进制格式（支持 .pt / .pth 扩展名）
tensor.save("data.gguf");  // GGUF 格式（支持附加元数据）
tensor.save("data.safetensors");                 // safetensors 格式
tensor.save("model.safetensors-00001-of-00001.safetensors"); // safetensors 分片命名

auto t = load<float>("data.pt");  // 根据扩展名自动检测
```

**支持类型：** `float`, `double`, `int32_t`, `int64_t`, `uint8_t`, `int16_t`, `half`, `bfloat16`, `tf32`, `fp8_e4m3`, `fp8_e5m2`（`.pt`/`.gguf`/`.safetensors` 格式支持全部类型；`.npy`/`.npz`/`.json` 仅支持数值类型）

**safetensors 互操作（与 HuggingFace 生态完全兼容）：**

```cpp
// 单张量
tensor.save("model.safetensors");
auto t1 = load<float>("model.safetensors");

// 多张量（同类型）
save_safetensors_multi<float>({{"w1", w1}, {"w2", w2}}, "model.safetensors");
auto model = load_safetensors_multi<float>("model.safetensors");

// 混合 dtype（通过 SafeTensor 载体，与 PyTorch save_file 等价）
std::vector<std::pair<std::string, SafeTensor>> state;
state.emplace_back("weight", make_safetensor(w));
state.emplace_back("ids", make_safetensor(ids));   // int64 张量
save_safetensors_multi(state, "model.safetensors", {{"format", "pt"}});

// 分片保存（默认单分片上限 5GB，输出 model.safetensors-00001-of-00002.safetensors）
save_safetensors_sharded(state, "model.safetensors", 2ULL * 1024 * 1024 * 1024);
// 分片加载（自动发现并合并全部分片）
auto sharded = load_safetensors_sharded<float>("model.safetensors");
```

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

---

## 🔄 零拷贝视图 & 内存池

- **`view(shape)` / `reshape(shape)`** — 返回共享底层数据的新张量，不分配内存
- **`memory_pool.hpp`** — CPU 桶分配器，提供 `PooledAllocator<T>` 和 `PooledVector<T>`
- **`from_pool(shape, pool)`** — 从内存池分配张量

---

## 🌊 CUDA 流与异步

所有 CUDA 运算均提供 `cudaStream_t` 重载，配合流池与异步内存池实现高效流水线：

```cpp
auto stream = CudaStreamPool::acquire();
auto a_dev = CudaTensor<float>::fromPinned(a_host, stream);
auto b_dev = CudaTensor<float>::fromPinned(b_host, stream);
auto c_dev = matmul(a_dev, b_dev, stream);        // 流感知 cuBLAS
c_dev.copyToHostAsync(result, stream);            // 异步回传
stream.sync();
```

- **`CudaStreamPool`** — 预创建 CUDA 流复用
- **`CudaMemoryPool` / `PinnedMemoryPool`** — 设备与页锁定内存池
- **`copyFromHostAsync()` / `copyToHostAsync()` / `copyFromDeviceAsync()`** — 异步数据传输
- **`memset_zero_async()`** — 异步零初始化
- **`view()` / `reshape()`** — 设备端零拷贝视图

---

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
> **矩阵乘法：** 512×512 | **逐元素：** 65536 元素 | **向量：** 4096 | **预热：** 2 | **重复：** 5

#### 线性代数（矩阵 512×512）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| matmul | 9101.528 | 64.036 | 0.033 | 142.1× | 279229.1× |
| gram (X·Xᵀ) | 9561.571 | 64.693 | 0.037 | 147.8× | 260459.5× |
| dot (vec 4096) | 0.177 | 0.002 | 0.084 | 105.3× | 2.1× |
| outer (1024) | 70.760 | 0.884 | 0.017 | 80.1× | 4282.1× |
| bilinear (xᵀAy) | 16.983 | 0.112 | 0.054 | 151.2× | 315.2× |
| axpy (65536) | 0.033 | 0.018 | 0.010 | 1.8× | 3.3× |
| trace | 0.012 | 0.010 | 0.050 | 1.1× | 0.2× |

#### 逐元素运算（65536 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
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

#### 激活函数（65536 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| relu | 0.035 | 0.179 | 0.009 | 0.2× | 3.8× |
| sigmoid | 0.180 | 1.361 | 0.014 | 0.1× | 12.4× |
| tanh | 0.321 | 0.387 | 0.009 | 0.8× | 36.3× |
| gelu | 0.369 | 0.509 | 0.006 | 0.7× | 64.6× |
| softmax (axis=1) | 11.981 | 0.049 | 0.048 | 242.1× | 251.7× |

#### 规约（65536 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
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

#### 转置（512×512）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| transpose | 17.296 | 0.375 | 0.011 | 46.1× | 1515.7× |

#### Conv2d（输入：1×3×64×64，卷积核：32×3×3×3，步长=1，填充=1）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| conv2d | 172.840 | 1.679 | 0.032 | 103.0× | 5343.5× |
| conv_transpose2d | 180.763 | 5.396 | 0.060 | 33.5× | 3038.0× |

#### 比较运算（65536 元素）

| 运算 | 原生(ms) | OpenBLAS(ms) | cuBLAS(ms) | BLAS/原生 | CUDA/原生 |
|---|---|---|---|---|---|
| greater (A>B) | 0.066 | 0.023 | 0.006 | 2.8× | 10.7× |
| equal (A==B) | 0.068 | 0.023 | 0.005 | 3.0× | 12.4× |

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
