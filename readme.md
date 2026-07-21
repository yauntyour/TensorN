# TensorN

> A safer, more efficient, and more universal native C++ Tensor library

TensorN 是一个现代、**header-only** 的 C++ 张量计算库，提供类 NumPy 的多维数组操作、Einstein 求和约定 (`einsum`) 引擎，以及可选的 OpenBLAS（CPU）和 CUDA（GPU）后端加速。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![C++26](https://img.shields.io/badge/C%2B%2B-26-blue)
[![CMake 3.20+](https://img.shields.io/badge/CMake-3.20%2B-brightgreen)](CMakeLists.txt)

---

## 特性

- **泛型 `Tensor<T>`** — 任意维度、任意数值类型的多维数组
- **Einstein 求和 (`einsum`)** — 支持显式/隐式模式、`...` 省略号、多输入张量
- **丰富的算子** — `matmul`、`dot`、`outer`、`transpose`、`trace`、`diag`、`sum`、`contract`、`gram`、`bilinear` 等
- **数学函数** — `exp`、`log`、`sqrt`、`sin`、`cos`、`mean`、`var`、`stddev`、`norm`、`frobenius_norm`
- **激活函数** — `relu`、`leaky_relu`、`elu`、`gelu`、`sigmoid`、`tanh`
- **序列化 I/O** — CSV、JSON、NumPy (`.npy`)、NumPy 压缩 (`.npz`)，支持自动检测格式
- **三后端架构** — 纯 C++（默认）、OpenBLAS（CPU 加速）、CUDA（GPU 加速）
- **CUDA 算子** — 完整的 GPU 实现：element-wise、reduction、matmul (cuBLAS)、conv2d、conv_transpose2d、softmax、argmax/argmin、比较运算
- **延迟求值 `opt<T>`** — 链式表达式构建，避免中间临时对象
- **OpenCV 互操作** — 条件编译支持 `Tensor<T>` ↔ `cv::Mat`
- **STL 兼容** — `begin()` / `end()` / `size()` 迭代器接口

---

## 目录结构

```
TensorN/
├── Tensor.hpp                    # 顶层便利头文件 (仅 #include "core/core.hpp")
├── CMakeLists.txt                # CMake 构建配置
├── LICENSE                       # MIT 许可证
├── core/
│   ├── core.hpp                  # 主包含头文件 (聚合所有子模块)
│   ├── tensor.hpp                # Tensor<T> 核心类 + opt<T> 延迟求值
│   ├── einsum.hpp                # Einstein 求和引擎 (纯 C++ 实现)
│   ├── operations.hpp            # 高级张量操作 (matmul/dot/outer/gram/bilinear/contract/激活函数等)
│   ├── static.hpp                # 序列化 I/O (CSV/JSON/NPY/NPZ)
│   ├── BLAS/
│   │   └── blas_tensor.hpp       # OpenBLAS 加速后端 (matmul/dot/norm/axpy/scal/outer/gram/bilinear/激活函数等)
│   ├── CUDA/
│   │   ├── cuda_tensor.hpp       # CudaTensor<T> 设备端内存管理
│   │   ├── cuda_tensor.cu        # 实现文件
│   │   ├── elementwise.hpp/cu    # CUDA element-wise 算子 (add/sub/mul/div/relu/gelu/sigmoid/tanh/exp/比较等)
│   │   ├── reduction.hpp/cu      # CUDA 归约算子 (sum/mean/max/min/argmax/argmin)
│   │   ├── matmul.hpp/cu         # CUDA 矩阵乘 (cuBLAS GEMM)
│   │   └── convolution.hpp/cu    # CUDA 卷积 (conv2d / conv_transpose2d)
│   └── cnpy/                     # NumPy 文件 I/O (rogersce/cnpy, MIT License)
├── tests/
│   ├── test_main.cpp             # CPU 算子测试 (Catch2, ~700行, 涵盖 tensor/einsum/operations/math 等)
│   └── test_cuda.cpp             # CUDA 算子测试 (Catch2, 涵盖 CudaTensor/elementwise/reduction/matmul/conv)
├── exp1.cpp ~ exp5.cpp           # 示例程序
├── benchmark.cpp                 # 三端混合性能基准测试 (Native/BLAS/CUDA)
├── benchmark_native.cpp          # 原生 C++ 性能基线基准
├── benchmark_blas.cpp            # 原生 vs OpenBLAS 基准
├── benchmark_cuda.cpp            # 原生 vs CUDA 基准
└── docx/                         # 文档图片
```

---

## 构建

### 依赖

| 依赖 | 说明 |
|------|------|
| **zlib** | 必需，用于 `.npz` 压缩 |
| **nlohmann/json** | 必需，CMake 自动拉取 (v3.11.3) |
| **Catch2** | 仅测试，CMake 自动拉取 (v3.7.1) |
| **OpenBLAS** | 可选，CPU BLAS 加速 |
| **CUDA Toolkit** | 可选，GPU 加速 (需要 cuBLAS) |
| **OpenCV** | 可选，条件编译 `#ifdef OPENCV_ALL_HPP` |

### CMake 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `TENSORN_BUILD_CUDA` | ON | 启用 CUDA 加速 |
| `TENSORN_BUILD_OPENBLAS` | ON | 启用 OpenBLAS 加速 |
| `TENSORN_BUILD_EXAMPLES` | ON | 构建示例程序 |
| `TENSORN_BUILD_TESTS` | ON | 构建测试 |
| `TENSORN_BUILD_BENCHMARKS` | ON | 构建性能基准 |

### 快速开始

```bash
git clone https://github.com/yauntyour/TensorN.git
cd TensorN

# CPU only (MSYS2/MinGW)
cmake -B build -G Ninja -DTENSORN_BUILD_CUDA=OFF
cmake --build build
cd build && ctest --output-on-failure

# CUDA (MSVC)
cmake -B build_cuda -G "NMake Makefiles" -DTENSORN_BUILD_OPENBLAS=OFF
cmake --build build_cuda

# Or with Visual Studio
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## 使用示例

### 基本操作

```cpp
#include "Tensor.hpp"
using namespace TensorN;

// 创建张量
auto a = Tensor<float>{{1, 2, 3}, {4, 5, 6}};  // 2x3
auto b = Tensor<float>::zeros({3, 2});
auto c = Tensor<float>::eye(3);

// 矩阵乘法 (einsum)
auto d = einsum("ij,jk->ik", a, b);

// 矩阵乘法 (高级 API)
auto e = matmul(a, b);

// 逐元素运算
auto f = a + b;              // 使用 opt<T> 延迟求值
auto g = exp(a);             // math::exp

// 归约
auto s = sum(a, 1);          // 沿 axis=1 求和
auto m = mean(a);            // math::mean

// 激活函数
auto r = relu(a);            // blas::relu
```

### Einstein 求和 (einsum)

`einsum` 是 TensorN 的核心引擎，支持 NumPy 风格的 Einstein 求和约定。所有高级操作（matmul、dot、outer、sum、transpose 等）均基于 `einsum` 构建。

#### 显式模式 (带 `->`)

```cpp
// 矩阵乘法: C_ik = A_ij * B_jk
auto C = einsum("ij,jk->ik", A, B);

// 向量内积: s = x_i * y_i
auto s = einsum("i,i->", x, y);

// 外积: O_ij = x_i * y_j
auto O = einsum("i,j->ij", A, B);

// 逐元素乘: C_ij = A_ij * B_ij
auto C = einsum("ij,ij->ij", A, B);

// 矩阵迹: t = A_ii
auto t = einsum("ii->", A);

// 提取对角线: d_i = A_ii
auto d = einsum("ii->i", A);

// 转置: B_ji = A_ij
auto B = einsum("ij->ji", A);

// 批量矩阵乘: C_bik = A_bij * B_bjk
auto C = einsum("bij,bjk->bik", A, B);

// 3D 缩并: C_ijl = A_ijk * B_ikl
auto C = einsum("ijk,ikl->ijl", A, B);

// 双线性型: s = x_i * A_ij * y_j
auto s = einsum("i,ij,j->", x, A, y);

// Gram 矩阵: G_ij = X_ik * X_jk
auto G = einsum("ik,jk->ij", X, X);

// 广播乘: C_ij = A_ij * B_j
auto C = einsum("ij,j->ij", A, B);
```

#### 隐式模式 (无 `->`)

在不指定输出下标时，引擎自动对重复下标求和，剩余下标按字母序输出：

```cpp
// 矩阵乘法 (隐式): C_ik = A_ij * B_jk, 重复下标 j 自动求和
auto C = einsum("ij,jk", A, B);

// 向量内积 (隐式): i 重复 -> 标量
auto s = einsum("i,i", x, y);

// 迹 (隐式): i 重复 -> 标量
auto t = einsum("ii", A);
```

#### 省略号 (`...`) 广播

`...` 表示零个或多个批量维度，实现批量广播语义：

```cpp
// 求和所有元素
auto s = einsum("...->", A);

// 批量矩阵乘: C_...ik = A_...ij * B_...jk
auto C = einsum("...ij,...jk->...ik", batch_A, batch_B);

// 逐元素 Hadamard 积
auto H = einsum("...,...->...", A, B);
```

#### 多输入缩并

支持任意数量输入张量：

```cpp
// 三输入矩阵乘: D_il = A_ij * B_jk * C_kl
auto D = einsum("ij,jk,kl->il", A, B, C);

// 四输入缩并: E_il = A_ij * B_jk * C_kl * D_lm
auto E = einsum("ij,jk,kl,lm->im", A, B, C, D);
```

#### 自定义缩并 (`contract`)

通过指定轴进行张量缩并：

```cpp
// 对第 1、2 维求和 (0-indexed)
auto C = contract(A, {1, 2});
```

### OpenBLAS 后端

当启用 OpenBLAS 时，所有 `blas::` 命名空间中的运算自动利用 BLAS 加速。无 OpenBLAS 时优雅降级为纯 C++ 实现。

```cpp
#include "Tensor.hpp"
using namespace TensorN;

// 使用 BLAS 加速的矩阵乘
auto C = blas::matmul(A, B);

// BLAS 加速的向量点积
float d = blas::dot(x, y);

// BLAS 加速的 L2 范数
float n = blas::norm(v);

// BLAS 加速的 AXPY: y = alpha * x + y
blas::axpy(0.5f, x, y);

// BLAS 加速的标量乘: x *= alpha
blas::scal(2.0f, x);

// BLAS 加速的 Gram 矩阵
auto G = blas::gram(X);

// BLAS 加速的双线性型
float b = blas::bilinear(x, M, y);

// BLAS 加速的激活函数
auto r = blas::relu(v);
auto g = blas::gelu(v);
auto s = blas::sigmoid(v);
auto t = blas::tanh(v);

// BLAS 加速的数学函数
auto e = blas::exp(v);
auto l = blas::log(v);
auto s = blas::sqrt(v);
```

### CUDA 后端

完整的 GPU 实现，支持设备内存管理、element-wise 运算、归约、矩阵乘 (cuBLAS) 和卷积。

```cpp
#include "Tensor.hpp"
#include "core/CUDA/cuda_tensor.hpp"
#include "core/CUDA/elementwise.hpp"
#include "core/CUDA/reduction.hpp"
#include "core/CUDA/matmul.hpp"
#include "core/CUDA/convolution.hpp"
using namespace TensorN::cuda;

// 创建 CUDA 张量 (从 CPU Tensor 上传)
CudaTensor<float> d_a(A);
CudaTensor<float> d_b(B);

// CUDA Element-wise 运算
CudaTensor<float> d_c(A.shape());
add(d_a, d_b, d_c);         // d_c = d_a + d_b
subtract(d_a, d_b, d_c);    // d_c = d_a - d_b
multiply(d_a, d_b, d_c);    // d_c = d_a * d_b
divide(d_a, d_b, d_c);      // d_c = d_a / d_b

// CUDA 标量运算
multiply_scalar(d_a, 2.0f, d_c);

// CUDA 激活函数
relu(d_a, d_c);
gelu(d_a, d_c);
sigmoid(d_a, d_c);
tanh(d_a, d_c);
softmax(d_a, d_c, 1);       // 沿 axis=1 做 softmax

// CUDA 矩阵乘法 (cuBLAS)
matmul(d_a, d_b, d_c);
batched_matmul(d_a, d_b, d_c);

// CUDA 卷积
CudaTensor<float> d_out({N, K, oH, oW});
conv2d(d_input, d_weight, d_bias, d_out, 1, 0);
conv_transpose2d(d_input, d_weight, d_bias, d_out, 1, 0);

// CUDA 归约
float s = sum(d_a);              // 设备归约，返回主机标量
float m = mean(d_a);
float mx = max(d_a);
float mn = min(d_a);
auto ax = argmax(d_a, 1);        // 返回 CudaTensor<int64_t>
auto an = argmin(d_a, 1);

// CUDA 比较运算
CudaTensor<int> d_comp(shape);
equal(d_a, d_b, d_comp);
greater(d_a, d_b, d_comp);
less(d_a, d_b, d_comp);

// 拷贝回主机
Tensor<float> result = d_c.toTensor();
```

### 序列化 I/O

支持自动格式检测（基于文件扩展名）：

```cpp
// 保存 (自动检测格式)
A.save("matrix.npy");
A.save("data.csv");
A.save("data.json");
A.save("data.npz");

// 加载 (自动检测格式)
auto B = load<float>("matrix.npy");
auto C = load<float>("data.csv");
auto D = load<float>("data.json");
auto E = load<float>("data.npz");

// 或使用独立函数
save_npy(A, "matrix.npy");
save_csv(A, "data.csv");
save_json(A, "data.json");
save_npz(A, "data.npz");

auto B = load_npy<float>("matrix.npy");
auto C = load_csv<float>("data.csv");
```

### OpenCV 互操作

条件编译（`#include <opencv2/core.hpp>` 前定义 `OPENCV_ALL_HPP`）：

```cpp
#ifdef OPENCV_ALL_HPP
// Tensor -> cv::Mat
Tensor<float> tensor({480, 640}, ...);
cv::Mat mat = tensor_to_mat(tensor);

// 支持 8U/8S/16U/16S/32S/32F/64F 等类型
#endif
```

### 工厂函数 / 辅助函数

```cpp
// 创建特殊张量
auto z = zeros<float>({3, 4});     // 全零
auto o = ones<float>({100});        // 全一
auto e = eye<float>(5);             // 单位矩阵
auto a = arange<float>(0, 10, 2);   // [0, 2, 4, 6, 8]

// 延迟求值 (opt<T>)
auto chain = (A + B) * 2.0f - C;   // 链式表达式，无中间临时对象
Tensor<float> result = chain;       // 隐式转换为 Tensor<T>
```

---

## 操作支持表

### BLAS 后端支持 (`TensorN::blas::`)

✓ = OpenBLAS 加速 (float/double) | ○ = 降级为纯 C++ 循环 | — = 不支持

| 操作 | OpenBLAS | 降级实现 | 说明 |
|------|----------|----------|------|
| `matmul(A, B)` | ✓ `cblas_?gemm` | `einsum("ij,jk->ik")` | 2D 矩阵乘 |
| `batched_matmul(A, B)` | ✓ 循环 `cblas_?gemm` | `einsum("bij,bjk->bik")` | 批量 3D 矩阵乘 |
| `dot(x, y)` | ✓ `cblas_?dot` | `einsum("i,i->")` | 向量点积 |
| `norm(v)` | ✓ `cblas_?nrm2` | 循环 | L2 范数 |
| `frobenius_norm(A)` | ✓ `cblas_?nrm2` | 循环 | Frobenius 范数 |
| `axpy(alpha, x, y)` | ✓ `cblas_?axpy` | 循环 | y = alpha*x + y |
| `scal(alpha, x)` | ✓ `cblas_?scal` | 循环 | x *= alpha |
| `outer(x, y)` | ✓ `cblas_?ger` | 双重循环 | 外积 |
| `gram(X)` | ✓ `cblas_?gemm(CblasTrans)` | `einsum("ik,jk->ij")` | Gram 矩阵 |
| `bilinear(x, M, y)` | ✓ `cblas_?gemv` | 循环 | x^T M y |
| `transpose(A)` | ○ | `einsum`/直接拷贝 | 转置 |
| `sum(A)` | ○ | 循环 | 全元素求和 |
| `sum(A, axis)` | ○ | `einsum` | 沿轴求和 |
| `mean(A)` | ○ | sum / N | 均值 |
| `mean(A, axis)` | ○ | 循环 | 沿轴均值 |
| `max(A)` / `min(A)` | ○ | 循环 | 最大/最小值 |
| `trace(A)` | ○ | 循环 | 矩阵迹 |
| `hadamard(A, B)` | ○ | `einsum("...,...->...")` | 逐元素乘 |
| `diag(A)` | ○ | 循环 | 取对角线 |
| `diag_matrix(v)` | ○ | 循环 | 创建对角矩阵 |
| `var(A)` | ○ | 组合实现 | 方差 |
| `stddev(A)` | ○ | 组合实现 | 标准差 |
| `apply(A, func)` | ○ | `std::transform` | 通用逐元素函数 |
| `exp/log/sqrt/sin/cos/abs/pow` | ○ | 逐元素循环 | 数学函数 |
| `relu/leaky_relu/elu` | ○ | 逐元素循环 | 激活函数 |
| `gelu` | ○ | tanh 近似 | GELU 激活 |
| `sigmoid` | ○ | 逐元素循环 | Sigmoid 激活 |
| `tanh` | ○ | 逐元素循环 | Tanh 激活 |

### CUDA 后端支持 (`TensorN::cuda::`)

✓ = 已实现 CUDA kernel

| 操作 | CUDA | 说明 |
|------|------|------|
| `add/subtract/multiply/divide` | ✓ | 逐元素二元运算 |
| `add_scalar/multiply_scalar/subtract_scalar/divide_scalar` | ✓ | 逐元素标量运算 |
| `negate/abs/sqrt/exp/log/sin/cos/pow` | ✓ | 逐元素一元运算 |
| `clip` | ✓ | 数值裁剪 |
| `relu/leaky_relu/elu/gelu/sigmoid/tanh/softmax` | ✓ | 激活函数 |
| `equal/not_equal/greater/less/greater_equal/less_equal` | ✓ | 比较运算 (返回 `CudaTensor<int>`) |
| `sum` (标量) | ✓ | 全元素归约求和 |
| `sum_axis` | ✓ | 沿轴归约求和 |
| `mean` / `mean_axis` | ✓ | 均值 |
| `max` / `max_axis` | ✓ | 最大值 |
| `min` / `min_axis` | ✓ | 最小值 |
| `argmax` / `argmin` | ✓ | 最大/最小值索引 |
| `matmul` | ✓ (cuBLAS) | 2D 矩阵乘 |
| `batched_matmul` | ✓ (cuBLAS) | 批量 3D 矩阵乘 |
| `dot` | ✓ | 向量点积 |
| `conv2d` | ✓ | 2D 卷积 (NCHW) |
| `conv_transpose2d` | ✓ | 2D 转置卷积 |

---

## 架构

```
Tensor<T> (header-only)
    │
    ├── Einsum 引擎 (纯 C++)  ←── 通用后备，所有高级操作的基石
    │
    ├── OpenBLAS 后端          ←── CPU 加速 (BLAS)
    │    └── blas_tensor.hpp   — matmul/dot/norm/axpy/scal/outer/gram/bilinear/激活函数/数学函数
    │
    └── CUDA 后端              ←── GPU 加速 (cuBLAS + CUDA kernels)
         ├── CudaTensor<T>     — 设备内存管理 (host↔device 拷贝)
         ├── elementwise       — 逐元素算子 (add/relu/gelu/softmax/比较等)
         ├── reduction         — 归约算子 (sum/mean/max/min/argmax/argmin)
         ├── matmul            — cuBLAS GEMM (matmul/batched_matmul/dot)
         └── convolution       — conv2d / conv_transpose2d
```

### 命名空间设计

| 命名空间 | 说明 |
|----------|------|
| `TensorN::` | 通用算子（基于 einsum 引擎），始终可用 |
| `TensorN::blas::` | OpenBLAS 加速算子，无 BLAS 时降级为纯 C++ |
| `TensorN::cuda::` | CUDA 加速算子，需要 CUDA Toolkit |
| `TensorN::math::` | 数学函数 (exp/log/sqrt/sin/cos/mean/var/stddev/norm/frobenius_norm) |
| `TensorN::einsum_tools::` | einsum 引擎内部工具 (解析/索引映射) |

### Einsum 引擎设计

核心位于 `core/einsum.hpp`，采用基于标签的通用缩并算法：

1. **解析** — `parse_expression()` 将表达式拆分为输入/输出标签，支持 `...`、隐式模式、匿名标签
2. **索引映射** — `build_index_mapping()` 将标签映射到维度，展开 `...` 为匿名标签，验证维度一致性
3. **主循环** — 基于标签的笛卡尔积迭代，计算每个输出位置的乘积和

所有 `TensorN::` 命名空间中的高级操作（matmul、dot、outer、sum、transpose、trace、diag、gram、bilinear、hadamard、contract）均构建在 `einsum` 之上。

---

## 性能基准

### 测试环境

| 项目 | 配置 |
|------|------|
| CPU | AMD Ryzen 9 9900X 12-Core (24T) |
| GPU | NVIDIA GeForce RTX 5060 Ti (16GB) |
| OS | Windows 11 Pro |
| g++ | MSYS2 MinGW-w64 GCC 16.1.0 |
| MSVC | Visual Studio 2022 v17.14 (19.44) |
| OpenBLAS | (depends on build config) |
| CUDA | 12.9 |

### Benchmark 1: 原生 C++ 基线 (g++ -O3)

| Benchmark | Native ms | Status |
|---|---|---|
| `matmul 256x256` | 5593.629 | OK |
| `bmm 16x64x128` | 3243.852 | OK |
| `dot 1M` | 222.004 | OK |
| `outer 500x500` | 71.385 | OK |
| `gram 200x100` | 1356.469 | OK |
| `bilinear 200x150` | 8.083 | OK |
| `hadamard 1M` | 265.259 | OK |
| `add 1M` | 3.816 | OK |
| `relu 500K` | 3.983 | OK |
| `gelu 500K` | 7.540 | OK |
| `sigmoid 500K` | 10.803 | OK |
| `tanh 500K` | 6.417 | OK |
| `exp 500K` | 9.803 | OK |
| `sum 1M` | 127.711 | OK |
| `sum_axis 100x200` | 4.957 | OK |
| `max 1M` | 4.081 | OK |
| `min 1M` | 3.631 | OK |
| `mean 1M` | 127.963 | OK |
| `norm 1M` | 222.806 | OK |
| `frobenius 1000x1000` | 474.786 | OK |
| `trace 500x500` | 0.083 | OK |
| `transpose 500x500` | 50.592 | OK |
| `diag 500x500` | 0.095 | OK |
| `diag_matrix 500` | 0.353 | OK |
| `axpy 1M` | 6.257 | OK |
| `scal 1M` | 3.504 | OK |
| `var 100K` | 51.940 | OK |
| `stddev 100K` | 52.659 | OK |

### Benchmark 2: 原生 C++ vs OpenBLAS (g++ -O3)

> 同一编译器、同一优化级别，仅 BLAS 加速为变量

| Benchmark | Native ms | BLAS ms | BLAS/Native | Status |
|---|---|---|---|---|
| `matmul 256x256` | 5586.142 | 154.584 | 36.1x | OK |
| `bmm 16x64x128` | 3201.664 | 0.079 | 40726.8x | OK |
| `dot 1M` | 214.967 | 0.129 | 1667.6x | OK |
| `outer 500x500` | 66.939 | 0.279 | 240.1x | OK |
| `gram 200x100` | 1304.510 | 74.370 | 17.5x | OK |
| `bilinear 200x150` | 8.075 | 0.002 | 3307.4x | OK |
| `hadamard 1M` | 256.172 | 8.849 | 28.9x | OK |
| `add 1M` | 3.913 | — | — | OK |
| `relu 500K` | 3.817 | 4.121 | 0.9x | OK |
| `gelu 500K` | 7.454 | 8.010 | 0.9x | OK |
| `sigmoid 500K` | 10.476 | 11.444 | 0.9x | OK |
| `tanh 500K` | 6.813 | 6.860 | 1.0x | OK |
| `exp 500K` | 9.037 | 9.722 | 0.9x | OK |
| `sum 1M` | 127.735 | 3.303 | 38.7x | OK |
| `sum_axis 100x200` | 3.927 | 0.100 | 39.1x | OK |
| `max 1M` | 3.120 | 1.370 | 2.3x | OK |
| `min 1M` | 3.502 | 1.363 | 2.6x | OK |
| `mean 1M` | 128.293 | 3.422 | 37.5x | OK |
| `norm 1M` | 216.850 | 0.105 | 2060.8x | OK |
| `frobenius 1000x1000` | 472.397 | 0.104 | 4531.5x | OK |
| `trace 500x500` | 0.076 | 0.036 | 2.1x | OK |
| `transpose 500x500` | 51.802 | 40.608 | 1.3x | OK |
| `diag 500x500` | 0.099 | 0.036 | 2.8x | OK |
| `diag_matrix 500` | 0.279 | 0.122 | 2.3x | OK |
| `axpy 1M` | 6.984 | 0.962 | 7.3x | OK |
| `scal 1M` | 3.939 | 0.632 | 6.2x | OK |
| `var 100K` | 52.463 | 0.642 | 81.7x | OK |
| `stddev 100K` | 50.392 | 0.641 | 78.6x | OK |

### Benchmark 3: 原生 C++ vs CUDA (MSVC -O3)

> 同一编译器、同一优化级别，仅 GPU 加速为变量（不含 BLAS）

| Benchmark | Native ms | CUDA ms | CUDA/Native | Status |
|---|---|---|---|---|
| `matmul 256x256` | 1154.877 | 0.048 | 24059.9x | OK |
| `bmm 16x64x128` | 571.823 | 0.020 | 28734.8x | OK |
| `dot 1M` | 45.035 | 0.059 | 756.9x | OK |
| `hadamard 1M` | 66.066 | 0.006 | 10486.7x | OK |
| `add 1M` | 0.821 | 0.010 | 83.8x | OK |
| `relu 500K` | 0.341 | 0.006 | 59.8x | OK |
| `gelu 500K` | 2.766 | 0.006 | 439.0x | OK |
| `sigmoid 500K` | 1.012 | 0.020 | 50.6x | OK |
| `tanh 500K` | 2.067 | 0.006 | 362.7x | OK |
| `exp 500K` | 1.466 | 0.007 | 203.7x | OK |
| `sum 1M` | 29.661 | 0.057 | 523.1x | OK |
| `sum_axis 100x200` | 0.859 | 0.017 | 51.2x | OK |
| `max 1M` | 0.362 | 0.060 | 6.0x | OK |
| `min 1M` | 0.367 | 0.067 | 5.5x | OK |
| `mean 1M` | 21.926 | 0.062 | 355.9x | OK |
| `scal 1M` | 0.766 | 0.007 | 109.4x | OK |
| `softmax 2048x1024` | — | 1.449 | — | OK |
| `conv2d 1x3x32x32` | — | 0.004 | — | OK |
| `argmax 100x200` | — | 0.017 | — | OK |
| `argmin 100x200` | — | 0.016 | — | OK |
| `equal 100K` | — | 0.003 | — | OK |
| `greater 100K` | — | 0.005 | — | OK |
| `convT2d 1x3x16x16` | — | 0.029 | — | OK |

### Benchmark 4: 三端混合对比 — Native / BLAS / CUDA (MSVC -O3)

> 同一编译器 (MSVC -O3)、同一次运行，三个后端全量对比
> BLAS 后端使用 MSVC 编译，若无 OpenBLAS 则回退到纯 C++ 实现
> 测试时间: 2026-07-21 (本档更新时运行)

| Benchmark | Native ms | BLAS ms | CUDA ms | BLAS/N | CUDA/N | Status |
|---|---|---|---|---|---|---|
| `matmul 256x256` | 1161.497 | 1132.593 | 0.045 | 1.0x | 25868.5x | OK |
| `bmm 16x64x128` | 558.487 | 618.925 | 0.008 | 0.9x | 70694.6x | OK |
| `dot 1M` | 48.392 | 46.373 | 0.068 | 1.0x | 706.5x | OK |
| `outer 500x500` | 18.276 | 4.814 | — | 3.8x | — | OK |
| `gram 200x100` | 312.248 | 252.292 | — | 1.2x | — | OK |
| `bilinear 200x150` | 1.889 | 0.586 | — | 3.2x | — | OK |
| `hadamard 1M` | 67.947 | 0.758 | 0.007 | 89.7x | 9992.2x | OK |
| `add 1M` | 0.766 | — | 0.008 | — | 99.5x | OK |
| `relu 500K` | 0.334 | 0.330 | 0.006 | 1.0x | 58.5x | OK |
| `gelu 500K` | 2.607 | 2.495 | 0.005 | 1.0x | 491.9x | OK |
| `sigmoid 500K` | 1.007 | 0.958 | 0.007 | 1.1x | 134.2x | OK |
| `tanh 500K` | 2.044 | 2.047 | 0.005 | 1.0x | 378.5x | OK |
| `exp 500K` | 0.814 | 0.823 | 0.008 | 1.0x | 95.7x | OK |
| `sum 1M` | 24.468 | 0.365 | 0.055 | 67.0x | 447.3x | OK |
| `sum_axis 100x200` | 0.823 | 0.004 | 0.015 | 191.4x | 54.9x | OK |
| `max 1M` | 0.367 | 0.061 | 0.059 | 6.1x | 6.2x | OK |
| `min 1M` | 0.361 | 0.056 | 0.059 | 6.5x | 6.1x | OK |
| `mean 1M` | 23.685 | 0.503 | 0.061 | 47.1x | 385.8x | OK |
| `norm 1M` | 43.933 | 0.365 | — | 120.3x | — | OK |
| `frobenius 1000x1000` | 112.837 | 0.370 | — | 304.8x | — | OK |
| `trace 500x500` | 0.013 | 0.009 | — | 1.4x | — | OK |
| `transpose 500x500` | 11.772 | 9.979 | — | 1.2x | — | OK |
| `diag 500x500` | 0.031 | 0.010 | — | 3.0x | — | OK |
| `diag_matrix 500` | 0.024 | 0.026 | — | 0.9x | — | OK |
| `axpy 1M` | 0.681 | 0.813 | — | 0.8x | — | OK |
| `scal 1M` | 0.592 | 0.599 | 0.006 | 1.0x | 107.7x | OK |
| `var 100K` | 11.126 | 0.073 | — | 151.8x | — | OK |
| `stddev 100K` | 10.890 | 0.073 | — | 148.6x | — | OK |
| `softmax 2048x1024` | — | — | 1.547 | — | — | OK |
| `conv2d 1x3x32x32` | — | — | 0.004 | — | — | OK |
| `argmax 100x200` | — | — | 0.022 | — | — | OK |
| `argmin 100x200` | — | — | 0.016 | — | — | OK |
| `equal 100K` | — | — | 0.003 | — | — | OK |
| `greater 100K` | — | — | 0.003 | — | — | OK |
| `convT2d 1x3x16x16` | — | — | 0.007 | — | — | OK |

**All benchmarks passed!**

### 结果分析

#### BLAS 加速效果 (g++ -O3)

| 加速最显著的操作 | 倍率 | 说明 |
|---|---|---|
| `bmm 16x64x128` | ~40727x | 批量矩阵乘，BLAS 高度优化 |
| `frobenius 1000x1000` | ~4532x | 大矩阵 Frobenius 范数 |
| `bilinear 200x150` | ~3307x | 双线性型，gemv 加速 |
| `norm 1M` | ~2061x | 向量 L2 范数，nrm2 加速 |
| `dot 1M` | ~1668x | 向量点积，dot 加速 |

- 大规模线性代数操作 BLAS 加速极显著
- 逐元素操作 (relu/gelu/sigmoid/tanh) BLAS 无加速（仍为纯 C++ 循环）

#### CUDA 加速效果 (MSVC -O3)

| 加速最显著的操作 | 倍率 | 说明 |
|---|---|---|
| `matmul 256x256` | ~24060x | cuBLAS GEMM 相对纯 CPU |
| `bmm 16x64x128` | ~28735x | 批量矩阵乘众核并行 |
| `hadamard 1M` | ~10487x | 逐元素乘 GPU 并行 |
| `dot 1M` | ~757x | 向量点积 GPU 归约 |
| `sum 1M` | ~523x | 大数组归约求和 |

- GPU 众核并行在计算密集型任务上优势极致
- 小规模操作 (max/min) 受 GPU 启动开销制约，加速比相对较小

#### 编译器差异

- g++ 原生 C++ 在大矩阵乘法上比 MSVC 慢约 3.8 倍（5593ms vs 1161ms 对于 256x256 matmul）
- 但 MSVC 使用了 `/O2 /Ot /Ob2 /Oi /GL /LTCG` 全程序优化
- 两个平台在各自基准中保持编译器一致，确保对比公平

---

## 许可证

[MIT License](LICENSE) © 2025 TensorN Contributors

内含的 [cnpy](https://github.com/rogersce/cnpy) 库同样基于 MIT 许可证。
