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
- **CUDA 算子** — 完整的 GPU 实现：element-wise、reduction、matmul (cuBLAS)、conv2d、conv_transpose2d
- **延迟求值 `opt<T>`** — 链式表达式构建，避免中间临时对象
- **OpenCV 互操作** — 条件编译支持 `Tensor<T>` ↔ `cv::Mat`
- **STL 兼容** — `begin()` / `end()` / `size()` 迭代器接口

---

## 目录结构

```
TensorN/
├── Tensor.hpp                    # 顶层便利头文件
├── CMakeLists.txt                # CMake 构建配置
├── LICENSE                       # MIT 许可证
├── core/
│   ├── core.hpp                  # 主包含头文件
│   ├── tensor.hpp                # Tensor<T> 核心类 + opt<T>
│   ├── einsum.hpp                # Einstein 求和引擎
│   ├── operations.hpp            # 高级张量操作
│   ├── static.hpp                # 序列化 (CSV/JSON/NPY/NPZ)
│   ├── BLAS/
│   │   └── blas_tensor.hpp       # OpenBLAS 加速后端
│   ├── CUDA/
│   │   ├── cuda_tensor.hpp/cu    # CudaTensor<T> 设备端实现
│   │   ├── elementwise.hpp/cu    # CUDA element-wise 算子
│   │   ├── reduction.hpp/cu      # CUDA 归约算子
│   │   ├── matmul.hpp/cu         # CUDA 矩阵乘 (cuBLAS)
│   │   └── convolution.hpp/cu    # CUDA 卷积 (conv2d / conv_transpose2d)
│   └── cnpy/                     # NumPy 文件 I/O (rogersce/cnpy)
├── tests/
│   ├── test_main.cpp             # CPU 算子测试 (Catch2)
│   └── test_cuda.cpp             # CUDA 算子测试 (Catch2)
├── exp1.cpp ~ exp5.cpp           # 示例程序
├── benchmark.cpp                 # 性能基准测试
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
cmake -B build -G Ninja
cmake --build build

# 运行测试
cd build
ctest --output-on-failure

# 或直接运行
./tensorn_tests
./benchmark
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
auto f = a + b;
auto g = exp(a);

// 归约
auto s = sum(a, 1);       // 沿 axis=1 求和
auto m = mean(a);

// 激活函数
auto r = relu(a);
```

### Einstein 求和

支持显式模式、隐式模式、省略号 (`...`) 以及多输入：

```cpp
// 矩阵乘法
auto c = einsum("ij,jk->ik", A, B);

// 迹
auto t = einsum("ii", A);

// 内积
auto d = einsum("i,i", x, y);

// 批量矩阵乘法
auto e = einsum("...ij,...jk->...ik", batch_A, batch_B);

// 三输入缩并
auto f = einsum("ij,jk,kl->il", A, B, C);
```

### CUDA 加速

```cpp
#include "Tensor.hpp"
#include "core/CUDA/cuda_tensor.hpp"
using namespace TensorN::cuda;

// 创建 CUDA 张量
CudaTensor<float> d_a(a);      // 从 Tensor 拷贝到设备
CudaTensor<float> d_b(b);

// CUDA element-wise 运算
add(d_a, d_b, d_c);            // d_c = d_a + d_b
relu(d_a, d_c);                // d_c = relu(d_a)

// CUDA 矩阵乘法 (cuBLAS)
matmul(d_a, d_b, d_c);

// CUDA 归约
float s = sum(d_a);            // 设备归约，返回主机标量

// 拷贝回主机
Tensor<float> result = d_c.to_host();
```

### 序列化

```cpp
// 保存
save_npy(A, "matrix.npy");
save_npz({{"A", A}, {"B", B}}, "data.npz");
save_csv(A, "data.csv");
save_json(A, "data.json");

// 自动检测格式
save(A, "matrix.npy");

// 加载
auto B = load_npy<float>("matrix.npy");
auto C = load_csv<float>("data.csv");
```

---

## 架构

```
Tensor<T> (header-only)
    │
    ├── Einsum 引擎 (纯 C++)  ←── 通用后备
    │
    ├── OpenBLAS 后端          ←── CPU 加速 (BLAS)
    │    └── blas_tensor.hpp
    │
    └── CUDA 后端              ←── GPU 加速 (cuBLAS + CUDA kernels)
         ├── CudaTensor<T>     (设备内存管理)
         ├── elementwise       (逐元素算子)
         ├── reduction         (归约算子)
         ├── matmul            (cuBLAS GEMM)
         └── convolution       (conv2d / conv_transpose2d)
```

所有算子通过三个命名空间暴露：

| 命名空间 | 说明 |
|----------|------|
| `TensorN::` | 通用算子（einsum 引擎） |
| `TensorN::blas::` | OpenBLAS 加速算子 |
| `TensorN::cuda::` | CUDA 加速算子 |

---

## 性能基准

提供三个独立的基准测试程序，采用**控制变量法**分别对比各后端：

| 程序 | 对比项 | 编译器 | 说明 |
|------|--------|--------|------|
| `benchmark_native` | 原生 C++ 基线 | g++ (MSYS2) -O3 | 纯原生实现，无任何加速 |
| `benchmark_blas` | 原生 C++ vs OpenBLAS | g++ (MSYS2) -O3 | 同一编译器下对比 BLAS 加速效果 |
| `benchmark_cuda` | 原生 C++ vs CUDA | MSVC -O3 | 同一编译器下对比 GPU 加速效果 |
| `benchmark` | **三端混合对比** | **MSVC -O3** | **Native / BLAS / CUDA 同一编译器全量对比，运行后自动写入 README 并关机** |

### 测试环境

| 项目 | 配置 |
|------|------|
| CPU | AMD Ryzen 9 5950X 16-Core |
| GPU | NVIDIA GeForce RTX 3090 (24GB) |
| OS | Windows 11 |
| g++ | MSYS2 MinGW-w64 GCC 16.1.0 |
| MSVC | Visual Studio 2022 v17.14 (19.44) |
| OpenBLAS | 0.3.33 |
| CUDA | 12.9.86 |

### Benchmark 1: 原生 C++ 基线 (g++ -O3)

```
Benchmark               Native ms       Status
------------------------------------------------------------
matmul 256x256          1278.462        OK
bmm 16x64x128           610.248         OK
dot 1M                  42.573          OK
outer 500x500           17.369          OK
gram 200x100            300.085         OK
bilinear 200x150        1.895           OK
hadamard 1M             74.223          OK
add 1M                  0.697           OK
relu 500K               0.295           OK
gelu 500K               3.501           OK
sigmoid 500K            4.622           OK
tanh 500K               2.807           OK
exp 500K                4.720           OK
sum 1M                  135.204         OK
sum_axis 100x200        1.492           OK
max 1M                  0.375           OK
min 1M                  0.371           OK
mean 1M                 35.459          OK
norm 1M                 49.848          OK
frobenius 1000x1000     102.532         OK
trace 500x500           0.022           OK
transpose 500x500       12.886          OK
diag 500x500            0.039           OK
diag_matrix 500         0.157           OK
axpy 1M                 0.000           OK
scal 1M                 0.000           OK
var 100K                18.215          OK
stddev 100K             10.277          OK
```

### Benchmark 2: 原生 C++ vs OpenBLAS (g++ -O3)

> 同一编译器、同一优化级别，仅 BLAS 加速为变量

```
Benchmark               Native ms   BLAS ms     BLAS/Native Status
----------------------------------------------------------------------
matmul 256x256          1527.332    53.382      28.6x       OK
bmm 16x64x128           602.111     0.063       9485.6x     OK
dot 1M                  43.989      0.121       364.0x      OK
outer 500x500           25.640      0.086       296.7x      OK
gram 200x100            319.425     79.523      4.0x        OK
bilinear 200x150        3.079       0.001       2101.7x     OK
hadamard 1M             76.546      0.569       134.6x      OK
add 1M                  0.982       -           -           OK
relu 500K               0.339       0.349       1.0x        OK
gelu 500K               4.021       4.872       0.8x        OK
sigmoid 500K            5.848       6.502       0.9x        OK
tanh 500K               2.435       2.467       1.0x        OK
exp 500K                4.741       4.160       1.1x        OK
sum 1M                  39.980      0.493       81.1x       OK
sum_axis 100x200        1.469       0.002       601.7x      OK
max 1M                  0.468       0.460       1.0x        OK
min 1M                  0.461       0.459       1.0x        OK
mean 1M                 22.145      0.373       59.4x       OK
norm 1M                 43.225      0.101       429.7x      OK
frobenius 1000x1000     85.106      0.102       836.0x      OK
trace 500x500           0.012       0.009       1.4x        OK
transpose 500x500       10.837      9.145       1.2x        OK
diag 500x500            0.027       0.013       2.1x        OK
diag_matrix 500         0.025       0.023       1.1x        OK
axpy 1M                 0.000       0.742       -           OK
scal 1M                 0.000       0.764       -           OK
var 100K                9.901       0.073       134.7x      OK
stddev 100K             9.818       0.073       133.6x      OK
```

### Benchmark 3: 原生 C++ vs CUDA (MSVC -O3)

> 同一编译器、同一优化级别，仅 GPU 加速为变量（不含 BLAS）

```
Benchmark               Native ms   CUDA ms     CUDA/Native Status
----------------------------------------------------------------------
matmul 256x256          1798.107    0.050       35606.1x    OK
bmm 16x64x128           907.602     0.010       91676.9x    OK
dot 1M                  69.403      0.059       1168.4x     OK
hadamard 1M             96.569      0.007       13795.6x    OK
add 1M                  0.814       0.009       93.6x       OK
relu 500K               0.367       0.005       73.3x       OK
gelu 500K               2.606       0.006       427.2x      OK
sigmoid 500K            0.870       0.012       75.7x       OK
tanh 500K               2.055       0.006       354.4x      OK
exp 500K                1.071       0.027       40.0x       OK
sum 1M                  38.630      0.080       484.1x      OK
sum_axis 100x200        1.406       0.018       77.7x       OK
max 1M                  0.384       0.080       4.8x        OK
min 1M                  0.365       0.069       5.3x        OK
mean 1M                 35.659      0.065       546.9x      OK
scal 1M                 0.924       0.006       149.0x      OK
softmax 2048x1024       -           1.670       -           OK
conv2d 1x3x32x32        -           0.005       -           OK
argmax 100x200          -           0.017       -           OK
argmin 100x200          -           0.020       -           OK
equal 100K              -           0.003       -           OK
greater 100K            -           0.006       -           OK
convT2d 1x3x16x16       -           0.008       -           OK
```

### Benchmark 4: 三端混合对比 (MSVC -O3)

> 同一编译器 (MSVC -O3)、同一次运行，三个后端全量对比
> 运行完成后自动写入 README 并关机

```
Benchmark               Native ms   BLAS ms     CUDA ms     BLAS/N    CUDA/N    Status
------------------------------------------------------------------------------------------
(运行 benchmark.exe 后自动填充)
```

### 结果分析

**BLAS 加速效果显著**（在大型线性代数/归约场景下）：
- `batched_matmul` 加速 ~9486x，`bilinear` ~2102x，`frobenius_norm` ~836x
- 但对于逐元素操作（relu/gelu/sigmoid/tanh），BLAS 无加速（仍为纯 C++ 循环）

**CUDA GPU 加速效果极致**：
- `batched_matmul` 加速 ~91677x，`matmul` ~35606x，`hadamard` ~13796x
- GPU 的众核并行优势在计算密集型任务上尤为突出
- 小规模逐元素操作（max/min）加速比较小（GPU 发射开销占比高）

**编译器差异**（g++ vs MSVC 原生 C++）：
- MSVC 原生 `matmul` (1798ms) vs g++ (1278ms)，g++ 在此场景快 ~40%
- 但两者在 CUDA 基准中统一使用 MSVC 编译，保证了 GPU vs CPU 对比的一致性

### 构建方式

```bash
# 原生 C++ 基线 + BLAS 对比 (MSYS2 g++)
cmake -B build -G Ninja -DTENSORN_BUILD_CUDA=OFF
cmake --build build --target benchmark_native benchmark_blas

# CUDA 对比 (MSVC)
cmake -B build_cuda -G "NMake Makefiles" -DTENSORN_BUILD_OPENBLAS=OFF
cmake --build build_cuda --target benchmark_cuda

# 三端混合对比 (MSVC -O3) — 运行后自动写入 README 并关机
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target benchmark
.\build\Release\benchmark.exe
```

---

## 许可证

[MIT License](LICENSE) © 2025 TensorN Contributors

内含的 [cnpy](https://github.com/rogersce/cnpy) 库同样基于 MIT 许可证。
