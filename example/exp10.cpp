// ============================================================================
// exp10: 低精度数据类型 (TF32 / FP16 / BF16 / FP8)
//
// 演示 TensorN 为加速计算提供的数据类型：
//   *  TensorN::half / fp16   — IEEE 754 binary16
//   *  TensorN::bfloat16 / bf16 — Google bfloat16
//   *  TensorN::tf32          — TF32 存储类型 (10 位尾数)
//   *  TensorN::fp8_e4m3      — NVIDIA FP8 (4 指数 / 3 尾数)
//   *  TensorN::fp8_e5m2      — NVIDIA FP8 (5 指数 / 2 尾数)
//
// CPU 端：全部类型可直接用于 Tensor<T> 与 einsum 运算；
// CUDA 端：CudaTensor<T> 经 cublasGemmEx 走张量核心加速。
// ============================================================================
#include "TensorN.hpp"
#include <iostream>
#include <iomanip>
#include <string>

using namespace TensorN;
// CUDA (cuda_fp16.h) 在全局命名空间声明 `half`，OpenBLAS 声明 `bfloat16`，
// 与 using 指令引入的 TensorN::half / TensorN::bfloat16 产生二义。
// 使用 fp16 / bf16 别名（无冲突），或写全限定名 TensorN::half。
using TensorN::fp16;
using TensorN::bf16;
using TensorN::tf32;
using TensorN::fp8_e4m3;
using TensorN::fp8_e5m2;

#ifdef TENSORN_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

int main()
{
    std::cout << "=== TensorN 低精度数据类型 ===\n\n";

    // ---------------------------------------------------------------
    // 1. CPU：类型基础用法
    // ---------------------------------------------------------------
    std::cout << "-- CPU 基本用法 --\n";
    fp16 h = 0.1f;                     // FP16
    bf16 b = 0.1f;                     // BF16
    tf32 t = 0.1f;                     // TF32
    fp8_e4m3 e4 = 0.1f;                // FP8 E4M3
    fp8_e5m2 e5 = 0.1f;                // FP8 E5M2

    std::cout << "0.1f 在各类中的表示:\n";
    std::cout << "  float32  : " << std::setprecision(8) << 0.1f << "\n";
    std::cout << "  fp16     : " << float(h) << "\n";
    std::cout << "  bf16     : " << float(b) << "\n";
    std::cout << "  tf32     : " << float(t) << "\n";
    std::cout << "  fp8_e4m3 : " << float(e4) << "\n";
    std::cout << "  fp8_e5m2 : " << float(e5) << "\n\n";

    // 类型自省
    std::cout << "dtype_of<fp16>     = " << name_of<fp16>() << " (" << int(dtype_of<fp16>()) << ")\n";
    std::cout << "dtype_of<bf16>     = " << name_of<bf16>() << " (" << int(dtype_of<bf16>()) << ")\n";
    std::cout << "dtype_of<tf32>     = " << name_of<tf32>() << " (" << int(dtype_of<tf32>()) << ")\n";
    std::cout << "dtype_of<fp8_e4m3> = " << name_of<fp8_e4m3>() << " (" << int(dtype_of<fp8_e4m3>()) << ")\n";
    std::cout << "dtype_of<fp8_e5m2> = " << name_of<fp8_e5m2>() << " (" << int(dtype_of<fp8_e5m2>()) << ")\n\n";

    // ---------------------------------------------------------------
    // 2. CPU：Tensor<T> + einsum 矩阵乘法
    // ---------------------------------------------------------------
    std::cout << "-- CPU 矩阵乘法 (einsum) --\n";
    Tensor<fp16> Ah({64, 64});
    Tensor<fp16> Bh({64, 64});
    for (size_t i = 0; i < Ah.size(); ++i)
    {
        Ah[i] = fp16(static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f);
        Bh[i] = fp16(static_cast<float>(static_cast<int>(i % 13) - 6) / 8.0f);
    }
    auto Ch = matmul(Ah, Bh);
    std::cout << "fp16     matmul[0,0] = " << float(Ch.tensor[{0, 0}]) << "\n";

    Tensor<fp8_e4m3> Ae({64, 64});
    Tensor<fp8_e4m3> Be({64, 64});
    for (size_t i = 0; i < Ae.size(); ++i)
    {
        Ae[i] = fp8_e4m3(static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f);
        Be[i] = fp8_e4m3(static_cast<float>(static_cast<int>(i % 13) - 6) / 8.0f);
    }
    auto Ce = matmul(Ae, Be);
    std::cout << "fp8_e4m3 matmul[0,0] = " << float(Ce.tensor[{0, 0}]) << "\n\n";

    // ---------------------------------------------------------------
    // 3. CPU：.pt 文件读写 (支持全部新类型)
    // ---------------------------------------------------------------
    std::cout << "-- .pt 序列化 --\n";
    auto X = ones<fp16>({2, 3}).mul_(fp16(1.5f));
    X.save("exp10_half.pt");
    auto X2 = load<fp16>("exp10_half.pt");
    std::cout << "fp16 .pt roundtrip: " << float(X2[{0, 0}]) << " (期望 1.5)\n\n";

    // ---------------------------------------------------------------
    // 4. CUDA：张量核心加速
    // ---------------------------------------------------------------
#ifdef TENSORN_CUDA_AVAILABLE
    std::cout << "-- CUDA 张量核心 (cublasGemmEx) --\n";

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")\n";

    const size_t M = 128, K = 128, N = 128;

    // 通用填充：值域 [-1, 1] 的分数，避免 unsigned 下溢
    auto fill_value = [](size_t i, int mod, int shift) {
        return static_cast<float>(static_cast<int>(i % mod) - shift) / 8.0f;
    };

    // FP16 matmul (FP32 累加)
    {
        auto A = zeros<fp16>({M, K});
        auto B = zeros<fp16>({K, N});
        for (size_t i = 0; i < A.size(); ++i)
        {
            A[i] = fp16(fill_value(i, 17, 8));
            B[i] = fp16(fill_value(i, 13, 6));
        }
        auto da = CudaTensor<fp16>(A);
        auto db = CudaTensor<fp16>(B);
        CudaTensor<fp16> dc({M, N});
        cuda::matmul(da, db, dc);
        auto C = dc.toTensor();
        std::cout << "fp16   matmul[0,0] = " << float(C[{0, 0}]) << "\n";
    }

    // BF16 matmul (FP32 累加)
    {
        auto A = zeros<bf16>({M, K});
        auto B = zeros<bf16>({K, N});
        for (size_t i = 0; i < A.size(); ++i)
        {
            A[i] = bf16(fill_value(i, 17, 8));
            B[i] = bf16(fill_value(i, 13, 6));
        }
        auto da = CudaTensor<bf16>(A);
        auto db = CudaTensor<bf16>(B);
        CudaTensor<bf16> dc({M, N});
        cuda::matmul(da, db, dc);
        auto C = dc.toTensor();
        std::cout << "bf16   matmul[0,0] = " << float(C[{0, 0}]) << "\n";
    }

    // TF32 matmul (Tensor<Tf32> 存储，张量核心计算)
    {
        auto A = zeros<tf32>({M, K});
        auto B = zeros<tf32>({K, N});
        for (size_t i = 0; i < A.size(); ++i)
        {
            A[i] = tf32(fill_value(i, 17, 8));
            B[i] = tf32(fill_value(i, 13, 6));
        }
        auto da = CudaTensor<tf32>(A);
        auto db = CudaTensor<tf32>(B);
        CudaTensor<tf32> dc({M, N});
        cuda::matmul(da, db, dc);
        auto C = dc.toTensor();
        std::cout << "tf32   matmul[0,0] = " << float(C[{0, 0}]) << "\n";
    }

    // float matmul 运行时启用 TF32 (Ampere+)
    {
        cuda::set_tf32(true);
        std::cout << "tf32 enabled for float matmul: " << (cuda::tf32_enabled() ? "yes" : "no") << "\n";
        auto A = zeros<float>({M, K});
        auto B = zeros<float>({K, N});
        for (size_t i = 0; i < A.size(); ++i)
        {
            A[i] = fill_value(i, 17, 8);
            B[i] = fill_value(i, 13, 6);
        }
        auto da = CudaTensor<float>(A);
        auto db = CudaTensor<float>(B);
        CudaTensor<float> dc({M, N});
        cuda::matmul(da, db, dc);
        auto C = dc.toTensor();
        std::cout << "float(TF32) matmul[0,0] = " << float(C[{0, 0}]) << "\n";
        cuda::set_tf32(false);
    }

    // FP8 matmul (需要 SM 8.9+ / CUDA 12+ 的硬件与 cuBLAS 支持)
    if (prop.major > 8 || (prop.major == 8 && prop.minor >= 9))
    {
        auto A = zeros<fp8_e4m3>({M, K});
        auto B = zeros<fp8_e4m3>({K, N});
        for (size_t i = 0; i < A.size(); ++i)
        {
            A[i] = fp8_e4m3(fill_value(i, 17, 8));
            B[i] = fp8_e4m3(fill_value(i, 13, 6));
        }
        try
        {
            auto da = CudaTensor<fp8_e4m3>(A);
            auto db = CudaTensor<fp8_e4m3>(B);
            CudaTensor<fp8_e4m3> dc({M, N});
            cuda::matmul(da, db, dc);
            auto C = dc.toTensor();
            std::cout << "fp8_e4m3 matmul[0,0] = " << float(C[{0, 0}]) << "\n";
        }
        catch (const std::exception &e)
        {
            std::cout << "fp8_e4m3 不可用: " << e.what() << "\n";
        }
    }
    else
    {
        std::cout << "fp8_e4m3: 需要 SM 8.9+，跳过\n";
    }

    // GPU 逐元素运算 (fp16)
    {
        auto A = zeros<fp16>({16});
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = fp16(static_cast<float>(i));
        auto da = CudaTensor<fp16>(A);
        CudaTensor<fp16> dc(A.shape());
        cuda::relu(da, dc);
        auto C = dc.toTensor();
        std::cout << "fp16 relu[4] = " << float(C[4]) << "\n";
        std::cout << "fp16 sum = " << float(cuda::sum(da)) << " (期望 120)\n";
    }

    std::cout << "\n";
#else
    std::cout << "CUDA 不可用，跳过 GPU 部分\n";
#endif

    std::cout << "=== 完成 ===\n";
    return 0;
}
