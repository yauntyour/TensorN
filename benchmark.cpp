#include "Tensor.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>

#ifdef TENSORN_CUDA_AVAILABLE
#include "core/CUDA/cuda_tensor.hpp"
#include "core/CUDA/elementwise.hpp"
#include "core/CUDA/reduction.hpp"
#include "core/CUDA/matmul.hpp"
#include "core/CUDA/convolution.hpp"
#endif

using namespace TensorN;

struct Bench {
    const char* name;
    double t_native;
    double t_blas;
    double t_cuda;
    bool ok;
};

std::vector<Bench> results;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

template <typename F>
static double measure(F&& fn) {
    fn();
    auto t0 = now_ms();
    fn();
    return now_ms() - t0;
}

int main() {
    std::cout << "TensorN Unified Performance Benchmark\n";
    std::cout << "Native C++ vs OpenBLAS vs CUDA\n";
    std::cout << std::string(80, '=') << "\n\n";

    // ===================== Group A: Linear Algebra =====================
    std::cout << "=== Group A: Linear Algebra ===\n";

    // A1. matmul
    {
        std::cout << " A1. matmul 256x256 ..." << std::flush;
        size_t N = 256;
        Tensor<float> A({N, N}), B({N, N});
        for (size_t i = 0; i < A.size(); i++) { A[i] = 1.0f; B[i] = 2.0f; }

        double tn = measure([&] { auto _ = einsum<float>("ij,jk->ik", A, B).tensor; });
        double tb = measure([&] { auto _ = blas::matmul(A, B); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A), dB(B), dC({N, N});
            tc = measure([&] { cuda::matmul(dA, dB, dC); });
            auto Cc = dC.toTensor();
            auto Cb = blas::matmul(A, B);
            ok = std::abs(Cc[{0, 0}] - Cb[{0, 0}]) < 0.5f;
        }
#endif
        results.push_back({"matmul 256x256", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // A2. batched_matmul
    {
        std::cout << " A2. batched_matmul 16x64x128 ..." << std::flush;
        Tensor<float> A({16, 64, 128}), B({16, 128, 64});
        for (size_t i = 0; i < A.size(); i++) { A[i] = 1.0f; B[i] = 0.5f; }

        double tn = measure([&] { auto _ = einsum<float>("bij,bjk->bik", A, B).tensor; });
        double tb = measure([&] { auto _ = blas::batched_matmul(A, B); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A), dB(B), dC({16, 64, 64});
            tc = measure([&] { cuda::batched_matmul(dA, dB, dC); });
            auto Cc = dC.toTensor();
            auto Cb = blas::batched_matmul(A, B);
            ok = std::abs(Cc[{0, 0, 0}] - Cb[{0, 0, 0}]) < 1.0f;
        }
#endif
        results.push_back({"bmm 16x64x128", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // A3. dot
    {
        std::cout << " A3. dot 1M ..." << std::flush;
        Tensor<float> x({1000000}), y({1000000});
        for (size_t i = 0; i < x.size(); i++) { x[i] = 1.0f; y[i] = 1.0f; }

        double tn = measure([&] { volatile auto _ = einsum<float>("i,i->", x, y).tensor[0]; });
        double tb = measure([&] { volatile auto _ = blas::dot(x, y); });
        double tc = -1;
        float r_native = einsum<float>("i,i->", x, y).tensor[0];
        float r_blas = blas::dot(x, y);

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dX(x), dY(y);
            tc = measure([&] { volatile auto _ = cuda::dot(dX, dY); });
        }
#endif
        results.push_back({"dot 1M", tn, tb, tc, std::abs(r_native - r_blas) < 1.0f});
        std::cout << " OK\n";
    }

    // A4. outer
    {
        std::cout << " A4. outer 500x500 ..." << std::flush;
        Tensor<float> a({500}), b({500});
        for (size_t i = 0; i < a.size(); i++) { a[i] = float(i + 1); b[i] = 1.0f; }

        double tn = measure([&] { auto _ = einsum<float>("i,j->ij", a, b).tensor; });
        double tb = measure([&] { auto _ = blas::outer(a, b); });
        auto Cb = blas::outer(a, b);
        results.push_back({"outer 500x500", tn, tb, -1, Cb[{0, 0}] == 1.0f});
        std::cout << " OK\n";
    }

    // A5. gram
    {
        std::cout << " A5. gram 200x100 ..." << std::flush;
        Tensor<float> X({200, 100});
        for (size_t i = 0; i < X.size(); i++) X[i] = float(i % 50);

        double tn = measure([&] { auto _ = einsum<float>("ik,jk->ij", X, X).tensor; });
        double tb = measure([&] { auto _ = blas::gram(X); });
        auto Cb = blas::gram(X);
        results.push_back({"gram 200x100", tn, tb, -1, Cb[{0, 0}] > 0});
        std::cout << " OK\n";
    }

    // A6. bilinear
    {
        std::cout << " A6. bilinear 200x150 ..." << std::flush;
        Tensor<float> x({200}), M({200, 150}), y({150});
        for (size_t i = 0; i < x.size(); i++) x[i] = 1.0f;
        for (size_t i = 0; i < M.size(); i++) M[i] = 1.0f;
        for (size_t i = 0; i < y.size(); i++) y[i] = 2.0f;

        double tn = measure([&] { volatile auto _ = TensorN::bilinear(x, M, y); });
        double tb = measure([&] { volatile auto _ = blas::bilinear(x, M, y); });
        float vn = TensorN::bilinear(x, M, y);
        float vb = blas::bilinear(x, M, y);
        results.push_back({"bilinear 200x150", tn, tb, -1, std::abs(vn - vb) < 0.1f});
        std::cout << " OK\n";
    }

    // ===================== Group B: Element-wise =====================
    std::cout << "\n=== Group B: Element-wise ===\n";

    // B1. hadamard
    {
        std::cout << " B1. hadamard 1M ..." << std::flush;
        Tensor<float> A({1000000}), B({1000000});
        for (size_t i = 0; i < A.size(); i++) { A[i] = float(i % 10); B[i] = 2.0f; }

        double tn = measure([&] { auto _ = einsum<float>("...,...->...", A, B).tensor; });
        double tb = measure([&] { auto _ = blas::hadamard(A, B); });
        double tc = -1;
        auto Cb = blas::hadamard(A, B);
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A), dB(B), dC({1000000});
            tc = measure([&] { cuda::multiply(dA, dB, dC); });
            auto Cc = dC.toTensor();
            ok = std::abs(Cc[0] - Cb[0]) < 0.01f;
        }
#endif
        results.push_back({"hadamard 1M", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // B2. add (element-wise)
    {
        std::cout << " B2. add 1M ..." << std::flush;
        Tensor<float> A({1000000}), B({1000000});
        for (size_t i = 0; i < A.size(); i++) { A[i] = float(i % 100); B[i] = float(i % 10); }

        double tn = measure([&] { auto _ = (A + B).tensor; });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A), dB(B), dC({1000000});
            tc = measure([&] { cuda::add(dA, dB, dC); });
            auto Cc = dC.toTensor();
            auto Cn = (A + B).tensor;
            ok = std::abs(Cc[0] - Cn[0]) < 0.01f;
        }
#endif
        results.push_back({"add 1M", tn, -1, tc, ok});
        std::cout << " OK\n";
    }

    // B3. relu
    {
        std::cout << " B3. relu 500K ..." << std::flush;
        Tensor<float> v({500000});
        for (size_t i = 0; i < v.size(); i++) v[i] = float(int(i % 7) - 3);

        double tn = measure([&] {
            Tensor<float> r(v.shape());
            for (size_t i = 0; i < v.size(); i++) r[i] = v[i] > 0 ? v[i] : 0;
        });
        double tb = measure([&] { auto _ = blas::relu(v); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(v), dC({500000});
            tc = measure([&] { cuda::relu(dA, dC); });
            auto Cc = dC.toTensor();
            auto Cb = blas::relu(v);
            ok = std::abs(Cc[0] - Cb[0]) < 0.01f;
        }
#endif
        results.push_back({"relu 500K", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // B4. gelu
    {
        std::cout << " B4. gelu 500K ..." << std::flush;
        Tensor<float> v({500000});
        for (size_t i = 0; i < v.size(); i++) v[i] = float(int(i % 7) - 3);

        double tn = measure([&] {
            Tensor<float> r(v.shape());
            for (size_t i = 0; i < v.size(); i++) {
                float x = v[i];
                float a = 0.7978845608028654f * (x + 0.044715f * x * x * x);
                r[i] = 0.5f * x * (1.0f + std::tanh(a));
            }
        });
        double tb = measure([&] { auto _ = blas::gelu(v); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(v), dC({500000});
            tc = measure([&] { cuda::gelu(dA, dC); });
            auto Cc = dC.toTensor();
            auto Cb = blas::gelu(v);
            ok = std::abs(Cc[0] - Cb[0]) < 0.01f;
        }
#endif
        results.push_back({"gelu 500K", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // B5. sigmoid
    {
        std::cout << " B5. sigmoid 500K ..." << std::flush;
        Tensor<float> v({500000});
        for (size_t i = 0; i < v.size(); i++) v[i] = float(int(i % 7) - 3);

        double tn = measure([&] {
            Tensor<float> r(v.shape());
            for (size_t i = 0; i < v.size(); i++) r[i] = 1.0f / (1.0f + std::exp(-v[i]));
        });
        double tb = measure([&] { auto _ = blas::sigmoid(v); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(v), dC({500000});
            tc = measure([&] { cuda::sigmoid(dA, dC); });
            auto Cc = dC.toTensor();
            auto Cb = blas::sigmoid(v);
            ok = std::abs(Cc[0] - Cb[0]) < 0.01f;
        }
#endif
        results.push_back({"sigmoid 500K", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // B6. tanh
    {
        std::cout << " B6. tanh 500K ..." << std::flush;
        Tensor<float> v({500000});
        for (size_t i = 0; i < v.size(); i++) v[i] = float(int(i % 7) - 3);

        double tn = measure([&] {
            Tensor<float> r(v.shape());
            for (size_t i = 0; i < v.size(); i++) r[i] = std::tanh(v[i]);
        });
        double tb = measure([&] { auto _ = blas::tanh(v); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(v), dC({500000});
            tc = measure([&] { cuda::tanh(dA, dC); });
            auto Cc = dC.toTensor();
            auto Cb = blas::tanh(v);
            ok = std::abs(Cc[0] - Cb[0]) < 0.01f;
        }
#endif
        results.push_back({"tanh 500K", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // B7. exp
    {
        std::cout << " B7. exp 500K ..." << std::flush;
        Tensor<float> v({500000});
        for (size_t i = 0; i < v.size(); i++) v[i] = float(int(i % 5) - 2) * 0.5f;

        double tn = measure([&] {
            Tensor<float> r(v.shape());
            for (size_t i = 0; i < v.size(); i++) r[i] = std::exp(v[i]);
        });
        double tb = measure([&] { auto _ = blas::exp(v); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(v), dC({500000});
            tc = measure([&] { cuda::exp(dA, dC); });
            auto Cc = dC.toTensor();
            auto Cb = blas::exp(v);
            ok = std::abs(Cc[0] - Cb[0]) < 0.01f;
        }
#endif
        results.push_back({"exp 500K", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // ===================== Group C: Reductions =====================
    std::cout << "\n=== Group C: Reductions ===\n";

    // C1. sum (total)
    {
        std::cout << " C1. sum 1M ..." << std::flush;
        Tensor<float> A({1000000});
        for (size_t i = 0; i < A.size(); i++) A[i] = 1.0f;

        double tn = measure([&] { volatile auto _ = einsum<float>("...->", A).tensor[0]; });
        double tb = measure([&] { volatile auto _ = blas::sum(A); });
        double tc = -1;
        float vr = einsum<float>("...->", A).tensor[0];

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A);
            tc = measure([&] { volatile auto _ = cuda::sum(dA); });
        }
#endif
        results.push_back({"sum 1M", tn, tb, tc, std::abs(vr - 1000000.0f) < 1.0f});
        std::cout << " OK\n";
    }

    // C2. sum_axis
    {
        std::cout << " C2. sum_axis 100x200 ..." << std::flush;
        Tensor<float> A({100, 200});
        for (size_t i = 0; i < A.size(); i++) A[i] = 1.0f;

        double tn = measure([&] { auto _ = einsum<float>("ij->j", A).tensor; });
        double tb = measure([&] { auto _ = blas::sum(A, 0); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A);
            tc = measure([&] { auto _ = cuda::sum_axis(dA, 0); });
            auto Cc = cuda::sum_axis(dA, 0).toTensor();
            auto Cb = blas::sum(A, 0);
            ok = std::abs(Cc[0] - Cb[0]) < 0.01f;
        }
#endif
        results.push_back({"sum_axis 100x200", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // C3. max
    {
        std::cout << " C3. max 1M ..." << std::flush;
        Tensor<float> A({1000000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 1000);

        double tn = measure([&] {
            float m = A[0];
            for (size_t i = 1; i < A.size(); i++) if (A[i] > m) m = A[i];
            volatile auto _ = m;
        });
        double tb = measure([&] { volatile auto _ = blas::max(A); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A);
            tc = measure([&] { volatile auto _ = cuda::max(dA); });
            ok = std::abs(cuda::max(dA) - blas::max(A)) < 0.01f;
        }
#endif
        results.push_back({"max 1M", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // C4. min
    {
        std::cout << " C4. min 1M ..." << std::flush;
        Tensor<float> A({1000000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 1000);

        double tn = measure([&] {
            float m = A[0];
            for (size_t i = 1; i < A.size(); i++) if (A[i] < m) m = A[i];
            volatile auto _ = m;
        });
        double tb = measure([&] { volatile auto _ = blas::min(A); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A);
            tc = measure([&] { volatile auto _ = cuda::min(dA); });
            ok = std::abs(cuda::min(dA) - blas::min(A)) < 0.01f;
        }
#endif
        results.push_back({"min 1M", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // C5. mean
    {
        std::cout << " C5. mean 1M ..." << std::flush;
        Tensor<float> A({1000000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);

        double tn = measure([&] { volatile auto _ = TensorN::math::mean(A); });
        double tb = measure([&] { volatile auto _ = blas::mean(A); });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(A);
            tc = measure([&] { volatile auto _ = cuda::mean(dA); });
            ok = std::abs(cuda::mean(dA) - blas::mean(A)) < 0.1f;
        }
#endif
        results.push_back({"mean 1M", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // C6. norm (L2)
    {
        std::cout << " C6. norm 1M ..." << std::flush;
        Tensor<float> v({1000000});
        for (size_t i = 0; i < v.size(); i++) v[i] = 3.0f;

        double tn = measure([&] { volatile auto _ = TensorN::math::norm(v); });
        double tb = measure([&] { volatile auto _ = blas::norm(v); });
        results.push_back({"norm 1M", tn, tb, -1,
            std::abs(blas::norm(v) - 3000.0f) < 1.0f});
        std::cout << " OK\n";
    }

    // C7. frobenius_norm
    {
        std::cout << " C7. frobenius_norm 1000x1000 ..." << std::flush;
        Tensor<float> A({1000, 1000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);

        double tn = measure([&] { volatile auto _ = TensorN::math::frobenius_norm(A); });
        double tb = measure([&] { volatile auto _ = blas::frobenius_norm(A); });
        results.push_back({"frobenius 1000x1000", tn, tb, -1, true});
        std::cout << " OK\n";
    }

    // C8. trace
    {
        std::cout << " C8. trace 500x500 ..." << std::flush;
        Tensor<float> A({500, 500});
        for (size_t i = 0; i < 500; i++)
            for (size_t j = 0; j < 500; j++)
                A[{i, j}] = float(i + j);

        double tn = measure([&] { volatile auto _ = TensorN::trace(A); });
        double tb = measure([&] { volatile auto _ = blas::trace(A); });
        results.push_back({"trace 500x500", tn, tb, -1,
            std::abs(TensorN::trace(A) - blas::trace(A)) < 0.1f});
        std::cout << " OK\n";
    }

    // ===================== Group D: Matrix/Other =====================
    std::cout << "\n=== Group D: Matrix/Other ===\n";

    // D1. transpose
    {
        std::cout << " D1. transpose 500x500 ..." << std::flush;
        Tensor<float> A({500, 500});
        for (size_t i = 0; i < 500; i++)
            for (size_t j = 0; j < 500; j++)
                A[{i, j}] = float(i * 500 + j);

        double tn = measure([&] { auto _ = TensorN::transpose(A).tensor; });
        double tb = measure([&] { auto _ = blas::transpose(A); });
        results.push_back({"transpose 500x500", tn, tb, -1, true});
        std::cout << " OK\n";
    }

    // D2. diag
    {
        std::cout << " D2. diag 500x500 ..." << std::flush;
        Tensor<float> A({500, 500});
        for (size_t i = 0; i < 500; i++)
            for (size_t j = 0; j < 500; j++)
                A[{i, j}] = float(i + j);

        double tn = measure([&] { auto _ = TensorN::diag(A).tensor; });
        double tb = measure([&] { auto _ = blas::diag(A); });
        results.push_back({"diag 500x500", tn, tb, -1, true});
        std::cout << " OK\n";
    }

    // D3. diag_matrix
    {
        std::cout << " D3. diag_matrix 500 ..." << std::flush;
        Tensor<float> v({500});
        for (size_t i = 0; i < 500; i++) v[i] = float(i + 1);

        double tn = measure([&] { auto _ = TensorN::diag_matrix(v).tensor; });
        double tb = measure([&] { auto _ = blas::diag_matrix(v); });
        results.push_back({"diag_matrix 500", tn, tb, -1, true});
        std::cout << " OK\n";
    }

    // D4. axpy
    {
        std::cout << " D4. axpy 1M ..." << std::flush;
        Tensor<float> x({1000000}), y({1000000});
        for (size_t i = 0; i < x.size(); i++) { x[i] = 1.0f; y[i] = 10.0f; }

        double tn = measure([&] {
            Tensor<float> yc = y;
            for (size_t i = 0; i < yc.size(); i++) yc[i] += 0.5f * x[i];
        });
        double tb = measure([&] {
            Tensor<float> yc = y;
            blas::axpy(0.5f, x, yc);
        });
        Tensor<float> yc = y;
        blas::axpy(0.5f, x, yc);
        results.push_back({"axpy 1M", tn, tb, -1, std::abs(yc[0] - 10.5f) < 0.01f});
        std::cout << " OK\n";
    }

    // D5. scal
    {
        std::cout << " D5. scal 1M ..." << std::flush;
        Tensor<float> x({1000000});
        for (size_t i = 0; i < x.size(); i++) x[i] = 5.0f;

        double tn = measure([&] {
            Tensor<float> xc = x;
            for (size_t i = 0; i < xc.size(); i++) xc[i] *= 2.0f;
        });
        double tb = measure([&] {
            Tensor<float> xc = x;
            blas::scal(2.0f, xc);
        });
        double tc = -1;
        bool ok = true;

#ifdef TENSORN_CUDA_AVAILABLE
        {
            CudaTensor<float> dA(x), dC({1000000});
            tc = measure([&] { cuda::multiply_scalar(dA, 2.0f, dC); });
            auto Cc = dC.toTensor();
            ok = std::abs(Cc[0] - 10.0f) < 0.01f;
        }
#endif
        results.push_back({"scal 1M", tn, tb, tc, ok});
        std::cout << " OK\n";
    }

    // D6. var
    {
        std::cout << " D6. var 100K ..." << std::flush;
        Tensor<float> A({100000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);

        double tn = measure([&] { volatile auto _ = TensorN::math::var(A); });
        double tb = measure([&] { volatile auto _ = blas::var(A); });
        results.push_back({"var 100K", tn, tb, -1, true});
        std::cout << " OK\n";
    }

    // D7. stddev
    {
        std::cout << " D7. stddev 100K ..." << std::flush;
        Tensor<float> A({100000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);

        double tn = measure([&] { volatile auto _ = TensorN::math::stddev(A); });
        double tb = measure([&] { volatile auto _ = blas::stddev(A); });
        results.push_back({"stddev 100K", tn, tb, -1, true});
        std::cout << " OK\n";
    }

    // ===================== Group E: CUDA-Specific =====================
#ifdef TENSORN_CUDA_AVAILABLE
    std::cout << "\n=== Group E: CUDA-Specific ===\n";

    // E1. softmax
    {
        std::cout << " E1. softmax 2048x1024 ..." << std::flush;
        Tensor<float> A({2048, 1024});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 10);

        CudaTensor<float> dA(A), dC({2048, 1024});
        double tc = measure([&] { cuda::softmax(dA, dC, 1); });
        auto Cc = dC.toTensor();
        float row_sum = Cc[{0, 0}];
        for (size_t j = 1; j < 1024; j++) row_sum += Cc[{0, j}];
        results.push_back({"softmax 2048x1024", -1, -1, tc, std::abs(row_sum - 1.0f) < 0.01f});
        std::cout << " OK\n";
    }

    // E2. conv2d
    {
        std::cout << " E2. conv2d 1x3x32x32 k=16x3x3x3 ..." << std::flush;
        size_t N = 1, C = 3, H = 32, W = 32;
        size_t K = 16, kH = 3, kW = 3;
        Tensor<float> input({N, C, H, W});
        Tensor<float> weight({K, C, kH, kW});
        Tensor<float> bias({K});
        for (size_t i = 0; i < input.size(); i++) input[i] = 1.0f;
        for (size_t i = 0; i < weight.size(); i++) weight[i] = 1.0f;
        for (size_t i = 0; i < bias.size(); i++) bias[i] = 1.0f;

        size_t oH = (H - kH) / 1 + 1;
        size_t oW = (W - kW) / 1 + 1;
        CudaTensor<float> dInput(input), dWeight(weight), dBias(bias), dOut({N, K, oH, oW});
        double tc = measure([&] { cuda::conv2d(dInput, dWeight, dBias, dOut, 1, 0); });
        auto Cc = dOut.toTensor();
        float expected = float(C * kH * kW) + 1.0f;
        results.push_back({"conv2d 1x3x32x32", -1, -1, tc,
            std::abs(Cc[{0, 0, 0, 0}] - expected) < 1.0f});
        std::cout << " OK\n";
    }

    // E3. argmax
    {
        std::cout << " E3. argmax 100x200 ..." << std::flush;
        Tensor<float> A({100, 200});
        for (size_t i = 0; i < 100; i++)
            for (size_t j = 0; j < 200; j++)
                A[{i, j}] = float(i * 200 + j);

        CudaTensor<float> dA(A);
        double tc = measure([&] { auto _ = cuda::argmax(dA, 1); });
        auto Cc = cuda::argmax(dA, 1).toTensor();
        bool ok = true;
        for (int64_t i = 0; i < 100; i++) ok &= (Cc[i] == 199);
        results.push_back({"argmax 100x200", -1, -1, tc, ok});
        std::cout << " OK\n";
    }

    // E4. argmin
    {
        std::cout << " E4. argmin 100x200 ..." << std::flush;
        Tensor<float> A({100, 200});
        for (size_t i = 0; i < 100; i++)
            for (size_t j = 0; j < 200; j++)
                A[{i, j}] = float(i * 200 + j);

        CudaTensor<float> dA(A);
        double tc = measure([&] { auto _ = cuda::argmin(dA, 1); });
        auto Cc = cuda::argmin(dA, 1).toTensor();
        bool ok = true;
        for (int64_t i = 0; i < 100; i++) ok &= (Cc[i] == 0);
        results.push_back({"argmin 100x200", -1, -1, tc, ok});
        std::cout << " OK\n";
    }

    // E5. element-wise: equal
    {
        std::cout << " E5. equal 100K ..." << std::flush;
        Tensor<float> A({100000}), B({100000});
        for (size_t i = 0; i < A.size(); i++) { A[i] = float(i % 10); B[i] = float(i % 10); }

        CudaTensor<float> dA(A), dB(B);
        CudaTensor<int> dC({100000});
        double tc = measure([&] { cuda::equal(dA, dB, dC); });
        auto Cc = dC.toTensor();
        bool ok = true;
        for (size_t i = 0; i < 100000; i++) ok &= (Cc[i] == 1);
        results.push_back({"equal 100K", -1, -1, tc, ok});
        std::cout << " OK\n";
    }

    // E6. element-wise: greater
    {
        std::cout << " E6. greater 100K ..." << std::flush;
        Tensor<float> A({100000}), B({100000});
        for (size_t i = 0; i < A.size(); i++) { A[i] = float(i % 10); B[i] = 5.0f; }

        CudaTensor<float> dA(A), dB(B);
        CudaTensor<int> dC({100000});
        double tc = measure([&] { cuda::greater(dA, dB, dC); });
        auto Cc = dC.toTensor();
        int cnt = 0;
        for (size_t i = 0; i < 100000; i++) cnt += Cc[i];
        results.push_back({"greater 100K", -1, -1, tc, cnt > 0});
        std::cout << " OK\n";
    }

    // E7. conv_transpose2d
    {
        std::cout << " E7. conv_transpose2d 1x3x16x16 k=16x3x3x3 ..." << std::flush;
        size_t N = 1, C = 3, H = 16, W = 16;
        size_t K = 16, kH = 3, kW = 3;
        Tensor<float> input({N, C, H, W});
        Tensor<float> weight({C, K, kH, kW});
        Tensor<float> bias({K});
        for (size_t i = 0; i < input.size(); i++) input[i] = 1.0f;
        for (size_t i = 0; i < weight.size(); i++) weight[i] = 0.1f;
        for (size_t i = 0; i < bias.size(); i++) bias[i] = 0.0f;

        size_t oH = (H - 1) * 1 + kH;
        size_t oW = (W - 1) * 1 + kW;
        CudaTensor<float> dInput(input), dWeight(weight), dBias(bias), dOut({N, K, oH, oW});
        double tc = measure([&] { cuda::conv_transpose2d(dInput, dWeight, dBias, dOut, 1, 0); });
        auto Cc = dOut.toTensor();
        results.push_back({"convT2d 1x3x16x16", -1, -1, tc, Cc[{0, 0, 0, 0}] != 0});
        std::cout << " OK\n";
    }
#endif

    // ===================== Summary Table =====================
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << std::left
              << std::setw(24) << "Benchmark"
              << std::setw(12) << "Native ms"
              << std::setw(12) << "BLAS ms"
              << std::setw(12) << "CUDA ms"
              << std::setw(10) << "B/N"
              << std::setw(10) << "C/N"
              << "Status\n";
    std::cout << std::string(80, '-') << "\n";

    auto fmt = [](double v) -> std::string {
        if (v < 0) return "-";
        char buf[16];
        snprintf(buf, sizeof(buf), "%.3f", v);
        return buf;
    };

    auto ratio = [](double num, double den) -> std::string {
        if (num <= 0 || den <= 0) return "-";
        char buf[16];
        snprintf(buf, sizeof(buf), "%.1fx", den / num);
        return buf;
    };

    for (auto& b : results) {
        std::cout << std::setw(24) << b.name
                  << std::setw(12) << fmt(b.t_native)
                  << std::setw(12) << fmt(b.t_blas)
                  << std::setw(12) << fmt(b.t_cuda)
                  << std::setw(10) << ratio(b.t_blas, b.t_native)
                  << std::setw(10) << ratio(b.t_cuda, b.t_native)
                  << (b.ok ? "OK" : "FAIL") << "\n";
    }
    std::cout << std::string(80, '-') << "\n";

    // ===================== Core Result Verification =====================
    std::cout << "\n=== Result Verification ===\n";
    {
        Tensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        std::cout << "  sum=" << blas::sum(t) << " mean=" << blas::mean(t)
                  << " max=" << blas::max(t) << " min=" << blas::min(t) << "\n";

        Tensor<float> m1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        Tensor<float> m2({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12});
        auto mm = blas::matmul(m1, m2);
        std::cout << "  matmul: [[" << mm[{0, 0}] << "," << mm[{0, 1}]
                  << "],[" << mm[{1, 0}] << "," << mm[{1, 1}] << "]] (exp [[58,64],[139,154]])\n";

        Tensor<float> v({4}, std::vector<float>{-1, 0, 1, 2});
        std::cout << "  relu(-1,0,1,2)=" << blas::relu(v) << "\n";
        std::cout << "  gelu(-1,0,1,2)=" << blas::gelu(v) << "\n";

        std::cout << "  norm([3,4,0])="
                  << blas::norm(Tensor<float>({3}, std::vector<float>{3, 4, 0})) << "\n";
        std::cout << "  trace(eye3)=" << blas::trace(eye<float>(3)) << "\n";

        std::cout << "  dot([1,2,3],[4,5,6])="
                  << blas::dot(Tensor<float>({3}, std::vector<float>{1, 2, 3}),
                               Tensor<float>({3}, std::vector<float>{4, 5, 6})) << "\n";
    }

    bool all_ok = true;
    for (auto& b : results) if (!b.ok) { all_ok = false; break; }
    std::cout << "\n" << (all_ok ? "All benchmarks passed!" : "SOME BENCHMARKS FAILED!")
              << "\n";
    return all_ok ? 0 : 1;
}
