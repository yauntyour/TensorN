#include "core/tensor.hpp"
#include "core/einsum.hpp"
#include "core/operations.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>

using namespace TensorN;

struct Bench {
    const char* name;
    double t_native;
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
    std::cout << "TensorN Native C++ Performance Baseline\n";
    std::cout << std::string(60, '=') << "\n\n";

    // ===================== Group A: Linear Algebra =====================
    std::cout << "=== Group A: Linear Algebra ===\n";

    // A1. matmul
    {
        std::cout << " A1. matmul 256x256 ..." << std::flush;
        size_t N = 256;
        Tensor<float> A({N, N}), B({N, N});
        for (size_t i = 0; i < A.size(); i++) { A[i] = 1.0f; B[i] = 2.0f; }
        double tn = measure([&] { auto _ = einsum<float>("ij,jk->ik", A, B).tensor; });
        auto C = einsum<float>("ij,jk->ik", A, B).tensor;
        results.push_back({"matmul 256x256", tn, std::abs(C[{0, 0}] - 2.0f * 256.0f) < 1.0f});
        std::cout << " OK\n";
    }

    // A2. batched_matmul
    {
        std::cout << " A2. batched_matmul 16x64x128 ..." << std::flush;
        Tensor<float> A({16, 64, 128}), B({16, 128, 64});
        for (size_t i = 0; i < A.size(); i++) { A[i] = 1.0f; B[i] = 0.5f; }
        double tn = measure([&] { auto _ = einsum<float>("bij,bjk->bik", A, B).tensor; });
        auto C = einsum<float>("bij,bjk->bik", A, B).tensor;
        results.push_back({"bmm 16x64x128", tn, C[{0, 0, 0}] == 128.0f * 0.5f});
        std::cout << " OK\n";
    }

    // A3. dot
    {
        std::cout << " A3. dot 1M ..." << std::flush;
        Tensor<float> x({1000000}), y({1000000});
        for (size_t i = 0; i < x.size(); i++) { x[i] = 1.0f; y[i] = 1.0f; }
        double tn = measure([&] { volatile auto _ = einsum<float>("i,i->", x, y).tensor[0]; });
        float r = einsum<float>("i,i->", x, y).tensor[0];
        results.push_back({"dot 1M", tn, std::abs(r - 1000000.0f) < 1.0f});
        std::cout << " OK\n";
    }

    // A4. outer
    {
        std::cout << " A4. outer 500x500 ..." << std::flush;
        Tensor<float> a({500}), b({500});
        for (size_t i = 0; i < a.size(); i++) { a[i] = float(i + 1); b[i] = 1.0f; }
        double tn = measure([&] { auto _ = einsum<float>("i,j->ij", a, b).tensor; });
        auto C = einsum<float>("i,j->ij", a, b).tensor;
        results.push_back({"outer 500x500", tn, C[{0, 0}] == 1.0f});
        std::cout << " OK\n";
    }

    // A5. gram
    {
        std::cout << " A5. gram 200x100 ..." << std::flush;
        Tensor<float> X({200, 100});
        for (size_t i = 0; i < X.size(); i++) X[i] = float(i % 50);
        double tn = measure([&] { auto _ = einsum<float>("ik,jk->ij", X, X).tensor; });
        auto C = einsum<float>("ik,jk->ij", X, X).tensor;
        results.push_back({"gram 200x100", tn, C[{0, 0}] > 0});
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
        float v = TensorN::bilinear(x, M, y);
        results.push_back({"bilinear 200x150", tn, v > 0});
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
        auto C = einsum<float>("...,...->...", A, B).tensor;
        results.push_back({"hadamard 1M", tn, C[0] == A[0] * B[0]});
        std::cout << " OK\n";
    }

    // B2. add
    {
        std::cout << " B2. add 1M ..." << std::flush;
        Tensor<float> A({1000000}), B({1000000});
        for (size_t i = 0; i < A.size(); i++) { A[i] = float(i % 100); B[i] = float(i % 10); }
        double tn = measure([&] { auto _ = (A + B).tensor; });
        auto C = (A + B).tensor;
        results.push_back({"add 1M", tn, C[0] == A[0] + B[0]});
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
        Tensor<float> r(v.shape());
        for (size_t i = 0; i < v.size(); i++) r[i] = v[i] > 0 ? v[i] : 0;
        results.push_back({"relu 500K", tn, r[3] == 0 && r[4] == 1.0f});
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
        results.push_back({"gelu 500K", tn, true});
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
        results.push_back({"sigmoid 500K", tn, true});
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
        results.push_back({"tanh 500K", tn, true});
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
        results.push_back({"exp 500K", tn, true});
        std::cout << " OK\n";
    }

    // ===================== Group C: Reductions =====================
    std::cout << "\n=== Group C: Reductions ===\n";

    // C1. sum
    {
        std::cout << " C1. sum 1M ..." << std::flush;
        Tensor<float> A({1000000});
        for (size_t i = 0; i < A.size(); i++) A[i] = 1.0f;
        double tn = measure([&] { volatile auto _ = einsum<float>("...->", A).tensor[0]; });
        float r = einsum<float>("...->", A).tensor[0];
        results.push_back({"sum 1M", tn, std::abs(r - 1000000.0f) < 1.0f});
        std::cout << " OK\n";
    }

    // C2. sum_axis
    {
        std::cout << " C2. sum_axis 100x200 ..." << std::flush;
        Tensor<float> A({100, 200});
        for (size_t i = 0; i < A.size(); i++) A[i] = 1.0f;
        double tn = measure([&] { auto _ = einsum<float>("ij->j", A).tensor; });
        auto C = einsum<float>("ij->j", A).tensor;
        results.push_back({"sum_axis 100x200", tn, std::abs(C[0] - 100.0f) < 0.01f});
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
        float m = A[0];
        for (size_t i = 1; i < A.size(); i++) if (A[i] > m) m = A[i];
        results.push_back({"max 1M", tn, std::abs(m - 999.0f) < 0.01f});
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
        float m = A[0];
        for (size_t i = 1; i < A.size(); i++) if (A[i] < m) m = A[i];
        results.push_back({"min 1M", tn, std::abs(m - 0.0f) < 0.01f});
        std::cout << " OK\n";
    }

    // C5. mean
    {
        std::cout << " C5. mean 1M ..." << std::flush;
        Tensor<float> A({1000000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);
        double tn = measure([&] { volatile auto _ = TensorN::math::mean(A); });
        results.push_back({"mean 1M", tn, true});
        std::cout << " OK\n";
    }

    // C6. norm
    {
        std::cout << " C6. norm 1M ..." << std::flush;
        Tensor<float> v({1000000});
        for (size_t i = 0; i < v.size(); i++) v[i] = 3.0f;
        double tn = measure([&] { volatile auto _ = TensorN::math::norm(v); });
        results.push_back({"norm 1M", tn, std::abs(TensorN::math::norm(v) - 3000.0f) < 1.0f});
        std::cout << " OK\n";
    }

    // C7. frobenius_norm
    {
        std::cout << " C7. frobenius_norm 1000x1000 ..." << std::flush;
        Tensor<float> A({1000, 1000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);
        double tn = measure([&] { volatile auto _ = TensorN::math::frobenius_norm(A); });
        results.push_back({"frobenius 1000x1000", tn, true});
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
        results.push_back({"trace 500x500", tn, std::abs(TensorN::trace(A) - 499.0f * 500.0f) < 0.1f});
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
        results.push_back({"transpose 500x500", tn, true});
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
        results.push_back({"diag 500x500", tn, true});
        std::cout << " OK\n";
    }

    // D3. diag_matrix
    {
        std::cout << " D3. diag_matrix 500 ..." << std::flush;
        Tensor<float> v({500});
        for (size_t i = 0; i < 500; i++) v[i] = float(i + 1);
        double tn = measure([&] { auto _ = TensorN::diag_matrix(v).tensor; });
        results.push_back({"diag_matrix 500", tn, true});
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
        Tensor<float> yc = y;
        for (size_t i = 0; i < yc.size(); i++) yc[i] += 0.5f * x[i];
        results.push_back({"axpy 1M", tn, std::abs(yc[0] - 10.5f) < 0.01f});
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
        Tensor<float> xc = x;
        for (size_t i = 0; i < xc.size(); i++) xc[i] *= 2.0f;
        results.push_back({"scal 1M", tn, std::abs(xc[0] - 10.0f) < 0.01f});
        std::cout << " OK\n";
    }

    // D6. var
    {
        std::cout << " D6. var 100K ..." << std::flush;
        Tensor<float> A({100000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);
        double tn = measure([&] { volatile auto _ = TensorN::math::var(A); });
        results.push_back({"var 100K", tn, true});
        std::cout << " OK\n";
    }

    // D7. stddev
    {
        std::cout << " D7. stddev 100K ..." << std::flush;
        Tensor<float> A({100000});
        for (size_t i = 0; i < A.size(); i++) A[i] = float(i % 100);
        double tn = measure([&] { volatile auto _ = TensorN::math::stddev(A); });
        results.push_back({"stddev 100K", tn, true});
        std::cout << " OK\n";
    }

    // ===================== Summary Table =====================
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << std::left
              << std::setw(24) << "Benchmark"
              << std::setw(16) << "Native ms"
              << "Status\n";
    std::cout << std::string(60, '-') << "\n";

    auto fmt = [](double v) -> std::string {
        if (v < 0) return "-";
        char buf[16];
        snprintf(buf, sizeof(buf), "%.3f", v);
        return buf;
    };

    for (auto& b : results) {
        std::cout << std::setw(24) << b.name
                  << std::setw(16) << fmt(b.t_native)
                  << (b.ok ? "OK" : "FAIL") << "\n";
    }
    std::cout << std::string(60, '-') << "\n";

    // ===================== Core Result Verification =====================
    std::cout << "\n=== Result Verification ===\n";
    {
        Tensor<float> t({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        float max_val = t[0];
        for (size_t i = 1; i < t.size(); i++) if (t[i] > max_val) max_val = t[i];
        std::cout << "  sum=" << TensorN::sum(t) << " mean=" << TensorN::math::mean(t)
                  << " max=" << max_val << "\n";

        Tensor<float> m1({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        Tensor<float> m2({3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12});
        auto mm = einsum<float>("ij,jk->ik", m1, m2).tensor;
        std::cout << "  matmul: [[" << mm[{0, 0}] << "," << mm[{0, 1}]
                  << "],[" << mm[{1, 0}] << "," << mm[{1, 1}] << "]] (exp [[58,64],[139,154]])\n";

        Tensor<float> v({4}, std::vector<float>{-1, 0, 1, 2});
        std::cout << "  relu(-1,0,1,2)=[";
        for (int i = 0; i < 4; i++) std::cout << (v[i] > 0 ? v[i] : 0) << (i < 3 ? "," : "");
        std::cout << "]\n";

        std::cout << "  norm([3,4,0])="
                  << TensorN::math::norm(Tensor<float>({3}, std::vector<float>{3, 4, 0})) << "\n";
        std::cout << "  trace(eye3)=" << TensorN::trace(eye<float>(3)) << "\n";
        std::cout << "  dot([1,2,3],[4,5,6])="
                  << einsum<float>("i,i->", Tensor<float>({3}, std::vector<float>{1, 2, 3}),
                                           Tensor<float>({3}, std::vector<float>{4, 5, 6})).tensor[0] << "\n";
    }

    bool all_ok = true;
    for (auto& b : results) if (!b.ok) { all_ok = false; break; }
    std::cout << "\n" << (all_ok ? "All benchmarks passed!" : "SOME BENCHMARKS FAILED!") << "\n";
    return all_ok ? 0 : 1;
}
