#include "TensorN.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <functional>
#include <algorithm>

#ifdef TENSORN_CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif

static std::mt19937 rng(42);

template <typename T>
TensorN::Tensor<T> random_tensor(const std::vector<size_t>& shape)
{
    size_t total = 1;
    for (auto s : shape) total *= s;
    std::uniform_real_distribution<T> dist(T(-1), T(1));
    TensorN::Tensor<T> t(shape);
    for (size_t i = 0; i < total; ++i)
        t[i] = dist(rng);
    return t;
}

template <typename T>
TensorN::Tensor<T> random_positive(const std::vector<size_t>& shape)
{
    size_t total = 1;
    for (auto s : shape) total *= s;
    std::uniform_real_distribution<T> dist(T(0.01), T(1));
    TensorN::Tensor<T> t(shape);
    for (size_t i = 0; i < total; ++i)
        t[i] = dist(rng);
    return t;
}

class Timer
{
    std::chrono::high_resolution_clock::time_point t0;
public:
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms()
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

#ifdef TENSORN_CUDA_AVAILABLE
class CudaTimer
{
    cudaEvent_t e_start, e_stop;
public:
    CudaTimer()
    {
        cudaEventCreate(&e_start);
        cudaEventCreate(&e_stop);
    }
    ~CudaTimer()
    {
        cudaEventDestroy(e_start);
        cudaEventDestroy(e_stop);
    }
    void start()
    {
        cudaDeviceSynchronize();
        cudaEventRecord(e_start);
    }
    double elapsed_ms()
    {
        cudaEventRecord(e_stop);
        cudaEventSynchronize(e_stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, e_start, e_stop);
        return static_cast<double>(ms);
    }
};
#endif

double bench_cpu(std::function<void()> fn, int warmup, int repeats)
{
    for (int i = 0; i < warmup; ++i) fn();
    Timer t;
    t.start();
    for (int i = 0; i < repeats; ++i) fn();
    return t.elapsed_ms() / repeats;
}

#ifdef TENSORN_CUDA_AVAILABLE
double bench_gpu(std::function<void()> fn, int warmup, int repeats)
{
    for (int i = 0; i < warmup; ++i) fn();
    cudaDeviceSynchronize();
    CudaTimer t;
    t.start();
    for (int i = 0; i < repeats; ++i) fn();
    return t.elapsed_ms() / repeats;
}
#endif

std::string fmt_ms(double ms)
{
    if (ms < 0.001) return "<0.001";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3) << ms;
    return ss.str();
}

std::string fmt_speedup(double base, double val)
{
    if (val <= 0 || base <= 0) return "-";
    double sp = base / val;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << sp << "x";
    return ss.str();
}

void print_section(const std::string& title)
{
    std::cout << "\n  " << title << std::endl;
    std::cout << "  " << std::string(88, '-') << std::endl;
    std::cout << "  " << std::left << std::setw(24) << "Operation"
              << std::right
              << std::setw(14) << "Native(ms)"
              << std::setw(14) << "OpenBLAS(ms)"
              << std::setw(14) << "cuBLAS(ms)"
              << std::setw(12) << "BLAS/Nat"
              << std::setw(12) << "CUDA/Nat"
              << std::endl;
    std::cout << "  " << std::string(88, '-') << std::endl;
}

void print_row(const std::string& name, double nat, double blas, double cuda)
{
    std::string blas_s = (blas > 0) ? fmt_ms(blas) : "N/A";
    std::string cuda_s = (cuda > 0) ? fmt_ms(cuda) : "N/A";
    std::string blas_sp = (blas > 0 && nat > 0) ? fmt_speedup(nat, blas) : "-";
    std::string cuda_sp = (cuda > 0 && nat > 0) ? fmt_speedup(nat, cuda) : "-";

    std::cout << "  " << std::left << std::setw(24) << name
              << std::right
              << std::setw(14) << fmt_ms(nat)
              << std::setw(14) << blas_s
              << std::setw(14) << cuda_s
              << std::setw(12) << blas_sp
              << std::setw(12) << cuda_sp
              << std::endl;
}

int main()
{
    using T = float;
    int warmup = 2;
    int repeats = 5;
    size_t M = 64;
    size_t E = 4096;
    size_t V = 1024;

    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "         TensorN Backend Benchmark: Native C++ vs OpenBLAS vs cuBLAS\n";
    std::cout << "================================================================================\n";
    std::cout << "\n  Matmul: " << M << "x" << M << "  |  Element-wise: " << E
              << " elements  |  Vector: " << V
              << "  |  Warmup: " << warmup << "  |  Repeats: " << repeats << std::endl;

#ifdef TENSORN_CUDA_AVAILABLE
    int dc = 0;
    cudaGetDeviceCount(&dc);
    if (dc > 0)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "  GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    }
#endif

    double nat, blas, cuda;

    // ========================================================================
    // 1. Linear Algebra
    // ========================================================================
    {
        print_section("Linear Algebra (Matrix " + std::to_string(M) + "x" + std::to_string(M) + ")");

        auto A = random_tensor<T>({M, M});
        auto B = random_tensor<T>({M, M});
        auto v1 = random_tensor<T>({V});
        auto v2 = random_tensor<T>({V});

        // matmul
        std::cout << "  > matmul..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::matmul(A, B); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::matmul(A, B); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dA(A), dB(B), dC({M, M});
        cuda = bench_gpu([&]{ TensorN::cuda::matmul_cublas(dA, dB, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("matmul", nat, blas, cuda);

        // gram (X * X^T)
        std::cout << "  > gram..." << std::flush;
        nat = bench_cpu([&]{ auto G = TensorN::gram(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto G = TensorN::blas::gram(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dG({M, M});
        cuda = bench_gpu([&]{ TensorN::cuda::gram(dA, dG); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("gram (X*X^T)", nat, blas, cuda);

        // dot
        std::cout << "  > dot..." << std::flush;
        nat = bench_cpu([&]{ auto d = TensorN::dot(v1, v2); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto d = TensorN::blas::dot(v1, v2); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dv1(v1), dv2(v2);
        cuda = bench_gpu([&]{ TensorN::cuda::dot(dv1, dv2); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("dot (vec " + std::to_string(V) + ")", nat, blas, cuda);

        // outer
        std::cout << "  > outer..." << std::flush;
        {
            auto a = random_tensor<T>({256});
            auto b = random_tensor<T>({256});
            nat = bench_cpu([&]{ auto O = TensorN::outer(a, b); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
            blas = bench_cpu([&]{ auto O = TensorN::blas::outer(a, b); }, warmup, repeats);
#else
            blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
            TensorN::CudaTensor<T> da(a), db(b), dO({256, 256});
            cuda = bench_gpu([&]{ TensorN::cuda::outer(da, db, dO); }, warmup, repeats);
#else
            cuda = -1;
#endif
        }
        std::cout << " done" << std::endl;
        print_row("outer (256)", nat, blas, cuda);

        // bilinear (x^T A y)
        std::cout << "  > bilinear..." << std::flush;
        {
            auto x = random_tensor<T>({M});
            auto y = random_tensor<T>({M});
            nat = bench_cpu([&]{ auto r = TensorN::bilinear(x, A, y); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
            blas = bench_cpu([&]{ auto r = TensorN::blas::bilinear(x, A, y); }, warmup, repeats);
#else
            blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
            TensorN::CudaTensor<T> dx(x), dy(y);
            cuda = bench_gpu([&]{ TensorN::cuda::bilinear(dx, dA, dy); }, warmup, repeats);
#else
            cuda = -1;
#endif
        }
        std::cout << " done" << std::endl;
        print_row("bilinear (x^T A y)", nat, blas, cuda);

        // axpy
        std::cout << "  > axpy..." << std::flush;
        {
            auto x = random_tensor<T>({E});
            auto y = random_tensor<T>({E});
            nat = bench_cpu([&]{
                auto yy = y;
                for (size_t i = 0; i < yy.size(); ++i) yy[i] += T(2) * x[i];
            }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
            blas = bench_cpu([&]{
                auto yy = y;
                TensorN::blas::axpy(T(2), x, yy);
            }, warmup, repeats);
#else
            blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
            TensorN::CudaTensor<T> dx2(x), dy2(y);
            cuda = bench_gpu([&]{ TensorN::cuda::axpy(T(2), dx2, dy2); }, warmup, repeats);
#else
            cuda = -1;
#endif
        }
        std::cout << " done" << std::endl;
        print_row("axpy (" + std::to_string(E) + ")", nat, blas, cuda);

        // trace
        std::cout << "  > trace..." << std::flush;
        nat = bench_cpu([&]{ auto t = TensorN::trace(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto t = TensorN::blas::trace(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::trace(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("trace", nat, blas, cuda);
    }

    // ========================================================================
    // 2. Element-wise Operations
    // ========================================================================
    {
        size_t SZ = E;
        std::vector<size_t> shape = {64, 64};
        print_section("Element-wise (" + std::to_string(SZ) + " elements)");

        auto A = random_tensor<T>(shape);
        auto B = random_tensor<T>(shape);
        auto P = random_positive<T>(shape);

#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dA(A), dB(B), dC(shape);
        TensorN::CudaTensor<T> dP(P);
#endif

        // add
        std::cout << "  > add..." << std::flush;
        nat = bench_cpu([&]{ auto C = A + B; }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::add(A, B); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::add(dA, dB, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("add (A+B)", nat, blas, cuda);

        // hadamard
        std::cout << "  > hadamard..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::hadamard(A, B); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::hadamard(A, B); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::multiply(dA, dB, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("hadamard (A*B)", nat, blas, cuda);

        // scalar multiply
        std::cout << "  > scalar_mul..." << std::flush;
        nat = bench_cpu([&]{ auto C = A * T(3.14); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = A; TensorN::blas::scal(T(3.14), C); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::multiply_scalar(dA, T(3.14), dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("scalar_mul (A*3.14)", nat, blas, cuda);

        // exp
        std::cout << "  > exp..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::math::exp(P); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::exp(P); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::exp(dP, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("exp", nat, blas, cuda);

        // log
        std::cout << "  > log..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::math::log(P); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::log(P); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::log(dP, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("log", nat, blas, cuda);

        // sqrt
        std::cout << "  > sqrt..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::math::sqrt(P); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::sqrt(P); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::sqrt(dP, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("sqrt", nat, blas, cuda);

        // sin
        std::cout << "  > sin..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::math::sin(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::sin(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::sin(dA, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("sin", nat, blas, cuda);

        // cos
        std::cout << "  > cos..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::math::cos(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::cos(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::cos(dA, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("cos", nat, blas, cuda);

        // pow
        std::cout << "  > pow..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::math::apply(P, [](T x){ return std::pow(x, T(2)); }); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::pow(P, T(2)); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::pow(dP, T(2), dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("pow (x^2)", nat, blas, cuda);

        // abs
        std::cout << "  > abs..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::math::apply(A, [](T x){ return std::abs(x); }); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::abs(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::abs(dA, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("abs", nat, blas, cuda);
    }

    // ========================================================================
    // 3. Activation Functions
    // ========================================================================
    {
        std::vector<size_t> shape = {64, 64};
        print_section("Activation Functions (" + std::to_string(E) + " elements)");

        auto A = random_tensor<T>(shape);

#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dA(A), dC(shape);
#endif

        // relu
        std::cout << "  > relu..." << std::flush;
        nat = bench_cpu([&]{
            TensorN::Tensor<T> C(A.shape());
            for (size_t i = 0; i < A.size(); ++i) C[i] = A[i] > T(0) ? A[i] : T(0);
        }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::relu(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::relu(dA, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("relu", nat, blas, cuda);

        // sigmoid
        std::cout << "  > sigmoid..." << std::flush;
        nat = bench_cpu([&]{
            TensorN::Tensor<T> C(A.shape());
            for (size_t i = 0; i < A.size(); ++i) C[i] = T(1) / (T(1) + std::exp(-A[i]));
        }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::sigmoid(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::sigmoid(dA, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("sigmoid", nat, blas, cuda);

        // tanh
        std::cout << "  > tanh..." << std::flush;
        nat = bench_cpu([&]{
            TensorN::Tensor<T> C(A.shape());
            for (size_t i = 0; i < A.size(); ++i) C[i] = std::tanh(A[i]);
        }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::tanh(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::tanh(dA, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("tanh", nat, blas, cuda);

        // gelu
        std::cout << "  > gelu..." << std::flush;
        nat = bench_cpu([&]{
            TensorN::Tensor<T> C(A.shape());
            for (size_t i = 0; i < A.size(); ++i) {
                T x = A[i];
                T a = T(0.7978845608028654) * (x + T(0.044715) * x * x * x);
                C[i] = T(0.5) * x * (T(1) + std::tanh(a));
            }
        }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::gelu(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::gelu(dA, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("gelu", nat, blas, cuda);

        // softmax
        std::cout << "  > softmax..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::softmax(A, 1); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::softmax(A, 1); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::softmax(dA, dC, 1); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("softmax (axis=1)", nat, blas, cuda);
    }

    // ========================================================================
    // 4. Reduction Operations
    // ========================================================================
    {
        std::vector<size_t> shape = {64, 64};
        print_section("Reduction (" + std::to_string(E) + " elements)");

        auto A = random_tensor<T>(shape);
        auto v = random_tensor<T>({V});

#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dA(A), dv(v);
#endif

        // sum
        std::cout << "  > sum..." << std::flush;
        nat = bench_cpu([&]{ auto s = TensorN::sum(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto s = TensorN::blas::sum(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::sum(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("sum", nat, blas, cuda);

        // mean
        std::cout << "  > mean..." << std::flush;
        nat = bench_cpu([&]{ auto m = TensorN::math::mean(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto m = TensorN::blas::mean(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::mean(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("mean", nat, blas, cuda);

        // max
        std::cout << "  > max..." << std::flush;
        nat = bench_cpu([&]{ auto m = *std::max_element(A.begin(), A.end()); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto m = TensorN::blas::max(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::max(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("max", nat, blas, cuda);

        // min
        std::cout << "  > min..." << std::flush;
        nat = bench_cpu([&]{ auto m = *std::min_element(A.begin(), A.end()); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto m = TensorN::blas::min(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::min(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("min", nat, blas, cuda);

        // L2 norm
        std::cout << "  > norm..." << std::flush;
        nat = bench_cpu([&]{ auto n = TensorN::math::norm(v); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto n = TensorN::blas::norm(v); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::norm(dv); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("L2 norm (vec " + std::to_string(V) + ")", nat, blas, cuda);

        // frobenius_norm
        std::cout << "  > frobenius_norm..." << std::flush;
        nat = bench_cpu([&]{ auto n = TensorN::math::frobenius_norm(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto n = TensorN::blas::frobenius_norm(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::frobenius_norm(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("frobenius_norm", nat, blas, cuda);

        // var
        std::cout << "  > variance..." << std::flush;
        nat = bench_cpu([&]{ auto v2 = TensorN::math::var(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto v2 = TensorN::blas::var(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::var(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("variance", nat, blas, cuda);

        // stddev
        std::cout << "  > stddev..." << std::flush;
        nat = bench_cpu([&]{ auto s = TensorN::math::stddev(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto s = TensorN::blas::stddev(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::stddev(dA); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("stddev", nat, blas, cuda);

        // argmax
        std::cout << "  > argmax..." << std::flush;
        nat = bench_cpu([&]{ auto a = TensorN::argmax(A, 1); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto a = TensorN::blas::argmax(A, 1); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ auto a = TensorN::cuda::argmax(dA, 1); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("argmax (axis=1)", nat, blas, cuda);

        // argmin
        std::cout << "  > argmin..." << std::flush;
        nat = bench_cpu([&]{ auto a = TensorN::argmin(A, 1); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto a = TensorN::blas::argmin(A, 1); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ auto a = TensorN::cuda::argmin(dA, 1); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("argmin (axis=1)", nat, blas, cuda);
    }

    // ========================================================================
    // 5. Transpose
    // ========================================================================
    {
        print_section("Transpose (" + std::to_string(M) + "x" + std::to_string(M) + ")");

        auto A = random_tensor<T>({M, M});

        std::cout << "  > transpose..." << std::flush;
        nat = bench_cpu([&]{ auto T2 = TensorN::transpose(A); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto T2 = TensorN::blas::transpose(A); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dA(A), dT({M, M});
        cuda = bench_gpu([&]{ TensorN::cuda::transpose(dA, dT); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("transpose", nat, blas, cuda);
    }

    // ========================================================================
    // 6. Convolution
    // ========================================================================
    {
        size_t cN = 1, cC = 3, cH = 32, cW = 32, cK = 16, kH = 3, kW = 3;
        print_section("Conv2d (input: 1x3x32x32, kernel: 16x3x3x3, stride=1, pad=1)");

        auto input = random_tensor<T>({cN, cC, cH, cW});
        auto weight = random_tensor<T>({cK, cC, kH, kW});
        auto bias = random_tensor<T>({cK});

        // conv2d
        std::cout << "  > conv2d..." << std::flush;
        nat = bench_cpu([&]{ auto O = TensorN::conv2d(input, weight, bias, 1, 1); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto O = TensorN::blas::conv2d(input, weight, bias, 1, 1); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dIn(input), dW(weight), dB(bias);
        TensorN::CudaTensor<T> dOut({cN, cK, cH, cW});
        cuda = bench_gpu([&]{ TensorN::cuda::conv2d(dIn, dW, dB, dOut, 1, 1); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("conv2d", nat, blas, cuda);

        // conv_transpose2d
        std::cout << "  > conv_transpose2d..." << std::flush;
        nat = bench_cpu([&]{ auto O = TensorN::conv_transpose2d(input, weight, bias, 1, 0); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto O = TensorN::blas::conv_transpose2d(input, weight, bias, 1, 0); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        size_t tH = (cH - 1) * 1 + kH;
        size_t tW = (cW - 1) * 1 + kW;
        TensorN::CudaTensor<T> dTOut({cN, cK, tH, tW});
        cuda = bench_gpu([&]{ TensorN::cuda::conv_transpose2d(dIn, dW, dB, dTOut, 1, 0); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("conv_transpose2d", nat, blas, cuda);
    }

    // ========================================================================
    // 7. Comparison Operations
    // ========================================================================
    {
        std::vector<size_t> shape = {64, 64};
        print_section("Comparison (" + std::to_string(E) + " elements)");

        auto A = random_tensor<T>(shape);
        auto B = random_tensor<T>(shape);

        // greater
        std::cout << "  > greater..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::greater(A, B); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::greater(A, B); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        TensorN::CudaTensor<T> dA(A), dB(B);
        TensorN::CudaTensor<int> dC(shape);
        cuda = bench_gpu([&]{ TensorN::cuda::greater(dA, dB, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("greater (A>B)", nat, blas, cuda);

        // equal
        std::cout << "  > equal..." << std::flush;
        nat = bench_cpu([&]{ auto C = TensorN::equal(A, B); }, warmup, repeats);
#if TENSORN_HAS_OPENBLAS
        blas = bench_cpu([&]{ auto C = TensorN::blas::equal(A, B); }, warmup, repeats);
#else
        blas = -1;
#endif
#ifdef TENSORN_CUDA_AVAILABLE
        cuda = bench_gpu([&]{ TensorN::cuda::equal(dA, dB, dC); }, warmup, repeats);
#else
        cuda = -1;
#endif
        std::cout << " done" << std::endl;
        print_row("equal (A==B)", nat, blas, cuda);
    }

    std::cout << "\n  " << std::string(88, '-') << std::endl;
    std::cout << "\n================================================================================\n";
    std::cout << "  Done. Speedup columns show how many times faster the backend is vs Native C++.\n";
    std::cout << "================================================================================\n" << std::endl;

    return 0;
}
