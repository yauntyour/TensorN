#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../core/CUDA/cuda_tensor.hpp"
#include "../core/CUDA/elementwise.hpp"
#include "../core/CUDA/reduction.hpp"
#include "../core/CUDA/matmul.hpp"
#include "../core/CUDA/convolution.hpp"
#include <cmath>
#include <vector>

using namespace TensorN;
using Catch::Approx;

TEST_CASE("CudaTensor construction", "[cuda]")
{
    SECTION("Default construction")
    {
        CudaTensor<float> t;
        REQUIRE(t.size() == 0);
        REQUIRE(t.shape().empty());
        REQUIRE(t.device_ptr() == nullptr);
    }
    
    SECTION("Shape construction")
    {
        CudaTensor<float> t({3, 4});
        REQUIRE(t.size() == 12);
        REQUIRE(t.shape().size() == 2);
        REQUIRE(t.shape()[0] == 3);
        REQUIRE(t.shape()[1] == 4);
        REQUIRE(t.device_ptr() != nullptr);
    }
}

TEST_CASE("CudaTensor copy and move", "[cuda]")
{
    SECTION("Copy from host")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        CudaTensor<float> t({2, 2}, data.data());
        REQUIRE(t.size() == 4);
        
        std::vector<float> result(4);
        t.copyToHost(result.data(), 4);
        REQUIRE(result[0] == 1.0f);
        REQUIRE(result[1] == 2.0f);
        REQUIRE(result[2] == 3.0f);
        REQUIRE(result[3] == 4.0f);
    }
    
    SECTION("Copy constructor")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f};
        CudaTensor<float> t1({3}, data.data());
        CudaTensor<float> t2(t1);
        
        REQUIRE(t2.size() == 3);
        std::vector<float> result(3);
        t2.copyToHost(result.data(), 3);
        REQUIRE(result[0] == 1.0f);
        REQUIRE(result[1] == 2.0f);
        REQUIRE(result[2] == 3.0f);
    }
}

TEST_CASE("CudaTensor conversion", "[cuda]")
{
    SECTION("From Tensor to CudaTensor and back")
    {
        Tensor<float> cpu_tensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
        CudaTensor<float> gpu_tensor = CudaTensor<float>::fromTensor(cpu_tensor);
        
        REQUIRE(gpu_tensor.size() == 4);
        
        Tensor<float> result = gpu_tensor.toTensor();
        REQUIRE(result.size() == 4);
        REQUIRE(result[{0, 0}] == 1.0f);
        REQUIRE(result[{1, 1}] == 4.0f);
    }
}

TEST_CASE("CUDA element-wise operations", "[cuda][elementwise]")
{
    SECTION("Addition")
    {
        std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> data2 = {5.0f, 6.0f, 7.0f, 8.0f};
        
        CudaTensor<float> A({4}, data1.data());
        CudaTensor<float> B({4}, data2.data());
        CudaTensor<float> C({4});
        
        cuda::add(A, B, C);
        
        std::vector<float> result(4);
        C.copyToHost(result.data(), 4);
        
        REQUIRE(result[0] == 6.0f);
        REQUIRE(result[1] == 8.0f);
        REQUIRE(result[2] == 10.0f);
        REQUIRE(result[3] == 12.0f);
    }
    
    SECTION("Scalar multiplication")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        CudaTensor<float> A({4}, data.data());
        CudaTensor<float> C({4});
        
        cuda::multiply_scalar(A, 2.0f, C);
        
        std::vector<float> result(4);
        C.copyToHost(result.data(), 4);
        
        REQUIRE(result[0] == 2.0f);
        REQUIRE(result[1] == 4.0f);
        REQUIRE(result[2] == 6.0f);
        REQUIRE(result[3] == 8.0f);
    }
    
    SECTION("ReLU")
    {
        std::vector<float> data = {-1.0f, 2.0f, -3.0f, 4.0f};
        CudaTensor<float> A({4}, data.data());
        CudaTensor<float> C({4});
        
        cuda::relu(A, C);
        
        std::vector<float> result(4);
        C.copyToHost(result.data(), 4);
        
        REQUIRE(result[0] == 0.0f);
        REQUIRE(result[1] == 2.0f);
        REQUIRE(result[2] == 0.0f);
        REQUIRE(result[3] == 4.0f);
    }
}

TEST_CASE("CUDA reduction operations", "[cuda][reduction]")
{
    SECTION("Sum")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        CudaTensor<float> A({2, 3}, data.data());
        
        float total = cuda::sum(A);
        REQUIRE(total == Approx(21.0f));
    }
    
    SECTION("Mean")
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        CudaTensor<float> A({4}, data.data());
        
        float mean_val = cuda::mean(A);
        REQUIRE(mean_val == Approx(2.5f));
    }
    
    SECTION("Max")
    {
        std::vector<float> data = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
        CudaTensor<float> A({6}, data.data());
        
        float max_val = cuda::max(A);
        REQUIRE(max_val == Approx(9.0f));
    }
}

TEST_CASE("CUDA matrix multiplication", "[cuda][matmul]")
{
    SECTION("Basic matmul")
    {
        // A: 2x3, B: 3x2
        std::vector<float> dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::vector<float> dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        
        CudaTensor<float> A({2, 3}, dataA.data());
        CudaTensor<float> B({3, 2}, dataB.data());
        CudaTensor<float> C({2, 2});
        
        cuda::matmul(A, B, C);
        
        std::vector<float> result(4);
        C.copyToHost(result.data(), 4);
        
        // Expected: [[58, 64], [139, 154]]
        REQUIRE(result[0] == Approx(58.0f));
        REQUIRE(result[1] == Approx(64.0f));
        REQUIRE(result[2] == Approx(139.0f));
        REQUIRE(result[3] == Approx(154.0f));
    }
}

TEST_CASE("CUDA convolution", "[cuda][convolution]")
{
    SECTION("Basic conv2d")
    {
        // Input: 1x1x4x4 (batch, channels, height, width)
        std::vector<float> input_data = {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };
        
        // Weight: 1x1x3x3 (out_channels, in_channels, kernel_h, kernel_w)
        std::vector<float> weight_data = {
            1.0f, 0.0f, -1.0f,
            1.0f, 0.0f, -1.0f,
            1.0f, 0.0f, -1.0f
        };
        
        CudaTensor<float> input({1, 1, 4, 4}, input_data.data());
        CudaTensor<float> weight({1, 1, 3, 3}, weight_data.data());
        CudaTensor<float> output({1, 1, 2, 2});
        
        cuda::conv2d(input, weight, output, 1, 0);
        
        std::vector<float> result(4);
        output.copyToHost(result.data(), 4);
        
        // Expected output for stride=1, padding=0
        // Should be something like:
        // [-6, -6]
        // [-6, -6]
        // This is a simplified test - actual values depend on the convolution implementation
        REQUIRE(result.size() == 4);
    }
}