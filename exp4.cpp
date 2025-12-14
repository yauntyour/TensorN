#include "Tensor.hpp"

// 爱因斯坦缩并的性能测试函数
template <typename T>
void einsum_performance_test()
{
    using namespace TensorN;
    std::cout << "Einsum Performance Test for type: " << typeid(T).name() << std::endl;

    // 创建测试张量
    Tensor<T> A({4, 4});
    Tensor<T> B({4, 4});
    Tensor<T> C({4, 4});

    // 填充随机数据
    std::fill(A.begin(), A.end(), T(1));
    std::fill(B.begin(), B.end(), T(2));
    std::fill(C.begin(), C.end(), T(3));

    auto start = std::chrono::high_resolution_clock::now();

    auto result = einsum<T>("ij,jk,kl->il", A, B, C);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Triple contraction: " << duration1.count() << " μs" << std::endl;

    std::cout << "Result: " << result << std::endl;
}

template <typename T>
void ellipsis_examples()
{
    using namespace TensorN;
    std::cout << "\n=== Ellipsis Examples for type: " << typeid(T).name() << " ===" << std::endl;

    Tensor<T> A({2, 3, 4});
    Tensor<T> B({2, 4, 5});
    std::fill(A.begin(), A.end(), T(1));
    std::fill(B.begin(), B.end(), T(2));

    auto result = einsum<T>("ijk,ikl->ijl", A, B);
    std::cout << "A shape: (2, 3, 4)" << std::endl;
    std::cout << "B shape: (2, 4, 5)" << std::endl;
    std::cout << "Result shape: " << result.tensor.shape()[0] << ", "
              << result.tensor.shape()[1] << ", " << result.tensor.shape()[2] << std::endl;
    std::cout << "Result[0,0,0] = " << result.tensor[{0, 0, 0}] << std::endl;
}

int main(int argc, char const *argv[])
{
    einsum_performance_test<int>();
    einsum_performance_test<size_t>();
    einsum_performance_test<float>();
    einsum_performance_test<double>();

    ellipsis_examples<int>();
    ellipsis_examples<size_t>();
    ellipsis_examples<float>();
    ellipsis_examples<double>();

    return 0;
}
