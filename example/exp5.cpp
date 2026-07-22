#include "TensorN.hpp"
#include <iostream>

using namespace TensorN;

int main()
{
    std::cout << "=== exp5: Reductions & Statistics ===\n" << std::endl;

    Tensor<double> M({3, 4}, {1.0,  2.0,  3.0,  4.0,
                               5.0,  6.0,  7.0,  8.0,
                               9.0, 10.0, 11.0, 12.0});
    std::cout << "M (3x4) = " << M << std::endl;

    // 1. Sum
    std::cout << "\n1. Sum:" << std::endl;
    double total = sum(M);
    std::cout << "  sum(all) = " << total << "  (expected 78)" << std::endl;
    auto s0 = sum(M, 0);
    std::cout << "  sum(axis=0) = " << s0 << std::endl;
    auto s1 = sum(M, 1);
    std::cout << "  sum(axis=1) = " << s1 << std::endl;

    // 2. Mean
    std::cout << "\n2. Mean:" << std::endl;
    using namespace math;
    std::cout << "  mean(all) = " << mean(M) << "  (expected 6.5)" << std::endl;

    // 3. Max / Min
    std::cout << "\n3. Max & Min:" << std::endl;
    std::cout << "  max = " << *std::max_element(M.begin(), M.end()) << std::endl;
    std::cout << "  min = " << *std::min_element(M.begin(), M.end()) << std::endl;

    // 4. Argmax / Argmin
    std::cout << "\n4. Argmax & Argmin:" << std::endl;
    auto amax = argmax(M, 1);
    auto amin = argmin(M, 0);
    std::cout << "  argmax(axis=1) = " << amax << std::endl;
    std::cout << "  argmin(axis=0) = " << amin << std::endl;

    // 5. Norms
    std::cout << "\n5. Norms:" << std::endl;
    Tensor<double> v({3}, {3.0, 4.0, 0.0});
    std::cout << "  v = " << v << std::endl;
    std::cout << "  norm(v) [L2] = " << norm(v) << "  (expected 5)" << std::endl;
    std::cout << "  frobenius_norm(M) = " << frobenius_norm(M) << std::endl;

    // 6. Variance & Stddev
    std::cout << "\n6. Variance & Stddev:" << std::endl;
    std::cout << "  var(M) = " << var(M) << std::endl;
    std::cout << "  stddev(M) = " << stddev(M) << std::endl;

    // 7. Conv2d
    std::cout << "\n7. Conv2d (N=1, C=1, H=3, W=3, K=1, k=2x2):" << std::endl;
    Tensor<double> input({1, 1, 3, 3}, {1.0, 2.0, 3.0,
                                         4.0, 5.0, 6.0,
                                         7.0, 8.0, 9.0});
    Tensor<double> weight({1, 1, 2, 2}, {1.0, 0.0,
                                          0.0, 1.0});
    Tensor<double> bias({1}, {0.0});
    auto conv_out = conv2d(input, weight, bias, 1, 0);
    std::cout << "  " << conv_out << std::endl;

    // 8. ConvTranspose2d
    std::cout << "\n8. ConvTranspose2d:" << std::endl;
    Tensor<double> ct_in({1, 1, 2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor<double> ct_w({1, 1, 2, 2}, {1.0, 0.0, 0.0, 1.0});
    Tensor<double> ct_b({1}, {0.0});
    auto ct_out = conv_transpose2d(ct_in, ct_w, ct_b, 1, 0);
    std::cout << "  " << ct_out << std::endl;

    return 0;
}
