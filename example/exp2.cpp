#include "TensorN.hpp"
#include <iostream>

using namespace TensorN;

int main()
{
    std::cout << "=== exp2: Element-wise Operations & Activations ===\n" << std::endl;

    // 1. Element-wise tensor-tensor ops
    std::cout << "1. Element-wise tensor ops:" << std::endl;
    Tensor<int> A({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<int> B({2, 3}, {10, 20, 30, 40, 50, 60});

    std::cout << "  A = " << A << std::endl;
    std::cout << "  B = " << B << std::endl;
    std::cout << "  A + B = " << (A + B) << std::endl;
    std::cout << "  A - B = " << (A - B) << std::endl;
    std::cout << "  A * B = " << (A * B) << std::endl;

    // 2. Element-wise scalar ops
    std::cout << "\n2. Element-wise scalar ops:" << std::endl;
    std::cout << "  A + 10 = " << (A + 10) << std::endl;
    std::cout << "  A - 1  = " << (A - 1) << std::endl;
    std::cout << "  A * 2  = " << (A * 2) << std::endl;
    std::cout << "  A / 2  = " << (A / 2) << std::endl;

    // 3. Hadamard product
    std::cout << "\n3. Hadamard product:" << std::endl;
    auto H = hadamard(A, B);
    std::cout << "  hadamard(A, B) = " << H << std::endl;

    // 4. In-place operations
    std::cout << "\n4. In-place operations:" << std::endl;
    Tensor<double> X({3}, {1.0, 2.0, 3.0});
    std::cout << "  X = " << X << std::endl;
    X.add_(10.0);
    std::cout << "  X.add_(10) = " << X << std::endl;
    X.mul_(2.0);
    std::cout << "  X.mul_(2)  = " << X << std::endl;
    X.sub_(5.0);
    std::cout << "  X.sub_(5)  = " << X << std::endl;
    X.div_(3.0);
    std::cout << "  X.div_(3)  = " << X << std::endl;

    // 5. apply_ with lambda
    std::cout << "\n5. apply_ (in-place transform):" << std::endl;
    Tensor<int> T({3}, {1, 2, 3});
    T.apply_([](int x) { return x * x; });
    std::cout << "  apply_(x*x) on {1,2,3}: " << T << std::endl;

    // 6. Math functions
    std::cout << "\n6. Math functions:" << std::endl;
    Tensor<double> V({3}, {1.0, 2.0, 3.0});
    using namespace math;
    std::cout << "  exp({1,2,3})  = " << exp(V) << std::endl;
    std::cout << "  log({1,2,3})  = " << log(V) << std::endl;
    std::cout << "  sqrt({1,2,3}) = " << sqrt(V) << std::endl;
    std::cout << "  sin({1,2,3})  = " << sin(V) << std::endl;
    std::cout << "  cos({1,2,3})  = " << cos(V) << std::endl;

    // 7. Softmax
    std::cout << "\n7. Softmax:" << std::endl;
    Tensor<double> logits({3}, {1.0, 2.0, 3.0});
    auto sm = softmax(logits);
    std::cout << "  softmax({1,2,3}) = " << sm << std::endl;
    // Verify sums to ~1
    double s = 0;
    for (size_t i = 0; i < 3; ++i) s += sm.tensor[i];
    std::cout << "  sum = " << s << "  (should be 1.0)" << std::endl;

    // 2D softmax
    Tensor<double> M2d({2, 3}, {1.0, 2.0, 3.0, 1.0, 2.0, 3.0});
    auto sm2d = softmax(M2d, 1);
    std::cout << "  softmax(axis=1) = " << sm2d << std::endl;

    // 8. Comparison ops
    std::cout << "\n8. Comparison:" << std::endl;
    Tensor<int> P({3}, {1, 5, 3});
    Tensor<int> Q({3}, {1, 2, 7});
    std::cout << "  P = " << P << std::endl;
    std::cout << "  Q = " << Q << std::endl;
    std::cout << "  equal(P,Q)   = " << equal(P, Q) << std::endl;
    std::cout << "  greater(P,Q) = " << greater(P, Q) << std::endl;

    return 0;
}
