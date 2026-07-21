#include "TensorN.hpp"

int main(int argc, char const *argv[])
{
    using namespace TensorN;

    // save as .pt
    Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
    t.save("example/data.pt");
    t.save("example/data.pth"); // .pth also recognized

    Tensor<int32_t> ti({3}, {10, 20, 30});
    ti.save("example/int_data.pt");

    Tensor<double> td({2, 2}, {1.5, 2.5, 3.5, 4.5});
    td.save("example/double_data.pt");

    // scalar tensor
    Tensor<float> scalar({}, {42.0f});
    scalar.save("example/scalar.pt");

    // load from .pt
    auto t0 = load<float>("example/data.pt");
    auto t1 = load<int32_t>("example/int_data.pt");
    auto t2 = load<double>("example/double_data.pt");
    auto t3 = load<float>("example/scalar.pt");

    std::cout << "load float32:  " << t0 << std::endl;
    std::cout << "load int32:    " << t1 << std::endl;
    std::cout << "load float64:  " << t2 << std::endl;
    std::cout << "load scalar:   " << t3 << std::endl;

    // verify round-trip
    std::cout << "Round-trip check: " << (t == t0 ? "PASS" : "FAIL") << std::endl;

    return 0;
}
