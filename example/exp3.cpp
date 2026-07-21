#include "Tensor.hpp"

int main(int argc, char const *argv[])
{
    using namespace TensorN;
    // 保存
    Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6});
    t.save("example/data.csv"); // 自动识别
    t.save("example/data.npy");
    t.save("example/data.npz");
    t.save("example/data.json");

    // 加载
    auto t0 = load<float>("example/data.csv");
    auto t1 = load<float>("example/data.npy");
    auto t2 = load<float>("example/data.npz");
    auto t3 = load<float>("example/data.json");

    std::cout << "load form csv: " << t0 << std::endl;
    std::cout << "load form npy: " << t1 << std::endl;
    std::cout << "load form npz: " << t2 << std::endl;
    std::cout << "load form json: " << t3 << std::endl;
    return 0;
}
