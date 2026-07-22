#include "TensorN.hpp"
#include <iostream>

using namespace TensorN;

int main()
{
    std::cout << "=== exp6: Data I/O (CSV, NPY, NPZ, JSON, .pt) ===\n" << std::endl;

    // Create test data
    Tensor<float> t({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::cout << "Original tensor: " << t << std::endl;

    // 1. CSV
    std::cout << "\n1. CSV save/load:" << std::endl;
    t.save("example/data.csv");
    auto csv_t = load<float>("example/data.csv");
    std::cout << "  loaded: " << csv_t << std::endl;
    std::cout << "  round-trip: " << (t == csv_t ? "PASS" : "FAIL") << std::endl;

    // 2. NPY
    std::cout << "\n2. NPY save/load:" << std::endl;
    t.save("example/data.npy");
    auto npy_t = load<float>("example/data.npy");
    std::cout << "  loaded: " << npy_t << std::endl;
    std::cout << "  round-trip: " << (t == npy_t ? "PASS" : "FAIL") << std::endl;

    // 3. NPZ
    std::cout << "\n3. NPZ save/load:" << std::endl;
    t.save("example/data.npz");
    auto npz_t = load<float>("example/data.npz");
    std::cout << "  loaded: " << npz_t << std::endl;
    std::cout << "  round-trip: " << (t == npz_t ? "PASS" : "FAIL") << std::endl;

    // 4. JSON
    std::cout << "\n4. JSON save/load:" << std::endl;
    t.save("example/data.json");
    auto json_t = load<float>("example/data.json");
    std::cout << "  loaded: " << json_t << std::endl;
    std::cout << "  round-trip: " << (t == json_t ? "PASS" : "FAIL") << std::endl;

    // 5. Custom .pt binary format
    std::cout << "\n5. .pt binary format:" << std::endl;

    // float
    t.save("example/data.pt");
    t.save("example/data.pth");
    auto pt_t = load<float>("example/data.pt");
    std::cout << "  float: " << pt_t << std::endl;
    std::cout << "  round-trip: " << (t == pt_t ? "PASS" : "FAIL") << std::endl;

    // int32
    Tensor<int32_t> ti({3}, {10, 20, 30});
    ti.save("example/int_data.pt");
    auto pti = load<int32_t>("example/int_data.pt");
    std::cout << "  int32: " << pti << std::endl;
    std::cout << "  round-trip: " << (ti == pti ? "PASS" : "FAIL") << std::endl;

    // double
    Tensor<double> td({2, 2}, {1.5, 2.5, 3.5, 4.5});
    td.save("example/double_data.pt");
    auto ptd = load<double>("example/double_data.pt");
    std::cout << "  double: " << ptd << std::endl;
    std::cout << "  round-trip: " << (td == ptd ? "PASS" : "FAIL") << std::endl;

    // scalar
    Tensor<float> scalar({}, {42.0f});
    scalar.save("example/scalar.pt");
    auto pts = load<float>("example/scalar.pt");
    std::cout << "  scalar: " << pts << std::endl;
    std::cout << "  round-trip: " << (scalar == pts ? "PASS" : "FAIL") << std::endl;

    // 6. Format auto-detection
    std::cout << "\n6. Auto-format detection:" << std::endl;
    std::cout << "  .csv  -> csv" << std::endl;
    std::cout << "  .npy  -> npy" << std::endl;
    std::cout << "  .npz  -> npz" << std::endl;
    std::cout << "  .pt   -> pt" << std::endl;
    std::cout << "  .pth  -> pt" << std::endl;
    std::cout << "  .json -> json" << std::endl;

    return 0;
}
