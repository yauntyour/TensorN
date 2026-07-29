#include "TensorN.hpp"
#include <iostream>
#include <cstdio>
#include <string>

using namespace TensorN;

int main()
{
    std::cout << "=== exp9: Multi-Tensor .pt/.pth Format Save & Load ===\n" << std::endl;

    Tensor<float> W({4, 3}, {0.1f, 0.2f, 0.3f,
                              0.4f, 0.5f, 0.6f,
                              0.7f, 0.8f, 0.9f,
                              1.0f, 1.1f, 1.2f});
    Tensor<float> bias({3}, {0.01f, 0.02f, 0.03f});
    Tensor<float> embedding({5, 3}, {1.0f, 2.0f, 3.0f,
                                     4.0f, 5.0f, 6.0f,
                                     7.0f, 8.0f, 9.0f,
                                     10.0f, 11.0f, 12.0f,
                                     13.0f, 14.0f, 15.0f});

    std::cout << "1. Save multiple tensors:" << std::endl;
    std::vector<std::pair<std::string, Tensor<float>>> tensors = {
        {"layers.0.weight", W},
        {"layers.0.bias", bias},
        {"embedding.weight", embedding}
    };

    save_pt_multi(tensors, "exp9_model.pt");
    std::cout << "  Saved " << tensors.size() << " tensors to exp9_model.pt" << std::endl;

    std::cout << "\n2. List tensors in file:" << std::endl;
    auto names = pt_list_tensors("exp9_model.pt");
    for (const auto &n : names)
        std::cout << "  - " << n << std::endl;

    std::cout << "\n3. Load all tensors:" << std::endl;
    auto loaded = load_pt_multi<float>("exp9_model.pt");
    std::cout << "  Loaded " << loaded.size() << " tensors:" << std::endl;
    for (const auto &[name, tensor] : loaded)
    {
        std::cout << "  " << name << ": shape={";
        auto &s = tensor.shape();
        for (size_t i = 0; i < s.size(); ++i)
        {
            if (i > 0) std::cout << ", ";
            std::cout << s[i];
        }
        std::cout << "}, size=" << tensor.size() << std::endl;
    }

    std::cout << "\n4. Round-trip verification:" << std::endl;
    std::cout << "  weight match: " << (W == loaded["layers.0.weight"] ? "YES" : "NO") << std::endl;
    std::cout << "  bias match:   " << (bias == loaded["layers.0.bias"] ? "YES" : "NO") << std::endl;
    std::cout << "  embed match:  " << (embedding == loaded["embedding.weight"] ? "YES" : "NO") << std::endl;

    std::cout << "\n5. Load single tensor from multi-tensor file:" << std::endl;
    auto single = load<float>("exp9_model.pt");
    std::cout << "  First tensor shape={";
    auto &s = single.shape();
    for (size_t i = 0; i < s.size(); ++i)
    {
        if (i > 0) std::cout << ", ";
        std::cout << s[i];
    }
    std::cout << "}" << std::endl;

    std::cout << "\n6. Different data types:" << std::endl;
    Tensor<int32_t> int_t({3}, {100, 200, 300});
    Tensor<double> dbl_t({2, 2}, {1.5, 2.5, 3.5, 4.5});

    std::vector<std::pair<std::string, Tensor<int32_t>>> int_tensors = {
        {"int_tensor", int_t}
    };
    save_pt_multi(int_tensors, "exp9_int.pt");
    auto int_loaded = load_pt_multi<int32_t>("exp9_int.pt");
    std::cout << "  int32 match: " << (int_t == int_loaded["int_tensor"] ? "YES" : "NO") << std::endl;

    std::vector<std::pair<std::string, Tensor<double>>> dbl_tensors = {
        {"dbl_tensor", dbl_t}
    };
    save_pt_multi(dbl_tensors, "exp9_double.pt");
    auto dbl_loaded = load_pt_multi<double>("exp9_double.pt");
    std::cout << "  double match: " << (dbl_t == dbl_loaded["dbl_tensor"] ? "YES" : "NO") << std::endl;

    std::cout << "\n7. .pth extension (same format):" << std::endl;
    save_pt_multi(tensors, "exp9_model.pth");
    auto pth_loaded = load_pt_multi<float>("exp9_model.pth");
    std::cout << "  Loaded " << pth_loaded.size() << " tensors from .pth" << std::endl;
    std::cout << "  weight match: " << (W == pth_loaded["layers.0.weight"] ? "YES" : "NO") << std::endl;

    std::cout << "\n8. Cleanup temporary files." << std::endl;
    std::remove("exp9_model.pt");
    std::remove("exp9_model.pth");
    std::remove("exp9_int.pt");
    std::remove("exp9_double.pt");

    std::cout << "\nAll multi-tensor .pt tests passed!" << std::endl;

    return 0;
}
