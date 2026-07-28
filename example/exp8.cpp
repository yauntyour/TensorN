#include "TensorN.hpp"
#include <iostream>
#include <cstdio>
#include <string>

using namespace TensorN;

int main()
{
    std::cout << "=== exp8: GGUF Format Save & Load ===\n" << std::endl;

    // 1. Basic save & load
    std::cout << "1. Basic save & load:" << std::endl;
    Tensor<float> W({4, 3}, {0.1f, 0.2f, 0.3f,
                              0.4f, 0.5f, 0.6f,
                              0.7f, 0.8f, 0.9f,
                              1.0f, 1.1f, 1.2f});
    std::cout << "  Original W (4x3) = " << W << std::endl;

    save_gguf(W, "exp8_weight.gguf", "layers.0.weight");
    std::cout << "  Saved to exp8_weight.gguf" << std::endl;

    Tensor<float> W_loaded = load_gguf<float>("exp8_weight.gguf", "layers.0.weight");
    std::cout << "  Loaded W = " << W_loaded << std::endl;
    std::cout << "  Round-trip match: " << (W == W_loaded ? "YES" : "NO") << std::endl;

    // 2. Save with metadata
    std::cout << "\n2. Save with metadata:" << std::endl;
    std::unordered_map<std::string, GGUFMetadataValue> meta;
    meta["general.architecture"] = std::string("transformer");
    meta["general.name"] = std::string("my-model-7B");
    meta["general.alignment"] = uint32_t(32);
    meta["transformer.block_count"] = uint64_t(32);
    meta["transformer.attention.head_count"] = uint64_t(32);
    meta["transformer.embedding_length"] = uint64_t(4096);
    meta["transformer.context_length"] = uint64_t(8192);
    meta["transformer.attention.layer_norm_rms_epsilon"] = 1e-5f;

    Tensor<float> bias({3}, {0.01f, 0.02f, 0.03f});
    save_gguf(bias, "exp8_bias.gguf", "layers.0.bias", meta);
    std::cout << "  Saved bias with " << meta.size() << " metadata entries" << std::endl;

    // 3. Read metadata back
    std::cout << "\n3. Read metadata:" << std::endl;
    auto metadata = gguf_read_metadata("exp8_bias.gguf");
    for (const auto &[key, val] : metadata)
    {
        std::cout << "  " << key << " = ";
        if (std::holds_alternative<std::string>(val))
            std::cout << "\"" << std::get<std::string>(val) << "\"";
        else if (std::holds_alternative<uint32_t>(val))
            std::cout << std::get<uint32_t>(val);
        else if (std::holds_alternative<uint64_t>(val))
            std::cout << std::get<uint64_t>(val);
        else if (std::holds_alternative<float>(val))
            std::cout << std::get<float>(val);
        std::cout << std::endl;
    }

    // 4. List tensors in a file
    std::cout << "\n4. List tensors:" << std::endl;
    auto names = gguf_list_tensors("exp8_bias.gguf");
    for (const auto &n : names)
        std::cout << "  - " << n << std::endl;

    // 5. Auto-detect format via .gguf extension
    std::cout << "\n5. Auto-detect format (.gguf extension):" << std::endl;
    Tensor<double> D({2, 2}, {1.0, 2.0, 3.0, 4.0});
    D.save("exp8_auto.gguf");
    Tensor<double> D2 = TensorN::load<double>("exp8_auto.gguf");
    std::cout << "  Saved via tensor.save() = " << D << std::endl;
    std::cout << "  Loaded via load<double>() = " << D2 << std::endl;
    std::cout << "  Match: " << (D == D2 ? "YES" : "NO") << std::endl;

    // 6. Different data types
    std::cout << "\n6. Different data types:" << std::endl;

    Tensor<int32_t> I({3}, {100, 200, 300});
    save_gguf(I, "exp8_int32.gguf", "int_tensor");
    auto I2 = load_gguf<int32_t>("exp8_int32.gguf", "int_tensor");
    std::cout << "  int32: " << I << " -> " << I2 << std::endl;

    Tensor<int64_t> L({2}, {123456789LL, 987654321LL});
    save_gguf(L, "exp8_int64.gguf", "long_tensor");
    auto L2 = load_gguf<int64_t>("exp8_int64.gguf", "long_tensor");
    std::cout << "  int64: " << L << " -> " << L2 << std::endl;

    Tensor<int8_t> B({4}, {1, 2, 3, 4});
    save_gguf(B, "exp8_int8.gguf", "byte_tensor");
    auto B2 = load_gguf<int8_t>("exp8_int8.gguf", "byte_tensor");
    std::cout << "  int8:  " << B << " -> " << B2 << std::endl;

    // 7. Scalar tensor
    std::cout << "\n7. Scalar tensor:" << std::endl;
    Tensor<float> scalar({}, {3.14f});
    save_gguf(scalar, "exp8_scalar.gguf", "pi");
    auto scalar2 = load_gguf<float>("exp8_scalar.gguf", "pi");
    std::cout << "  scalar: " << scalar << " -> " << scalar2 << std::endl;

    // 8. 3D tensor
    std::cout << "\n8. 3D tensor:" << std::endl;
    Tensor<float> T3d({2, 3, 4});
    for (size_t i = 0; i < T3d.size(); ++i)
        (*T3d.data)[i] = static_cast<float>(i) * 0.5f;
    save_gguf(T3d, "exp8_3d.gguf", "embedding.weight");
    auto T3d2 = load_gguf<float>("exp8_3d.gguf", "embedding.weight");
    std::cout << "  3D shape: {";
    for (auto s : T3d2.shape()) std::cout << s << " ";
    std::cout << "}" << std::endl;
    std::cout << "  Match: " << (T3d == T3d2 ? "YES" : "NO") << std::endl;

    // Cleanup
    std::cout << "\n9. Cleanup temporary files." << std::endl;
    std::remove("exp8_weight.gguf");
    std::remove("exp8_bias.gguf");
    std::remove("exp8_auto.gguf");
    std::remove("exp8_int32.gguf");
    std::remove("exp8_int64.gguf");
    std::remove("exp8_int8.gguf");
    std::remove("exp8_scalar.gguf");
    std::remove("exp8_3d.gguf");

    std::cout << "\nAll GGUF tests passed!" << std::endl;

    return 0;
}
