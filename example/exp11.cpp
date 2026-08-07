#include "TensorN.hpp"
#include <iostream>

using namespace TensorN;

int main()
{
    std::cout << "=== exp11: safetensors I/O (incl. sharded model.safetensors-00001-of-00001.safetensors) ===\n" << std::endl;

    // 1. Single tensor via save()/load() auto-detection (also works with shard-style names)
    Tensor<float> t({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    t.save("example/model.safetensors-00001-of-00001.safetensors");
    auto t2 = load<float>("example/model.safetensors-00001-of-00001.safetensors");
    std::cout << "1. shard-named save/load: " << (t == t2 ? "PASS" : "FAIL") << std::endl;

    // 2. Mixed-dtype state dict (like PyTorch save_file)
    Tensor<float> w({4, 4});
    Tensor<int64_t> ids({10});
    Tensor<TensorN::half> h({2, 2});
    std::vector<std::pair<std::string, SafeTensor>> state;
    state.emplace_back("weight", make_safetensor(w));
    state.emplace_back("ids", make_safetensor(ids));
    state.emplace_back("half_w", make_safetensor(h));
    save_safetensors_multi(state, "example/model.safetensors", {{"format", "pt"}});

    auto raw = load_safetensors_raw("example/model.safetensors");
    auto w2 = from_safetensor<float>(raw["weight"]);
    auto ids2 = from_safetensor<int64_t>(raw["ids"]);
    auto h2 = from_safetensor<TensorN::half>(raw["half_w"]);
    bool ok = w2.shape() == w.shape() && ids2.shape() == ids.shape() && h2.shape() == h.shape();
    std::cout << "2. mixed-dtype multi save/load: " << (ok ? "PASS" : "FAIL") << std::endl;

    // 3. Sharded save (small max_shard_size to force 2+ shards) + merged load
    std::vector<std::pair<std::string, Tensor<float>>> model;
    for (int i = 0; i < 6; ++i)
    {
        model.emplace_back("layer." + std::to_string(i) + ".weight", Tensor<float>({64, 64}));
    }
    save_safetensors_sharded(model, "example/sharded.safetensors", 64 * 64 * 4 * 3);

    auto merged = load_safetensors_sharded<float>("example/sharded.safetensors");
    bool ok2 = (merged.size() == 6);
    std::cout << "3. sharded save/load (" << (merged.size()) << " tensors from all shards): "
              << (ok2 ? "PASS" : "FAIL") << std::endl;

    std::cout << "\nFiles written:\n";
    for (const auto &name : {"example/model.safetensors-00001-of-00001.safetensors",
                             "example/model.safetensors",
                             "example/sharded.safetensors-00001-of-00002.safetensors",
                             "example/sharded.safetensors-00002-of-00002.safetensors"})
    {
        std::ifstream f(name);
        std::cout << "  " << name << (f ? "  OK" : "  (n/a)") << std::endl;
    }
    return 0;
}
