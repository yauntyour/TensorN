#pragma once
#ifndef __SAFETENSORS__H__
#define __SAFETENSORS__H__

#include "tensor.hpp"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <nlohmann/json.hpp>

// ============================================================================
// safetensors (https://github.com/huggingface/safetensors) 格式读写支持。
//
// 文件布局：
//   8 字节  : uint64 LE 头部长度 N
//   N 字节  : JSON UTF-8 头部，形如
//             {"NAME": {"dtype": "F32", "shape": [2, 3],
//                       "data_offsets": [BEGIN, END]}, "__metadata__": {...}}
//   剩余部分: 连续的张量原始字节（little-endian，C 行主序）
//
// data_offsets 相对于数据区起始位置（即 8 + N 处）。
// 支持 HuggingFace 分片命名 model.safetensors-00001-of-00002.safetensors，
// 单分片时同样输出 model.safetensors-00001-of-00001.safetensors。
// ============================================================================

namespace TensorN
{
    // ------------------------------------------------------------
    // dtype <-> 字符串 映射
    // ------------------------------------------------------------

    template <typename T>
    constexpr bool is_supported_safetensors_type()
    {
        return std::is_same_v<T, float> ||
               std::is_same_v<T, double> ||
               std::is_same_v<T, TensorN::half> ||
               std::is_same_v<T, TensorN::bfloat16> ||
               std::is_same_v<T, TensorN::tf32> ||
               std::is_same_v<T, TensorN::fp8_e4m3> ||
               std::is_same_v<T, TensorN::fp8_e5m2> ||
               std::is_same_v<T, int8_t> ||
               std::is_same_v<T, int16_t> ||
               std::is_same_v<T, int32_t> ||
               std::is_same_v<T, int64_t> ||
               std::is_same_v<T, uint8_t> ||
               std::is_same_v<T, uint16_t> ||
               std::is_same_v<T, uint32_t> ||
               std::is_same_v<T, uint64_t>;
    }

    template <typename T>
    constexpr const char *get_safetensors_dtype()
    {
        if constexpr (std::is_same_v<T, float>)
            return "F32";
        if constexpr (std::is_same_v<T, double>)
            return "F64";
        if constexpr (std::is_same_v<T, TensorN::half>)
            return "F16";
        if constexpr (std::is_same_v<T, TensorN::bfloat16>)
            return "BF16";
        if constexpr (std::is_same_v<T, TensorN::tf32>)
            return "F32"; // TF32 以 float 存储，位模式即合法 F32
        if constexpr (std::is_same_v<T, TensorN::fp8_e4m3>)
            return "F8_E4M3";
        if constexpr (std::is_same_v<T, TensorN::fp8_e5m2>)
            return "F8_E5M2";
        if constexpr (std::is_same_v<T, int8_t>)
            return "I8";
        if constexpr (std::is_same_v<T, int16_t>)
            return "I16";
        if constexpr (std::is_same_v<T, int32_t>)
            return "I32";
        if constexpr (std::is_same_v<T, int64_t>)
            return "I64";
        if constexpr (std::is_same_v<T, uint8_t>)
            return "U8";
        if constexpr (std::is_same_v<T, uint16_t>)
            return "U16";
        if constexpr (std::is_same_v<T, uint32_t>)
            return "U32";
        if constexpr (std::is_same_v<T, uint64_t>)
            return "U64";
        TENSOR_THROW("Unsupported type for safetensors format");
    }

    inline size_t safetensors_dtype_size(const std::string &dtype)
    {
        if (dtype == "F64" || dtype == "I64" || dtype == "U64")
            return 8;
        if (dtype == "F32" || dtype == "I32" || dtype == "U32")
            return 4;
        if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16")
            return 2;
        if (dtype == "F8_E4M3" || dtype == "F8_E5M2" || dtype == "I8" || dtype == "U8" || dtype == "BOOL")
            return 1;
        return 0;
    }

    // ------------------------------------------------------------
    // SafeTensor —— 与类型无关的序列化载体（支持混合 dtype）
    // ------------------------------------------------------------

    struct SafeTensor
    {
        std::string dtype;          // "F32" / "I64" / "BF16" / ...
        std::vector<int64_t> shape; // 各维大小（空 == 标量）
        std::vector<uint8_t> data;  // 原始 little-endian 字节
    };

    template <typename T>
    SafeTensor make_safetensor(const Tensor<T> &tensor)
    {
        if (!is_supported_safetensors_type<T>())
        {
            TENSOR_THROW("Type not supported for safetensors format");
        }

        SafeTensor st;
        st.dtype = get_safetensors_dtype<T>();
        const auto &shape = tensor.shape();
        st.shape.reserve(shape.size());
        for (auto d : shape)
        {
            st.shape.push_back(static_cast<int64_t>(d));
        }

        st.data.resize(tensor.data->size() * sizeof(T));
        if (!st.data.empty())
        {
            std::memcpy(st.data.data(), tensor.data->data(), st.data.size());
        }
        return st;
    }

    template <typename T>
    Tensor<T> from_safetensor(const SafeTensor &st)
    {
        if (st.dtype != get_safetensors_dtype<T>())
        {
            TENSOR_THROW("Dtype mismatch for safetensors tensor: expected " +
                         std::string(get_safetensors_dtype<T>()) + ", got " + st.dtype);
        }

        std::vector<size_t> shape(st.shape.size());
        size_t numel = 1;
        for (size_t i = 0; i < st.shape.size(); ++i)
        {
            shape[i] = static_cast<size_t>(st.shape[i]);
            numel *= shape[i];
        }

        if (st.data.size() != numel * sizeof(T))
        {
            TENSOR_THROW("Data size mismatch for safetensors tensor");
        }

        std::vector<T> data_vec(numel);
        if (numel > 0)
        {
            std::memcpy(data_vec.data(), st.data.data(), st.data.size());
        }
        return Tensor<T>(shape, data_vec);
    }

    // ------------------------------------------------------------
    // 内部写入辅助
    // ------------------------------------------------------------

    inline std::string build_safetensors_header(
        const std::vector<std::pair<std::string, SafeTensor>> &tensors,
        const std::unordered_map<std::string, std::string> &metadata)
    {
        nlohmann::json j = nlohmann::json::object();
        uint64_t offset = 0;
        for (const auto &[name, st] : tensors)
        {
            nlohmann::json entry;
            entry["dtype"] = st.dtype;
            entry["shape"] = st.shape;
            entry["data_offsets"] = {offset, offset + st.data.size()};
            j[name] = entry;
            offset += st.data.size();
        }
        if (!metadata.empty())
        {
            j["__metadata__"] = metadata;
        }
        return j.dump();
    }

    inline void write_safetensors_file(
        const std::vector<std::pair<std::string, SafeTensor>> &tensors,
        const std::string &filename,
        const std::unordered_map<std::string, std::string> &metadata)
    {
        if (tensors.empty())
        {
            TENSOR_THROW("No tensors to save");
        }

        std::string header = build_safetensors_header(tensors, metadata);

        std::ofstream file(filename, std::ios::binary);
        if (!file)
        {
            TENSOR_THROW("Cannot open file for writing: " + filename);
        }

        uint64_t header_size = static_cast<uint64_t>(header.size());
        file.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
        file.write(header.data(), static_cast<std::streamsize>(header.size()));

        for (const auto &[name, st] : tensors)
        {
            if (!st.data.empty())
            {
                file.write(reinterpret_cast<const char *>(st.data.data()),
                           static_cast<std::streamsize>(st.data.size()));
            }
        }

        if (!file)
        {
            TENSOR_THROW("Error writing safetensors file: " + filename);
        }
    }

    // 分片文件名：model.safetensors -> model.safetensors-00001-of-00002.safetensors；
    // 无扩展名基名 model -> model-00001-of-00002.safetensors
    inline std::string safetensors_shard_filename(const std::string &base_filename,
                                                  size_t shard_index,
                                                  size_t num_shards)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "-%05zu-of-%05zu.safetensors",
                      shard_index + 1, num_shards);
        return base_filename + buf;
    }

    // ------------------------------------------------------------
    // 保存
    // ------------------------------------------------------------

    // 单张量保存
    template <typename T>
    void save_safetensors(const Tensor<T> &tensor,
                          const std::string &filename,
                          const std::string &tensor_name = "tensor",
                          const std::unordered_map<std::string, std::string> &metadata = {})
    {
        if (!is_supported_safetensors_type<T>())
        {
            TENSOR_THROW("Type not supported for safetensors format");
        }

        std::vector<std::pair<std::string, SafeTensor>> tensors;
        tensors.emplace_back(tensor_name, make_safetensor(tensor));
        write_safetensors_file(tensors, filename, metadata);
    }

    // 多张量保存（同一种 C++ 类型）
    template <typename T>
    void save_safetensors_multi(
        const std::vector<std::pair<std::string, Tensor<T>>> &tensors,
        const std::string &filename,
        const std::unordered_map<std::string, std::string> &metadata = {})
    {
        std::vector<std::pair<std::string, SafeTensor>> sts;
        sts.reserve(tensors.size());
        for (const auto &[name, tensor] : tensors)
        {
            sts.emplace_back(name, make_safetensor(tensor));
        }
        write_safetensors_file(sts, filename, metadata);
    }

    // 多张量保存（混合 dtype，通过 SafeTensor 载体）
    inline void save_safetensors_multi(
        const std::vector<std::pair<std::string, SafeTensor>> &tensors,
        const std::string &filename,
        const std::unordered_map<std::string, std::string> &metadata = {})
    {
        write_safetensors_file(tensors, filename, metadata);
    }

    // 分片保存：按 max_shard_size 贪心装箱，输出
    // model.safetensors-00001-of-00002.safetensors 等文件（单分片时为
    // model.safetensors-00001-of-00001.safetensors）。
    inline void save_safetensors_sharded(
        const std::vector<std::pair<std::string, SafeTensor>> &tensors,
        const std::string &base_filename,
        uint64_t max_shard_size = 5ULL * 1024 * 1024 * 1024,
        const std::unordered_map<std::string, std::string> &metadata = {})
    {
        if (tensors.empty())
        {
            TENSOR_THROW("No tensors to save");
        }
        if (max_shard_size == 0)
        {
            TENSOR_THROW("max_shard_size must be > 0");
        }

        std::vector<std::vector<std::pair<std::string, SafeTensor>>> shards;
        for (const auto &kv : tensors)
        {
            uint64_t tsize = static_cast<uint64_t>(kv.second.data.size());
            if (tsize > max_shard_size)
            {
                TENSOR_THROW("Tensor '" + kv.first + "' is larger than max_shard_size");
            }

            bool packed = false;
            if (!shards.empty())
            {
                uint64_t cur = 0;
                for (const auto &s : shards.back())
                {
                    cur += static_cast<uint64_t>(s.second.data.size());
                }
                if (cur + tsize <= max_shard_size)
                {
                    shards.back().push_back(kv);
                    packed = true;
                }
            }
            if (!packed)
            {
                shards.emplace_back();
                shards.back().push_back(kv);
            }
        }

        for (size_t i = 0; i < shards.size(); ++i)
        {
            write_safetensors_file(shards[i],
                                   safetensors_shard_filename(base_filename, i, shards.size()),
                                   metadata);
        }
    }

    // 分片保存（同一种 C++ 类型）
    template <typename T>
    void save_safetensors_sharded(
        const std::vector<std::pair<std::string, Tensor<T>>> &tensors,
        const std::string &base_filename,
        uint64_t max_shard_size = 5ULL * 1024 * 1024 * 1024,
        const std::unordered_map<std::string, std::string> &metadata = {})
    {
        std::vector<std::pair<std::string, SafeTensor>> sts;
        sts.reserve(tensors.size());
        for (const auto &[name, tensor] : tensors)
        {
            sts.emplace_back(name, make_safetensor(tensor));
        }
        save_safetensors_sharded(sts, base_filename, max_shard_size, metadata);
    }

    // ------------------------------------------------------------
    // 加载
    // ------------------------------------------------------------

    // 读取文件内全部张量（保留原始 dtype，供混合类型使用）
    inline std::unordered_map<std::string, SafeTensor> load_safetensors_raw(
        const std::string &filename, nlohmann::json *metadata_out = nullptr)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            TENSOR_THROW("Cannot open file: " + filename);
        }

        uint64_t header_size = 0;
        file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
        if (!file)
        {
            TENSOR_THROW("Cannot read safetensors header size");
        }
        if (header_size > 100ULL * 1024 * 1024)
        {
            TENSOR_THROW("safetensors header too large (>100MB)");
        }

        std::string header(static_cast<size_t>(header_size), '\0');
        file.read(header.data(), static_cast<std::streamsize>(header_size));
        if (!file)
        {
            TENSOR_THROW("Cannot read safetensors header");
        }

        nlohmann::json j;
        try
        {
            j = nlohmann::json::parse(header);
        }
        catch (...)
        {
            TENSOR_THROW("Invalid safetensors JSON header");
        }
        if (!j.is_object())
        {
            TENSOR_THROW("Invalid safetensors header (not a JSON object)");
        }

        uint64_t data_base = 8 + header_size;
        std::unordered_map<std::string, SafeTensor> result;
        for (auto it = j.begin(); it != j.end(); ++it)
        {
            if (it.key() == "__metadata__")
            {
                if (metadata_out != nullptr)
                {
                    *metadata_out = it.value();
                }
                continue;
            }
            if (!it.value().is_object())
            {
                TENSOR_THROW("Invalid safetensors tensor entry: " + it.key());
            }

            SafeTensor st;
            st.dtype = it.value().value("dtype", "");
            if (safetensors_dtype_size(st.dtype) == 0)
            {
                TENSOR_THROW("Unsupported dtype '" + st.dtype + "' for tensor " + it.key());
            }

            st.shape = it.value()["shape"].get<std::vector<int64_t>>();
            for (auto d : st.shape)
            {
                if (d < 0)
                {
                    TENSOR_THROW("Negative dimension in tensor " + it.key());
                }
            }

            const auto &offsets = it.value()["data_offsets"];
            uint64_t begin = offsets[0].get<uint64_t>();
            uint64_t end = offsets[1].get<uint64_t>();
            if (end < begin)
            {
                TENSOR_THROW("Invalid data_offsets for tensor " + it.key());
            }

            st.data.resize(static_cast<size_t>(end - begin));
            if (!st.data.empty())
            {
                file.seekg(static_cast<std::streamoff>(data_base + begin));
                file.read(reinterpret_cast<char *>(st.data.data()),
                          static_cast<std::streamsize>(st.data.size()));
                if (!file)
                {
                    TENSOR_THROW("Error reading tensor data: " + it.key());
                }
            }
            result[it.key()] = std::move(st);
        }
        return result;
    }

    // 读取全部张量并转换为 T（文件内 dtype 必须与 T 一致）
    template <typename T>
    std::unordered_map<std::string, Tensor<T>> load_safetensors_multi(
        const std::string &filename)
    {
        auto raw = load_safetensors_raw(filename);
        if (raw.empty())
        {
            TENSOR_THROW("No tensors found in safetensors file");
        }

        std::unordered_map<std::string, Tensor<T>> result;
        for (auto &[name, st] : raw)
        {
            result[name] = from_safetensor<T>(st);
        }
        return result;
    }

    // 读取单个张量；tensor_name 为空时要求文件内只有一个张量
    template <typename T>
    Tensor<T> load_safetensors(const std::string &filename,
                               const std::string &tensor_name = "")
    {
        auto raw = load_safetensors_raw(filename);
        if (raw.empty())
        {
            TENSOR_THROW("No tensors found in safetensors file");
        }

        std::string target = tensor_name;
        if (target.empty())
        {
            if (raw.size() > 1)
            {
                TENSOR_THROW("safetensors file contains multiple tensors; specify tensor_name");
            }
            target = raw.begin()->first;
        }

        auto it = raw.find(target);
        if (it == raw.end())
        {
            TENSOR_THROW("Tensor not found: " + target);
        }
        return from_safetensor<T>(it->second);
    }

    // 加载分片模型：合并 model.safetensors-00001-of-*.safetensors 全部分片
    template <typename T>
    std::unordered_map<std::string, Tensor<T>> load_safetensors_sharded(
        const std::string &base_filename)
    {
        const std::string ext = ".safetensors";
        std::filesystem::path base_path(base_filename);
        std::string base_name = base_path.filename().string();
        std::string stem_name = base_name;
        if (stem_name.size() >= ext.size() &&
            stem_name.compare(stem_name.size() - ext.size(), ext.size(), ext) == 0)
        {
            stem_name = stem_name.substr(0, stem_name.size() - ext.size());
        }
        // 兼容两种命名：model.safetensors-00001-of-00002.safetensors 与
        // model-00001-of-00002.safetensors
        const std::string prefix_ext = base_name + "-";
        const std::string prefix_stem = stem_name + "-";

        std::filesystem::path dir = base_path.parent_path();
        if (dir.empty())
        {
            dir = ".";
        }

        auto is_shard_file = [&](const std::string &fname) -> bool
        {
            size_t plen = 0;
            if (fname.compare(0, prefix_ext.size(), prefix_ext) == 0)
            {
                plen = prefix_ext.size();
            }
            else if (fname.compare(0, prefix_stem.size(), prefix_stem) == 0)
            {
                plen = prefix_stem.size();
            }
            else
            {
                return false;
            }

            if (fname.size() <= plen + ext.size())
            {
                return false;
            }
            if (fname.compare(fname.size() - ext.size(), ext.size(), ext) != 0)
            {
                return false;
            }
            std::string middle = fname.substr(plen, fname.size() - plen - ext.size());
            return middle.find("-of-") != std::string::npos;
        };

        std::vector<std::string> shards;
        for (const auto &entry : std::filesystem::directory_iterator(dir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            std::string fname = entry.path().filename().string();
            if (is_shard_file(fname))
            {
                shards.push_back(entry.path().string());
            }
        }
        std::sort(shards.begin(), shards.end());

        if (shards.empty())
        {
            TENSOR_THROW("No safetensors shards found for: " + base_filename);
        }

        std::unordered_map<std::string, Tensor<T>> merged;
        for (const auto &shard : shards)
        {
            auto tensors = load_safetensors_multi<T>(shard);
            for (auto &[name, tensor] : tensors)
            {
                merged[name] = std::move(tensor);
            }
        }
        return merged;
    }

} // namespace TensorN

#endif // !__SAFETENSORS__H__
