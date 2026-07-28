#pragma once
#ifndef __GGUF__H__
#define __GGUF__H__

#include "../tensor.hpp"
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <cstdint>
#include <algorithm>

namespace TensorN
{

    constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian
    constexpr uint32_t GGUF_VERSION = 3;
    constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;

    enum class GGMLType : uint32_t
    {
        F32 = 0,
        F16 = 1,
        Q4_0 = 2,
        Q4_1 = 3,
        Q5_0 = 6,
        Q5_1 = 7,
        Q8_0 = 8,
        Q8_1 = 9,
        Q2_K = 10,
        Q3_K = 11,
        Q4_K = 12,
        Q5_K = 13,
        Q6_K = 14,
        Q8_K = 15,
        IQ2_XXS = 16,
        IQ2_XS = 17,
        IQ3_XXS = 18,
        IQ1_S = 19,
        IQ4_NL = 20,
        IQ3_S = 21,
        IQ2_S = 22,
        IQ4_XS = 23,
        I8 = 24,
        I16 = 25,
        I32 = 26,
        I64 = 27,
        F64 = 28,
        IQ1_M = 29,
        BF16 = 30,
        TQ1_0 = 34,
        TQ2_0 = 35,
        MXFP4 = 39,
    };

    enum class GGUFMetadataValueType : uint32_t
    {
        UINT8 = 0,
        INT8 = 1,
        UINT16 = 2,
        INT16 = 3,
        UINT32 = 4,
        INT32 = 5,
        FLOAT32 = 6,
        BOOL = 7,
        STRING = 8,
        ARRAY = 9,
        UINT64 = 10,
        INT64 = 11,
        FLOAT64 = 12,
    };

    using GGUFMetadataValue = std::variant<
        uint8_t,
        int8_t,
        uint16_t,
        int16_t,
        uint32_t,
        int32_t,
        float,
        bool,
        std::string,
        uint64_t,
        int64_t,
        double>;

    struct GGUFMetadataArray
    {
        GGUFMetadataValueType type;
        std::vector<GGUFMetadataArray> values;

        GGUFMetadataArray() : type(GGUFMetadataValueType::UINT8) {}
    };

    template <typename T>
    constexpr GGMLType get_gguf_type()
    {
        if constexpr (std::is_same_v<T, float>)
            return GGMLType::F32;
        if constexpr (std::is_same_v<T, double>)
            return GGMLType::F64;
        if constexpr (std::is_same_v<T, int8_t>)
            return GGMLType::I8;
        if constexpr (std::is_same_v<T, int16_t>)
            return GGMLType::I16;
        if constexpr (std::is_same_v<T, int32_t>)
            return GGMLType::I32;
        if constexpr (std::is_same_v<T, int64_t>)
            return GGMLType::I64;
        if constexpr (std::is_same_v<T, uint8_t>)
            return GGMLType::I8;
        TENSOR_THROW("Unsupported type for GGUF format");
    }

    template <typename T>
    constexpr bool is_supported_gguf_type()
    {
        return std::is_same_v<T, float> ||
               std::is_same_v<T, double> ||
               std::is_same_v<T, int8_t> ||
               std::is_same_v<T, int16_t> ||
               std::is_same_v<T, int32_t> ||
               std::is_same_v<T, int64_t> ||
               std::is_same_v<T, uint8_t>;
    }

    inline size_t ggml_type_size(GGMLType type)
    {
        switch (type)
        {
        case GGMLType::F32:
            return 4;
        case GGMLType::F16:
            return 2;
        case GGMLType::Q4_0:
            return 18;
        case GGMLType::Q4_1:
            return 20;
        case GGMLType::Q5_0:
            return 22;
        case GGMLType::Q5_1:
            return 24;
        case GGMLType::Q8_0:
            return 34;
        case GGMLType::Q8_1:
            return 36;
        case GGMLType::Q2_K:
            return 84;
        case GGMLType::Q3_K:
            return 110;
        case GGMLType::Q4_K:
            return 144;
        case GGMLType::Q5_K:
            return 176;
        case GGMLType::Q6_K:
            return 210;
        case GGMLType::Q8_K:
            return 292;
        case GGMLType::I8:
            return 1;
        case GGMLType::I16:
            return 2;
        case GGMLType::I32:
            return 4;
        case GGMLType::I64:
            return 8;
        case GGMLType::F64:
            return 8;
        case GGMLType::BF16:
            return 2;
        default:
            TENSOR_THROW("Unsupported GGML type for size query");
        }
        return 0;
    }

    inline size_t ggml_type_element_size(GGMLType type)
    {
        switch (type)
        {
        case GGMLType::F32:
            return 4;
        case GGMLType::F16:
            return 2;
        case GGMLType::I8:
            return 1;
        case GGMLType::I16:
            return 2;
        case GGMLType::I32:
            return 4;
        case GGMLType::I64:
            return 8;
        case GGMLType::F64:
            return 8;
        case GGMLType::BF16:
            return 2;
        default:
            TENSOR_THROW("Element size not defined for quantized GGML type");
        }
        return 0;
    }

    inline uint64_t align_offset(uint64_t offset, uint32_t alignment)
    {
        return offset + (alignment - (offset % alignment)) % alignment;
    }

    inline void write_gguf_string(std::ofstream &file, const std::string &str)
    {
        uint64_t len = static_cast<uint64_t>(str.size());
        file.write(reinterpret_cast<const char *>(&len), sizeof(len));
        file.write(str.data(), static_cast<std::streamsize>(len));
    }

    inline std::string read_gguf_string(std::ifstream &file)
    {
        uint64_t len;
        file.read(reinterpret_cast<char *>(&len), sizeof(len));
        std::string str(static_cast<size_t>(len), '\0');
        file.read(&str[0], static_cast<std::streamsize>(len));
        return str;
    }

    inline void write_metadata_value(std::ofstream &file, const GGUFMetadataValue &value)
    {
        if (std::holds_alternative<uint8_t>(value))
        {
            uint8_t v = std::get<uint8_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<int8_t>(value))
        {
            int8_t v = std::get<int8_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<uint16_t>(value))
        {
            uint16_t v = std::get<uint16_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<int16_t>(value))
        {
            int16_t v = std::get<int16_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<uint32_t>(value))
        {
            uint32_t v = std::get<uint32_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<int32_t>(value))
        {
            int32_t v = std::get<int32_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<float>(value))
        {
            float v = std::get<float>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<bool>(value))
        {
            uint8_t v = std::get<bool>(value) ? 1 : 0;
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<std::string>(value))
        {
            write_gguf_string(file, std::get<std::string>(value));
        }
        else if (std::holds_alternative<uint64_t>(value))
        {
            uint64_t v = std::get<uint64_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<int64_t>(value))
        {
            int64_t v = std::get<int64_t>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
        else if (std::holds_alternative<double>(value))
        {
            double v = std::get<double>(value);
            file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        }
    }

    inline GGUFMetadataValueType get_metadata_value_type(const GGUFMetadataValue &value)
    {
        if (std::holds_alternative<uint8_t>(value))
            return GGUFMetadataValueType::UINT8;
        if (std::holds_alternative<int8_t>(value))
            return GGUFMetadataValueType::INT8;
        if (std::holds_alternative<uint16_t>(value))
            return GGUFMetadataValueType::UINT16;
        if (std::holds_alternative<int16_t>(value))
            return GGUFMetadataValueType::INT16;
        if (std::holds_alternative<uint32_t>(value))
            return GGUFMetadataValueType::UINT32;
        if (std::holds_alternative<int32_t>(value))
            return GGUFMetadataValueType::INT32;
        if (std::holds_alternative<float>(value))
            return GGUFMetadataValueType::FLOAT32;
        if (std::holds_alternative<bool>(value))
            return GGUFMetadataValueType::BOOL;
        if (std::holds_alternative<std::string>(value))
            return GGUFMetadataValueType::STRING;
        if (std::holds_alternative<uint64_t>(value))
            return GGUFMetadataValueType::UINT64;
        if (std::holds_alternative<int64_t>(value))
            return GGUFMetadataValueType::INT64;
        if (std::holds_alternative<double>(value))
            return GGUFMetadataValueType::FLOAT64;
        TENSOR_THROW("Unknown metadata value type");
    }

    struct GGUFLoadResult
    {
        std::vector<size_t> shape;
        std::vector<uint8_t> raw_data;
        GGMLType type;
    };

    template <typename T>
    void save_gguf(const Tensor<T> &tensor,
                   const std::string &filename,
                   const std::string &tensor_name = "tensor",
                   const std::unordered_map<std::string, GGUFMetadataValue> &metadata = {})
    {
        if (!is_supported_gguf_type<T>())
        {
            TENSOR_THROW("Type not supported for .gguf format");
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file for writing: " + filename);

        const auto &shape = tensor.shape();
        GGMLType gguf_type = get_gguf_type<T>();
        uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;

        auto it_align = metadata.find("general.alignment");
        if (it_align != metadata.end())
        {
            if (std::holds_alternative<uint32_t>(it_align->second))
                alignment = std::get<uint32_t>(it_align->second);
        }

        uint32_t magic = GGUF_MAGIC;
        uint32_t version = GGUF_VERSION;
        uint64_t tensor_count = 1;
        uint64_t metadata_kv_count = static_cast<uint64_t>(metadata.size());

        file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char *>(&version), sizeof(version));
        file.write(reinterpret_cast<const char *>(&tensor_count), sizeof(tensor_count));
        file.write(reinterpret_cast<const char *>(&metadata_kv_count), sizeof(metadata_kv_count));

        for (const auto &[key, value] : metadata)
        {
            write_gguf_string(file, key);
            GGUFMetadataValueType vtype = get_metadata_value_type(value);
            uint32_t vtype_u32 = static_cast<uint32_t>(vtype);
            file.write(reinterpret_cast<const char *>(&vtype_u32), sizeof(vtype_u32));
            write_metadata_value(file, value);
        }

        write_gguf_string(file, tensor_name);
        uint32_t n_dims = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char *>(&n_dims), sizeof(n_dims));
        for (auto dim : shape)
        {
            uint64_t dim64 = static_cast<uint64_t>(dim);
            file.write(reinterpret_cast<const char *>(&dim64), sizeof(dim64));
        }
        uint32_t type_u32 = static_cast<uint32_t>(gguf_type);
        file.write(reinterpret_cast<const char *>(&type_u32), sizeof(type_u32));

        uint64_t tensor_data_offset = 0;
        file.write(reinterpret_cast<const char *>(&tensor_data_offset), sizeof(tensor_data_offset));

        uint64_t current_pos = static_cast<uint64_t>(file.tellp());
        uint64_t aligned_pos = align_offset(current_pos, alignment);
        uint64_t padding = aligned_pos - current_pos;
        for (uint64_t i = 0; i < padding; ++i)
        {
            char zero = 0;
            file.write(&zero, 1);
        }

        const char *raw_ptr = reinterpret_cast<const char *>(tensor.data->data());
        size_t raw_size = tensor.data->size() * sizeof(T);
        file.write(raw_ptr, static_cast<std::streamsize>(raw_size));

        if (!file)
            TENSOR_THROW("Error writing GGUF file: " + filename);
    }

    template <typename T>
    Tensor<T> load_gguf(const std::string &filename,
                        const std::string &tensor_name = "")
    {
        if (!is_supported_gguf_type<T>())
        {
            TENSOR_THROW("Type not supported for .gguf format");
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file: " + filename);

        uint32_t magic;
        file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (magic != GGUF_MAGIC)
            TENSOR_THROW("Not a valid GGUF file (bad magic)");

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(version));
        if (version != GGUF_VERSION)
            TENSOR_THROW("Unsupported GGUF version: " + std::to_string(version));

        uint64_t tensor_count;
        file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));

        uint64_t metadata_kv_count;
        file.read(reinterpret_cast<char *>(&metadata_kv_count), sizeof(metadata_kv_count));

        for (uint64_t i = 0; i < metadata_kv_count; ++i)
        {
            read_gguf_string(file);
            uint32_t vtype_u32;
            file.read(reinterpret_cast<char *>(&vtype_u32), sizeof(vtype_u32));
            GGUFMetadataValueType vtype = static_cast<GGUFMetadataValueType>(vtype_u32);

            switch (vtype)
            {
            case GGUFMetadataValueType::UINT8:
            case GGUFMetadataValueType::INT8:
            {
                int8_t v;
                file.read(reinterpret_cast<char *>(&v), 1);
                break;
            }
            case GGUFMetadataValueType::UINT16:
            case GGUFMetadataValueType::INT16:
            {
                int16_t v;
                file.read(reinterpret_cast<char *>(&v), 2);
                break;
            }
            case GGUFMetadataValueType::UINT32:
            case GGUFMetadataValueType::INT32:
            case GGUFMetadataValueType::FLOAT32:
            case GGUFMetadataValueType::BOOL:
            {
                int32_t v;
                file.read(reinterpret_cast<char *>(&v), 4);
                break;
            }
            case GGUFMetadataValueType::UINT64:
            case GGUFMetadataValueType::INT64:
            case GGUFMetadataValueType::FLOAT64:
            {
                int64_t v;
                file.read(reinterpret_cast<char *>(&v), 8);
                break;
            }
            case GGUFMetadataValueType::STRING:
            {
                read_gguf_string(file);
                break;
            }
            case GGUFMetadataValueType::ARRAY:
            {
                uint32_t arr_type;
                file.read(reinterpret_cast<char *>(&arr_type), sizeof(arr_type));
                uint64_t arr_len;
                file.read(reinterpret_cast<char *>(&arr_len), sizeof(arr_len));
                for (uint64_t j = 0; j < arr_len; ++j)
                {
                    GGUFMetadataValueType elem_type = static_cast<GGUFMetadataValueType>(arr_type);
                    switch (elem_type)
                    {
                    case GGUFMetadataValueType::UINT8:
                    case GGUFMetadataValueType::INT8:
                    {
                        int8_t v;
                        file.read(reinterpret_cast<char *>(&v), 1);
                        break;
                    }
                    case GGUFMetadataValueType::UINT16:
                    case GGUFMetadataValueType::INT16:
                    {
                        int16_t v;
                        file.read(reinterpret_cast<char *>(&v), 2);
                        break;
                    }
                    case GGUFMetadataValueType::UINT32:
                    case GGUFMetadataValueType::INT32:
                    case GGUFMetadataValueType::FLOAT32:
                    case GGUFMetadataValueType::BOOL:
                    {
                        int32_t v;
                        file.read(reinterpret_cast<char *>(&v), 4);
                        break;
                    }
                    case GGUFMetadataValueType::UINT64:
                    case GGUFMetadataValueType::INT64:
                    case GGUFMetadataValueType::FLOAT64:
                    {
                        int64_t v;
                        file.read(reinterpret_cast<char *>(&v), 8);
                        break;
                    }
                    case GGUFMetadataValueType::STRING:
                    {
                        read_gguf_string(file);
                        break;
                    }
                    case GGUFMetadataValueType::ARRAY:
                        TENSOR_THROW("Nested arrays not supported in GGUF metadata reader");
                    }
                }
                break;
            }
            }
        }

        struct TensorInfo
        {
            std::string name;
            uint32_t n_dims;
            std::vector<uint64_t> dims;
            GGMLType type;
            uint64_t offset;
        };

        std::vector<TensorInfo> tensor_infos(tensor_count);
        for (uint64_t i = 0; i < tensor_count; ++i)
        {
            tensor_infos[i].name = read_gguf_string(file);
            file.read(reinterpret_cast<char *>(&tensor_infos[i].n_dims), sizeof(uint32_t));
            tensor_infos[i].dims.resize(tensor_infos[i].n_dims);
            for (uint32_t d = 0; d < tensor_infos[i].n_dims; ++d)
            {
                file.read(reinterpret_cast<char *>(&tensor_infos[i].dims[d]), sizeof(uint64_t));
            }
            uint32_t type_u32;
            file.read(reinterpret_cast<char *>(&type_u32), sizeof(uint32_t));
            tensor_infos[i].type = static_cast<GGMLType>(type_u32);
            file.read(reinterpret_cast<char *>(&tensor_infos[i].offset), sizeof(uint64_t));
        }

        if (tensor_count == 0)
            TENSOR_THROW("No tensors found in GGUF file");

        uint64_t tensor_data_base = static_cast<uint64_t>(file.tellg());
        tensor_data_base = align_offset(tensor_data_base, GGUF_DEFAULT_ALIGNMENT);

        size_t target_idx = 0;
        if (!tensor_name.empty())
        {
            bool found = false;
            for (size_t i = 0; i < tensor_infos.size(); ++i)
            {
                if (tensor_infos[i].name == tensor_name)
                {
                    target_idx = i;
                    found = true;
                    break;
                }
            }
            if (!found)
                TENSOR_THROW("Tensor '" + tensor_name + "' not found in GGUF file");
        }

        const auto &info = tensor_infos[target_idx];
        GGMLType expected_type = get_gguf_type<T>();

        std::vector<size_t> shape(info.n_dims);
        size_t total_elements = 1;
        for (uint32_t d = 0; d < info.n_dims; ++d)
        {
            shape[d] = static_cast<size_t>(info.dims[d]);
            total_elements *= shape[d];
        }

        if (info.type == expected_type ||
            (info.type == GGMLType::F32 && expected_type == GGMLType::F32) ||
            (info.type == GGMLType::F64 && expected_type == GGMLType::F64) ||
            (info.type == GGMLType::I8 && expected_type == GGMLType::I8) ||
            (info.type == GGMLType::I16 && expected_type == GGMLType::I16) ||
            (info.type == GGMLType::I32 && expected_type == GGMLType::I32) ||
            (info.type == GGMLType::I64 && expected_type == GGMLType::I64))
        {
            uint64_t abs_offset = tensor_data_base + info.offset;
            file.seekg(static_cast<std::streamoff>(abs_offset));

            std::vector<T> data_vec(total_elements);
            file.read(reinterpret_cast<char *>(data_vec.data()),
                      static_cast<std::streamsize>(total_elements * sizeof(T)));

            if (!file)
                TENSOR_THROW("Error reading tensor data from GGUF file");

            return Tensor<T>(shape, data_vec);
        }
        else
        {
            TENSOR_THROW(
                "Type mismatch in GGUF file. Expected GGML type " +
                std::to_string(static_cast<uint32_t>(expected_type)) +
                ", got " + std::to_string(static_cast<uint32_t>(info.type)));
        }
    }

    inline std::vector<std::string> gguf_list_tensors(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file: " + filename);

        uint32_t magic;
        file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (magic != GGUF_MAGIC)
            TENSOR_THROW("Not a valid GGUF file (bad magic)");

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(version));

        uint64_t tensor_count;
        file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));

        uint64_t metadata_kv_count;
        file.read(reinterpret_cast<char *>(&metadata_kv_count), sizeof(metadata_kv_count));

        for (uint64_t i = 0; i < metadata_kv_count; ++i)
        {
            read_gguf_string(file);
            uint32_t vtype_u32;
            file.read(reinterpret_cast<char *>(&vtype_u32), sizeof(vtype_u32));
            GGUFMetadataValueType vtype = static_cast<GGUFMetadataValueType>(vtype_u32);

            switch (vtype)
            {
            case GGUFMetadataValueType::UINT8:
            case GGUFMetadataValueType::INT8:
            {
                int8_t v;
                file.read(reinterpret_cast<char *>(&v), 1);
                break;
            }
            case GGUFMetadataValueType::UINT16:
            case GGUFMetadataValueType::INT16:
            {
                int16_t v;
                file.read(reinterpret_cast<char *>(&v), 2);
                break;
            }
            case GGUFMetadataValueType::UINT32:
            case GGUFMetadataValueType::INT32:
            case GGUFMetadataValueType::FLOAT32:
            case GGUFMetadataValueType::BOOL:
            {
                int32_t v;
                file.read(reinterpret_cast<char *>(&v), 4);
                break;
            }
            case GGUFMetadataValueType::UINT64:
            case GGUFMetadataValueType::INT64:
            case GGUFMetadataValueType::FLOAT64:
            {
                int64_t v;
                file.read(reinterpret_cast<char *>(&v), 8);
                break;
            }
            case GGUFMetadataValueType::STRING:
            {
                read_gguf_string(file);
                break;
            }
            case GGUFMetadataValueType::ARRAY:
            {
                uint32_t arr_type;
                file.read(reinterpret_cast<char *>(&arr_type), sizeof(arr_type));
                uint64_t arr_len;
                file.read(reinterpret_cast<char *>(&arr_len), sizeof(arr_len));
                for (uint64_t j = 0; j < arr_len; ++j)
                {
                    GGUFMetadataValueType elem_type = static_cast<GGUFMetadataValueType>(arr_type);
                    switch (elem_type)
                    {
                    case GGUFMetadataValueType::UINT8:
                    case GGUFMetadataValueType::INT8:
                    {
                        int8_t v;
                        file.read(reinterpret_cast<char *>(&v), 1);
                        break;
                    }
                    case GGUFMetadataValueType::UINT16:
                    case GGUFMetadataValueType::INT16:
                    {
                        int16_t v;
                        file.read(reinterpret_cast<char *>(&v), 2);
                        break;
                    }
                    case GGUFMetadataValueType::UINT32:
                    case GGUFMetadataValueType::INT32:
                    case GGUFMetadataValueType::FLOAT32:
                    case GGUFMetadataValueType::BOOL:
                    {
                        int32_t v;
                        file.read(reinterpret_cast<char *>(&v), 4);
                        break;
                    }
                    case GGUFMetadataValueType::UINT64:
                    case GGUFMetadataValueType::INT64:
                    case GGUFMetadataValueType::FLOAT64:
                    {
                        int64_t v;
                        file.read(reinterpret_cast<char *>(&v), 8);
                        break;
                    }
                    case GGUFMetadataValueType::STRING:
                    {
                        read_gguf_string(file);
                        break;
                    }
                    case GGUFMetadataValueType::ARRAY:
                        TENSOR_THROW("Nested arrays not supported");
                    }
                }
                break;
            }
            }
        }

        std::vector<std::string> names;
        for (uint64_t i = 0; i < tensor_count; ++i)
        {
            std::string name = read_gguf_string(file);
            names.push_back(name);

            uint32_t n_dims;
            file.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            for (uint32_t d = 0; d < n_dims; ++d)
            {
                uint64_t dim;
                file.read(reinterpret_cast<char *>(&dim), sizeof(dim));
            }
            uint32_t type_u32;
            file.read(reinterpret_cast<char *>(&type_u32), sizeof(type_u32));
            uint64_t offset;
            file.read(reinterpret_cast<char *>(&offset), sizeof(offset));
        }

        return names;
    }

    using GGUFMetadataMap = std::unordered_map<std::string, GGUFMetadataValue>;

    inline GGUFMetadataMap gguf_read_metadata(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file: " + filename);

        uint32_t magic;
        file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (magic != GGUF_MAGIC)
            TENSOR_THROW("Not a valid GGUF file (bad magic)");

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(version));

        uint64_t tensor_count;
        file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));

        uint64_t metadata_kv_count;
        file.read(reinterpret_cast<char *>(&metadata_kv_count), sizeof(metadata_kv_count));

        GGUFMetadataMap result;

        for (uint64_t i = 0; i < metadata_kv_count; ++i)
        {
            std::string key = read_gguf_string(file);
            uint32_t vtype_u32;
            file.read(reinterpret_cast<char *>(&vtype_u32), sizeof(vtype_u32));
            GGUFMetadataValueType vtype = static_cast<GGUFMetadataValueType>(vtype_u32);

            switch (vtype)
            {
            case GGUFMetadataValueType::UINT8:
            {
                uint8_t v;
                file.read(reinterpret_cast<char *>(&v), 1);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::INT8:
            {
                int8_t v;
                file.read(reinterpret_cast<char *>(&v), 1);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::UINT16:
            {
                uint16_t v;
                file.read(reinterpret_cast<char *>(&v), 2);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::INT16:
            {
                int16_t v;
                file.read(reinterpret_cast<char *>(&v), 2);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::UINT32:
            {
                uint32_t v;
                file.read(reinterpret_cast<char *>(&v), 4);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::INT32:
            {
                int32_t v;
                file.read(reinterpret_cast<char *>(&v), 4);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::FLOAT32:
            {
                float v;
                file.read(reinterpret_cast<char *>(&v), 4);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::BOOL:
            {
                uint8_t v;
                file.read(reinterpret_cast<char *>(&v), 1);
                result[key] = (v != 0);
                break;
            }
            case GGUFMetadataValueType::STRING:
            {
                result[key] = read_gguf_string(file);
                break;
            }
            case GGUFMetadataValueType::UINT64:
            {
                uint64_t v;
                file.read(reinterpret_cast<char *>(&v), 8);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::INT64:
            {
                int64_t v;
                file.read(reinterpret_cast<char *>(&v), 8);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::FLOAT64:
            {
                double v;
                file.read(reinterpret_cast<char *>(&v), 8);
                result[key] = v;
                break;
            }
            case GGUFMetadataValueType::ARRAY:
            {
                uint32_t arr_type;
                file.read(reinterpret_cast<char *>(&arr_type), sizeof(arr_type));
                uint64_t arr_len;
                file.read(reinterpret_cast<char *>(&arr_len), sizeof(arr_len));
                for (uint64_t j = 0; j < arr_len; ++j)
                {
                    GGUFMetadataValueType elem_type = static_cast<GGUFMetadataValueType>(arr_type);
                    switch (elem_type)
                    {
                    case GGUFMetadataValueType::UINT8:
                    case GGUFMetadataValueType::INT8:
                    {
                        int8_t v;
                        file.read(reinterpret_cast<char *>(&v), 1);
                        break;
                    }
                    case GGUFMetadataValueType::UINT16:
                    case GGUFMetadataValueType::INT16:
                    {
                        int16_t v;
                        file.read(reinterpret_cast<char *>(&v), 2);
                        break;
                    }
                    case GGUFMetadataValueType::UINT32:
                    case GGUFMetadataValueType::INT32:
                    case GGUFMetadataValueType::FLOAT32:
                    case GGUFMetadataValueType::BOOL:
                    {
                        int32_t v;
                        file.read(reinterpret_cast<char *>(&v), 4);
                        break;
                    }
                    case GGUFMetadataValueType::UINT64:
                    case GGUFMetadataValueType::INT64:
                    case GGUFMetadataValueType::FLOAT64:
                    {
                        int64_t v;
                        file.read(reinterpret_cast<char *>(&v), 8);
                        break;
                    }
                    case GGUFMetadataValueType::STRING:
                    {
                        read_gguf_string(file);
                        break;
                    }
                    case GGUFMetadataValueType::ARRAY:
                        TENSOR_THROW("Nested arrays not supported");
                    }
                }
                break;
            }
            }
        }

        return result;
    }

} // namespace TensorN

#endif // !__GGUF__H__
