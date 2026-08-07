#pragma once
#ifndef __STATIC__H__
#define __STATIC__H__
#include "tensor.hpp"
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "cnpy/cnpy.hpp"
#include "GGUF/gguf.hpp"
#include "HF/safetensors.hpp"

template <typename T>
constexpr bool is_supported_json_type()
{
    return std::is_arithmetic_v<T>; // JSON 只支持数值类型
}
template <typename T>
constexpr bool is_supported_npy_type()
{
    return std::is_same_v<T, float> ||
           std::is_same_v<T, double> ||
           std::is_same_v<T, int32_t> ||
           std::is_same_v<T, uint8_t> ||
           std::is_same_v<T, int64_t>; // 按需扩展
}

template <typename T>
constexpr bool is_supported_pt_type()
{
    return std::is_same_v<T, float> ||
           std::is_same_v<T, double> ||
           std::is_same_v<T, int32_t> ||
           std::is_same_v<T, int64_t> ||
           std::is_same_v<T, uint8_t> ||
           std::is_same_v<T, int16_t> ||
           std::is_same_v<T, TensorN::half> ||
           std::is_same_v<T, TensorN::bfloat16> ||
           std::is_same_v<T, TensorN::tf32> ||
           std::is_same_v<T, TensorN::fp8_e4m3> ||
           std::is_same_v<T, TensorN::fp8_e5m2>;
}

enum class PTDtype : uint8_t {
    FLOAT32 = 0,
    FLOAT64 = 1,
    INT32   = 2,
    INT64   = 3,
    UINT8   = 4,
    INT16   = 5,
    FLOAT16 = 6,
    BFLOAT16 = 7,
    TF32    = 8,
    FP8_E4M3 = 9,
    FP8_E5M2 = 10,
};

constexpr const char PT_MAGIC[] = "TENSORPT!";
constexpr uint32_t PT_VERSION = 1;
constexpr uint32_t PT_VERSION_MULTI = 2;

template <typename T>
PTDtype get_pt_dtype()
{
    if constexpr (std::is_same_v<T, float>)       return PTDtype::FLOAT32;
    if constexpr (std::is_same_v<T, double>)      return PTDtype::FLOAT64;
    if constexpr (std::is_same_v<T, int32_t>)     return PTDtype::INT32;
    if constexpr (std::is_same_v<T, int64_t>)     return PTDtype::INT64;
    if constexpr (std::is_same_v<T, uint8_t>)     return PTDtype::UINT8;
    if constexpr (std::is_same_v<T, int16_t>)     return PTDtype::INT16;
    if constexpr (std::is_same_v<T, TensorN::half>)      return PTDtype::FLOAT16;
    if constexpr (std::is_same_v<T, TensorN::bfloat16>)  return PTDtype::BFLOAT16;
    if constexpr (std::is_same_v<T, TensorN::tf32>)      return PTDtype::TF32;
    if constexpr (std::is_same_v<T, TensorN::fp8_e4m3>)  return PTDtype::FP8_E4M3;
    if constexpr (std::is_same_v<T, TensorN::fp8_e5m2>)  return PTDtype::FP8_E5M2;
    TENSOR_THROW("Unsupported type for .pt format");
}
namespace TensorN
{
    template <typename T>
    void save_csv(const Tensor<T> &A, const std::string &filename)
    {
        auto &_shape = A.shape();
        if (_shape.size() > 2)
        {
            TENSOR_THROW("CSV only supports 1D or 2D tensors");
        }
        std::ofstream file(filename);
        if (!file)
            TENSOR_THROW("Cannot open file for writing: " + filename);

        size_t rows = _shape.empty() ? 1 : (_shape.size() == 1 ? 1 : _shape[0]);
        size_t cols = _shape.empty() ? 1 : (_shape.size() == 1 ? _shape[0] : _shape[1]);

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                size_t idx = (rows == 1) ? j : i * cols + j;
                file << (*A.data)[idx];
                if (j < cols - 1)
                    file << ",";
            }
            file << "\n";
        }
    }

    template <typename T>
    Tensor<T> load_csv(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file)
            TENSOR_THROW("Cannot open file: " + filename);

        std::vector<std::vector<T>> rows;
        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty())
                continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<T> row;
            while (std::getline(ss, cell, ','))
            {
                row.push_back(static_cast<T>(std::stod(cell))); // 支持浮点/整数
            }
            rows.push_back(row);
        }

        if (rows.empty())
            return Tensor<T>();

        size_t cols = rows[0].size();
        for (const auto &r : rows)
        {
            if (r.size() != cols)
                TENSOR_THROW("Inconsistent CSV columns");
        }

        std::vector<size_t> shape;
        if (rows.size() == 1 && cols == 1)
        {
            shape = {};
        }
        else if (rows.size() == 1)
        {
            shape = {cols};
        }
        else
        {
            shape = {rows.size(), cols};
        }

        std::vector<T> flat;
        for (const auto &r : rows)
        {
            flat.insert(flat.end(), r.begin(), r.end());
        }
        return Tensor(shape, flat);
    }
    template <typename T>
    void save_npy(const Tensor<T> &A, const std::string &filename)
    {
        auto &_shape = A.shape();
        if (!is_supported_npy_type<T>())
        {
            TENSOR_THROW("Type not supported for .npy");
        }
        std::vector<size_t> shape(_shape.begin(), _shape.end());
        cnpy::npy_save(filename, A.data->data(), shape, "w");
    }

    template <typename T>
    Tensor<T> load_npy(const std::string &filename)
    {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        if (arr.word_size != sizeof(T))
        {
            TENSOR_THROW("Data type size mismatch in .npy file");
        }
        std::vector<size_t> shape(arr.shape.begin(), arr.shape.end());
        std::vector<T> data_vec(arr.data<T>(), arr.data<T>() + arr.num_vals);
        return Tensor(shape, data_vec);
    }

    template <typename T>
    void save_npz(const Tensor<T> &A, const std::string &filename)
    {
        auto &_shape = A.shape();
        if (!is_supported_npy_type<T>())
        {
            TENSOR_THROW("Type not supported for .npz");
        }
        std::vector<size_t> shape(_shape.begin(), _shape.end());

        // cnpy 支持直接保存为 .npz（内部用 zlib 压缩）
        // 注意：cnpy::npz_save 要求传入 "key" 名称
        cnpy::npz_save(filename, "arr_0", A.data->data(), shape, "w");
    }
    template <typename T>
    Tensor<T> load_npz(const std::string &filename)
    {
        // 加载整个 .npz 文件为 map<string, NpyArray>
        auto npz_map = cnpy::npz_load(filename);
        if (npz_map.empty())
        {
            TENSOR_THROW("Empty or invalid .npz file: " + filename);
        }

        // 取第一个数组（兼容 np.savez(arr) 生成的 arr_0）
        const cnpy::NpyArray *arr_ptr = nullptr;
        if (npz_map.find("arr_0") != npz_map.end())
        {
            arr_ptr = &npz_map.at("arr_0");
        }
        else
        {
            // 如果没有 arr_0，取任意第一个
            arr_ptr = &npz_map.begin()->second;
        }
        const cnpy::NpyArray &arr = *arr_ptr;

        // 类型检查
        if (arr.word_size != sizeof(T))
        {
            TENSOR_THROW(
                "Data type size mismatch in .npz file. "
                "Expected: " +
                std::to_string(sizeof(T)) +
                ", got: " + std::to_string(arr.word_size));
        }

        // 构造 shape 和 data
        std::vector<size_t> shape(arr.shape.begin(), arr.shape.end());
        std::vector<T> data_vec(arr.data<T>(), arr.data<T>() + arr.num_vals);
        return Tensor(shape, data_vec);
    }

    template <typename T>
    void save_pt(const Tensor<T> &A, const std::string &filename)
    {
        if (!is_supported_pt_type<T>())
        {
            TENSOR_THROW("Type not supported for .pt");
        }
        std::ofstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file for writing: " + filename);

        const auto &_shape = A.shape();
        auto dtype = get_pt_dtype<T>();

        file.write(PT_MAGIC, 9);

        uint32_t version = PT_VERSION;
        file.write(reinterpret_cast<const char *>(&version), sizeof(version));

        uint8_t dtype_byte = static_cast<uint8_t>(dtype);
        file.write(reinterpret_cast<const char *>(&dtype_byte), sizeof(dtype_byte));

        uint32_t ndims = static_cast<uint32_t>(_shape.size());
        file.write(reinterpret_cast<const char *>(&ndims), sizeof(ndims));

        for (auto dim : _shape)
        {
            uint64_t dim64 = static_cast<uint64_t>(dim);
            file.write(reinterpret_cast<const char *>(&dim64), sizeof(dim64));
        }

        file.write(reinterpret_cast<const char *>(A.data->data()), A.data->size() * sizeof(T));
    }

    template <typename T>
    void save_pt_multi(
        const std::vector<std::pair<std::string, Tensor<T>>> &tensors,
        const std::string &filename)
    {
        if (!is_supported_pt_type<T>())
        {
            TENSOR_THROW("Type not supported for .pt format");
        }
        if (tensors.empty())
        {
            TENSOR_THROW("No tensors to save");
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file for writing: " + filename);

        auto dtype = get_pt_dtype<T>();

        file.write(PT_MAGIC, 9);

        uint32_t version = PT_VERSION_MULTI;
        file.write(reinterpret_cast<const char *>(&version), sizeof(version));

        uint64_t tensor_count = static_cast<uint64_t>(tensors.size());
        file.write(reinterpret_cast<const char *>(&tensor_count), sizeof(tensor_count));

        uint64_t current_data_offset = 0;
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            const auto &[name, tensor] = tensors[i];

            uint32_t name_len = static_cast<uint32_t>(name.size());
            file.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
            file.write(name.data(), name_len);

            uint8_t dtype_byte = static_cast<uint8_t>(dtype);
            file.write(reinterpret_cast<const char *>(&dtype_byte), sizeof(dtype_byte));

            const auto &shape = tensor.shape();
            uint32_t ndims = static_cast<uint32_t>(shape.size());
            file.write(reinterpret_cast<const char *>(&ndims), sizeof(ndims));

            for (auto dim : shape)
            {
                uint64_t dim64 = static_cast<uint64_t>(dim);
                file.write(reinterpret_cast<const char *>(&dim64), sizeof(dim64));
            }

            file.write(reinterpret_cast<const char *>(&current_data_offset), sizeof(uint64_t));

            size_t raw_size = tensor.data->size() * sizeof(T);
            file.write(reinterpret_cast<const char *>(&raw_size), sizeof(uint64_t));

            current_data_offset += raw_size;
        }

        for (size_t i = 0; i < tensors.size(); ++i)
        {
            const auto &[name, tensor] = tensors[i];
            file.write(reinterpret_cast<const char *>(tensor.data->data()),
                       static_cast<std::streamsize>(tensor.data->size() * sizeof(T)));
        }

        if (!file)
            TENSOR_THROW("Error writing .pt file: " + filename);
    }

    template <typename T>
    std::unordered_map<std::string, Tensor<T>> load_pt_multi(const std::string &filename)
    {
        if (!is_supported_pt_type<T>())
        {
            TENSOR_THROW("Type not supported for .pt format");
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file: " + filename);

        char magic_buf[9];
        file.read(magic_buf, 9);
        if (std::memcmp(magic_buf, PT_MAGIC, 9) != 0)
        {
            TENSOR_THROW("Not a valid TensorN .pt file (bad magic)");
        }

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(version));
        if (version != PT_VERSION_MULTI)
        {
            TENSOR_THROW("Not a multi-tensor .pt file (version=" + std::to_string(version) + ")");
        }

        uint64_t tensor_count;
        file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));
        if (tensor_count == 0)
        {
            TENSOR_THROW("No tensors found in .pt file");
        }

        struct TensorInfo
        {
            std::string name;
            PTDtype dtype;
            std::vector<size_t> shape;
            uint64_t data_offset;
            uint64_t data_size;
        };

        std::vector<TensorInfo> infos(tensor_count);
        PTDtype expected_dtype = get_pt_dtype<T>();

        for (uint64_t i = 0; i < tensor_count; ++i)
        {
            uint32_t name_len;
            file.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
            infos[i].name.resize(name_len);
            file.read(&infos[i].name[0], name_len);

            uint8_t dtype_byte;
            file.read(reinterpret_cast<char *>(&dtype_byte), sizeof(dtype_byte));
            infos[i].dtype = static_cast<PTDtype>(dtype_byte);

            if (infos[i].dtype != expected_dtype)
            {
                TENSOR_THROW(
                    "Type mismatch for tensor '" + infos[i].name +
                    "' in .pt file");
            }

            uint32_t ndims;
            file.read(reinterpret_cast<char *>(&ndims), sizeof(ndims));

            infos[i].shape.resize(ndims);
            for (uint32_t d = 0; d < ndims; ++d)
            {
                uint64_t dim;
                file.read(reinterpret_cast<char *>(&dim), sizeof(dim));
                infos[i].shape[d] = static_cast<size_t>(dim);
            }

            file.read(reinterpret_cast<char *>(&infos[i].data_offset), sizeof(uint64_t));
            file.read(reinterpret_cast<char *>(&infos[i].data_size), sizeof(uint64_t));
        }

        uint64_t data_base = static_cast<uint64_t>(file.tellg());

        std::unordered_map<std::string, Tensor<T>> result;

        for (const auto &info : infos)
        {
            uint64_t abs_offset = data_base + info.data_offset;
            file.seekg(static_cast<std::streamoff>(abs_offset));

            size_t total_elements = info.shape.empty() ? 1 : 1;
            for (auto dim : info.shape)
                total_elements *= dim;

            std::vector<T> data_vec(total_elements);
            file.read(reinterpret_cast<char *>(data_vec.data()),
                      static_cast<std::streamsize>(total_elements * sizeof(T)));

            if (!file)
                TENSOR_THROW("Error reading tensor data from .pt file");

            result[info.name] = Tensor<T>(info.shape, data_vec);
        }

        return result;
    }

    inline std::vector<std::string> pt_list_tensors(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file: " + filename);

        char magic_buf[9];
        file.read(magic_buf, 9);
        if (std::memcmp(magic_buf, PT_MAGIC, 9) != 0)
        {
            TENSOR_THROW("Not a valid TensorN .pt file (bad magic)");
        }

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(version));
        if (version != PT_VERSION_MULTI)
        {
            if (version == PT_VERSION)
            {
                return {"tensor"};
            }
            TENSOR_THROW("Unsupported .pt version: " + std::to_string(version));
        }

        uint64_t tensor_count;
        file.read(reinterpret_cast<char *>(&tensor_count), sizeof(tensor_count));

        std::vector<std::string> names;
        for (uint64_t i = 0; i < tensor_count; ++i)
        {
            uint32_t name_len;
            file.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            file.read(&name[0], name_len);
            names.push_back(name);

            uint8_t dtype_byte;
            file.read(reinterpret_cast<char *>(&dtype_byte), sizeof(dtype_byte));

            uint32_t ndims;
            file.read(reinterpret_cast<char *>(&ndims), sizeof(ndims));
            for (uint32_t d = 0; d < ndims; ++d)
            {
                uint64_t dim;
                file.read(reinterpret_cast<char *>(&dim), sizeof(dim));
            }

            uint64_t data_offset, data_size;
            file.read(reinterpret_cast<char *>(&data_offset), sizeof(uint64_t));
            file.read(reinterpret_cast<char *>(&data_size), sizeof(uint64_t));
        }

        return names;
    }

    template <typename T>
    Tensor<T> load_pt(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            TENSOR_THROW("Cannot open file: " + filename);

        char magic_buf[9];
        file.read(magic_buf, 9);
        if (std::memcmp(magic_buf, PT_MAGIC, 9) != 0)
        {
            TENSOR_THROW("Not a valid TensorN .pt file (bad magic)");
        }

        uint32_t version;
        file.read(reinterpret_cast<char *>(&version), sizeof(version));

        if (version == PT_VERSION_MULTI)
        {
            file.seekg(0);
            auto tensors = load_pt_multi<T>(filename);
            if (tensors.empty())
                TENSOR_THROW("No tensors in multi-tensor .pt file");
            return tensors.begin()->second;
        }

        if (version != PT_VERSION)
        {
            TENSOR_THROW("Unsupported .pt version: " + std::to_string(version));
        }

        uint8_t dtype_byte;
        file.read(reinterpret_cast<char *>(&dtype_byte), sizeof(dtype_byte));
        PTDtype stored_dtype = static_cast<PTDtype>(dtype_byte);
        PTDtype expected = get_pt_dtype<T>();
        if (stored_dtype != expected)
        {
            TENSOR_THROW("Type mismatch in .pt file");
        }

        uint32_t ndims;
        file.read(reinterpret_cast<char *>(&ndims), sizeof(ndims));

        std::vector<size_t> shape(ndims);
        size_t total = ndims == 0 ? 1 : 1;
        for (uint32_t i = 0; i < ndims; ++i)
        {
            uint64_t dim;
            file.read(reinterpret_cast<char *>(&dim), sizeof(dim));
            shape[i] = static_cast<size_t>(dim);
            total *= shape[i];
        }

        if (ndims == 0)
        {
            shape = {};
        }

        std::vector<T> data_vec(total);
        file.read(reinterpret_cast<char *>(data_vec.data()), total * sizeof(T));

        return Tensor<T>(shape, data_vec);
    }

    template <typename T>
    void save_json(const Tensor<T> &A, const std::string &filename)
    {
        if constexpr (is_supported_json_type<T>())
        {
            auto &_shape = A.shape();
            nlohmann::json j;
            j["shape"] = _shape;
            j["data"] = *A.data;
            std::ofstream file(filename);
            file << j.dump(2); // pretty print
        }
        else
        {
            TENSOR_THROW("Only arithmetic types supported for JSON");
        }
    }

    template <typename T>
    Tensor<T> load_json(const std::string &filename)
    {
        if constexpr (is_supported_json_type<T>())
        {
            std::ifstream file(filename);
            nlohmann::json j;
            file >> j;
            std::vector<size_t> shape = j["shape"].get<std::vector<size_t>>();
            std::vector<T> data_vec = j["data"].get<std::vector<T>>();
            return Tensor(shape, data_vec);
        }
        else
        {
            TENSOR_THROW("Only arithmetic types supported for JSON");
        }
    }
    template <typename T>
    void Tensor<T>::save(const std::string &filename, const std::string &format) const
    {
        std::string fmt = format;
        if (fmt == "auto")
        {
            if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".csv")
                fmt = "csv";
            else if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npy")
                fmt = "npy";
            else if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npz")
                fmt = "npz";
            else if (filename.size() >= 3 && filename.substr(filename.size() - 3) == ".pt")
                fmt = "pt";
            else if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".pth")
                fmt = "pt";
            else if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".json")
                fmt = "json";
            else if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".gguf")
                fmt = "gguf";
            else if (filename.size() >= 12 && filename.substr(filename.size() - 12) == ".safetensors")
                fmt = "safetensors";
            else
                TENSOR_THROW("Cannot infer format from filename: " + filename);
        }

        if (fmt == "csv")
        {
            save_csv<T>(*this, filename);
        }
        else if (fmt == "npy")
        {
            save_npy<T>(*this, filename);
        }
        else if (fmt == "npz")
        {
            save_npz<T>(*this, filename);
        }
        else if (fmt == "pt")
        {
            save_pt<T>(*this, filename);
        }
        else if (fmt == "json")
        {
            save_json<T>(*this, filename);
        }
        else if (fmt == "gguf")
        {
            save_gguf<T>(*this, filename);
        }
        else if (fmt == "safetensors")
        {
            save_safetensors<T>(*this, filename);
        }
        else
        {
            TENSOR_THROW("Unsupported format: " + fmt);
        }
    }
    template <typename T>
    Tensor<T> load(const std::string &filename, const std::string &format = "auto")
    {
        std::string fmt = format;
        if (fmt == "auto")
        {
            if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".csv")
                fmt = "csv";
            else if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npy")
                fmt = "npy";
            else if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".npz")
                fmt = "npz";
            else if (filename.size() >= 3 && filename.substr(filename.size() - 3) == ".pt")
                fmt = "pt";
            else if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".pth")
                fmt = "pt";
            else if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".json")
                fmt = "json";
            else if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".gguf")
                fmt = "gguf";
            else if (filename.size() >= 12 && filename.substr(filename.size() - 12) == ".safetensors")
                fmt = "safetensors";
            else
                TENSOR_THROW("Cannot infer format from filename: " + filename);
        }

        if (fmt == "csv")
        {
            return load_csv<T>(filename);
        }
        else if (fmt == "npy")
        {
            return load_npy<T>(filename);
        }
        else if (fmt == "npz")
        {
            return load_npz<T>(filename);
        }
        else if (fmt == "pt")
        {
            return load_pt<T>(filename);
        }
        else if (fmt == "json")
        {
            return load_json<T>(filename);
        }
        else if (fmt == "gguf")
        {
            return load_gguf<T>(filename);
        }
        else if (fmt == "safetensors")
        {
            return load_safetensors<T>(filename);
        }
        else
        {
            TENSOR_THROW("Unsupported format: " + fmt);
        }
    }
} // TensorN

#endif //!__STATIC__H__