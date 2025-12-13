#pragma once
#ifndef __STATIC__H__
#define __STATIC__H__
#include "tensor.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <nlohmann/json.hpp>
#include "cnpy/cnpy.hpp"

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
namespace TensorN
{
    template <typename T>
    void save_csv(const Tensor<T> &A, const std::string &filename)
    {
        auto &_shape = A.shape();
        if (_shape.size() > 2)
        {
            throw std::runtime_error("CSV only supports 1D or 2D tensors");
        }
        std::ofstream file(filename);
        if (!file)
            throw std::runtime_error("Cannot open file for writing: " + filename);

        size_t rows = _shape.empty() ? 1 : (_shape.size() == 1 ? 1 : _shape[0]);
        size_t cols = _shape.empty() ? 1 : (_shape.size() == 1 ? _shape[0] : _shape[1]);

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                size_t idx = (rows == 1) ? j : i * cols + j;
                file << A.data[idx];
                if (j < cols - 1)
                    file << ",";
            }
            file << "\n";
        }
    }

    template <typename T>
    static Tensor<T> load_csv(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file)
            throw std::runtime_error("Cannot open file: " + filename);

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
                throw std::runtime_error("Inconsistent CSV columns");
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
            throw std::runtime_error("Type not supported for .npy");
        }
        std::vector<size_t> shape(_shape.begin(), _shape.end());
        cnpy::npy_save(filename, A.data.data(), shape, "w");
    }

    template <typename T>
    static Tensor<T> load_npy(const std::string &filename)
    {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        if (arr.word_size != sizeof(T))
        {
            throw std::runtime_error("Data type size mismatch in .npy file");
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
            throw std::runtime_error("Type not supported for .npz");
        }
        std::vector<size_t> shape(_shape.begin(), _shape.end());

        // cnpy 支持直接保存为 .npz（内部用 zlib 压缩）
        // 注意：cnpy::npz_save 要求传入 "key" 名称
        cnpy::npz_save(filename, "arr_0", A.data.data(), shape, "w");
    }
    template <typename T>
    static Tensor<T> load_npz(const std::string &filename)
    {
        // 加载整个 .npz 文件为 map<string, NpyArray>
        auto npz_map = cnpy::npz_load(filename);
        if (npz_map.empty())
        {
            throw std::runtime_error("Empty or invalid .npz file: " + filename);
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
            throw std::runtime_error(
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
    void save_json(const Tensor<T> &A, const std::string &filename)
    {
        auto &_shape = A.shape();
        if (!is_supported_json_type<T>())
        {
            throw std::runtime_error("Only arithmetic types supported for JSON");
        }
        nlohmann::json j;
        j["shape"] = _shape;
        j["data"] = A.data; // json.hpp 支持 vector 序列化
        std::ofstream file(filename);
        file << j.dump(2); // pretty print
    }

    template <typename T>
    static Tensor<T> load_json(const std::string &filename)
    {
        std::ifstream file(filename);
        nlohmann::json j;
        file >> j;
        std::vector<size_t> shape = j["shape"].get<std::vector<size_t>>();
        std::vector<T> data_vec = j["data"].get<std::vector<T>>();
        return Tensor(shape, data_vec);
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
            else if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".json")
                fmt = "json";
            else
                throw std::invalid_argument("Cannot infer format from filename: " + filename);
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
            return save_npz<T>(*this, filename);
        }
        else if (fmt == "json")
        {
            save_json<T>(*this, filename);
        }
        else
        {
            throw std::invalid_argument("Unsupported format: " + fmt);
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
            else if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".json")
                fmt = "json";
            else
                throw std::invalid_argument("Cannot infer format from filename: " + filename);
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
        else if (fmt == "json")
        {
            return load_json<T>(filename);
        }
        else
        {
            throw std::invalid_argument("Unsupported format: " + fmt);
        }
    }
} // TensorN

#endif //!__STATIC__H__