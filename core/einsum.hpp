#pragma once
#ifndef __EINSUM__H__
#define __EINSUM__H__

#include "tensor.hpp"

namespace TensorN
{
    // 辅助函数：解析einsum表达式
    std::vector<std::string> parse_einsum_expression(const std::string &exp)
    {
        // 移除空格
        std::string cleaned;
        for (char c : exp)
        {
            if (!isspace(c))
                cleaned += c;
        }

        // 查找箭头
        size_t arrow_pos = cleaned.find("->");
        if (arrow_pos == std::string::npos)
        {
            throw std::invalid_argument("Invalid einsum expression: missing '->'");
        }

        // 分割输入和输出部分
        std::string input_part = cleaned.substr(0, arrow_pos);
        std::string output_part = cleaned.substr(arrow_pos + 2);

        // 分割多个输入
        std::vector<std::string> inputs;
        size_t start = 0;
        size_t comma_pos;

        while ((comma_pos = input_part.find(',', start)) != std::string::npos)
        {
            inputs.push_back(input_part.substr(start, comma_pos - start));
            start = comma_pos + 1;
        }
        inputs.push_back(input_part.substr(start));

        // 添加输出部分作为最后一个元素
        inputs.push_back(output_part);

        return inputs;
    }

    // 辅助函数：计算多维索引的线性偏移
    size_t compute_index(const std::vector<size_t> &indices,
                         const std::vector<size_t> &shape,
                         const std::vector<size_t> &strides)
    {
        size_t idx = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    // 辅助函数：计算张量的strides
    std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
    {
        std::vector<size_t> strides(shape.size(), 1);
        for (int i = shape.size() - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    // 辅助函数：获取下一个索引
    bool increment_index(std::vector<size_t> &indices,
                         const std::vector<size_t> &limits)
    {
        for (int i = indices.size() - 1; i >= 0; --i)
        {
            if (indices[i] + 1 < limits[i])
            {
                indices[i]++;
                return true;
            }
            indices[i] = 0;
        }
        return false;
    }

    template <typename T>
    opt<T> einsum(std::string exp, const Tensor<T> &A)
    {
        // 移除空格
        std::string cleaned;
        for (char c : exp)
        {
            if (!isspace(c))
                cleaned += c;
        }

        // 查找箭头
        size_t arrow_pos = cleaned.find("->");
        if (arrow_pos == std::string::npos)
        {
            throw std::invalid_argument("Invalid einsum expression: missing '->'");
        }

        std::string input_labels = cleaned.substr(0, arrow_pos);
        std::string output_labels = cleaned.substr(arrow_pos + 2);

        // 验证标签长度与维度匹配
        if (input_labels.size() != A.shape().size())
        {
            throw std::invalid_argument("Label count doesn't match tensor dimensions");
        }

        // 特殊情况：标量输出（求迹、求和等）
        if (output_labels.empty())
        {
            // 处理求迹操作：ii->
            if (input_labels.size() == 2 && input_labels[0] == input_labels[1])
            {
                // 必须是方阵才能求迹
                if (A.shape().size() != 2 || A.shape()[0] != A.shape()[1])
                {
                    throw std::invalid_argument("Trace requires a square matrix");
                }

                T trace = T(0);
                for (size_t i = 0; i < A.shape()[0]; ++i)
                {
                    trace += A[{i, i}];
                }

                // 创建标量张量
                Tensor<T> result({1});
                result.data[0] = trace;
                return opt<T>(result);
            }
            // 求和所有元素：...-> 或 ij-> 等
            else
            {
                T sum = T(0);
                for (size_t i = 0; i < A.size(); ++i)
                {
                    sum += A.data[i];
                }

                // 创建标量张量
                Tensor<T> result({1});
                result.data[0] = sum;
                return opt<T>(result);
            }
        }
        // 输出不是标量（转置、切片等操作）
        else
        {
            // 检查输出标签是否只是输入标签的重排
            std::string sorted_input = input_labels;
            std::string sorted_output = output_labels;
            std::sort(sorted_input.begin(), sorted_input.end());
            std::sort(sorted_output.begin(), sorted_output.end());

            if (sorted_input != sorted_output)
            {
                throw std::invalid_argument("Invalid einsum expression for single tensor operation");
            }

            // 确定输出形状
            std::vector<size_t> output_shape(output_labels.size());
            std::unordered_map<char, size_t> label_to_dim;

            // 构建标签到维度的映射
            for (size_t i = 0; i < input_labels.size(); ++i)
            {
                label_to_dim[input_labels[i]] = A.shape()[i];
            }

            for (size_t i = 0; i < output_labels.size(); ++i)
            {
                auto it = label_to_dim.find(output_labels[i]);
                if (it == label_to_dim.end())
                {
                    throw std::invalid_argument("Output label not found in input labels");
                }
                output_shape[i] = it->second;
            }

            // 创建输出张量
            Tensor<T> result(output_shape);

            // 计算strides
            std::vector<size_t> strides_A = compute_strides(A.shape());
            std::vector<size_t> strides_out = compute_strides(output_shape);

            // 构建输入和输出的索引映射
            std::vector<size_t> index_map_A(input_labels.size());
            std::vector<size_t> index_map_out(output_labels.size());

            // 为每个标签找到在输入和输出中的位置
            std::unordered_map<char, size_t> label_to_pos;
            for (size_t i = 0; i < input_labels.size(); ++i)
            {
                label_to_pos[input_labels[i]] = i;
            }

            for (size_t i = 0; i < output_labels.size(); ++i)
            {
                auto it = label_to_pos.find(output_labels[i]);
                if (it == label_to_pos.end())
                {
                    throw std::invalid_argument("Label mapping error");
                }
                index_map_out[i] = it->second;
            }

            // 遍历所有可能的索引组合
            std::vector<size_t> index_limits;

            for (char label : input_labels)
            {
                index_limits.push_back(label_to_dim[label]);
            }

            std::vector<size_t> indices(input_labels.size(), 0);

            do
            {
                // 计算输入索引
                std::vector<size_t> indices_A(input_labels.size());
                for (size_t i = 0; i < input_labels.size(); ++i)
                {
                    indices_A[i] = indices[i];
                }

                // 计算输出索引
                std::vector<size_t> indices_out(output_labels.size());
                for (size_t i = 0; i < output_labels.size(); ++i)
                {
                    indices_out[i] = indices[index_map_out[i]];
                }

                // 计算线性偏移
                size_t idx_A = compute_index(indices_A, A.shape(), strides_A);
                size_t idx_out = compute_index(indices_out, output_shape, strides_out);

                // 复制数据
                result.data[idx_out] = A.data[idx_A];

            } while (increment_index(indices, index_limits));

            return opt<T>(result);
        }
    }
    // 主einsum函数实现
    template <typename T>
    opt<T> einsum(std::string exp, const Tensor<T> &A, const Tensor<T> &B)
    {
        // 解析表达式
        auto parsed = parse_einsum_expression(exp);
        if (parsed.size() != 3)
        {
            throw std::invalid_argument("einsum currently only supports two input tensors");
        }

        const std::string &labelsA = parsed[0];
        const std::string &labelsB = parsed[1];
        const std::string &labelsOut = parsed[2];

        // 验证标签长度与维度匹配
        if (labelsA.size() != A.shape().size())
        {
            throw std::invalid_argument("Label count doesn't match tensor A dimensions");
        }
        if (labelsB.size() != B.shape().size())
        {
            throw std::invalid_argument("Label count doesn't match tensor B dimensions");
        }

        // 构建标签到维度的映射
        std::unordered_map<char, size_t> label_to_dim;
        std::unordered_map<char, std::vector<size_t>> label_to_tensor_indices;

        // 处理张量A的标签
        for (size_t i = 0; i < labelsA.size(); ++i)
        {
            char label = labelsA[i];
            label_to_dim[label] = A.shape()[i];
            label_to_tensor_indices[label].push_back(0); // 0表示来自A
            label_to_tensor_indices[label].push_back(i); // 维度索引
        }

        // 处理张量B的标签
        for (size_t i = 0; i < labelsB.size(); ++i)
        {
            char label = labelsB[i];
            auto it = label_to_dim.find(label);
            if (it != label_to_dim.end())
            {
                // 标签已存在，检查维度是否匹配
                if (it->second != B.shape()[i])
                {
                    throw std::invalid_argument("Dimension mismatch for shared label");
                }
                label_to_tensor_indices[label].push_back(1); // 1表示来自B
                label_to_tensor_indices[label].push_back(i);
            }
            else
            {
                label_to_dim[label] = B.shape()[i];
                label_to_tensor_indices[label].push_back(1);
                label_to_tensor_indices[label].push_back(i);
            }
        }

        // 确定输出形状
        std::vector<size_t> output_shape;
        std::vector<char> output_labels;
        for (char label : labelsOut)
        {
            output_labels.push_back(label);
            auto it = label_to_dim.find(label);
            if (it == label_to_dim.end())
            {
                throw std::invalid_argument("Output label not found in input labels");
            }
            output_shape.push_back(it->second);
        }

        // 确定求和标签（在输入中出现但不在输出中出现的标签）
        std::vector<char> sum_labels;
        for (const auto &pair : label_to_dim)
        {
            char label = pair.first;
            if (labelsOut.find(label) == std::string::npos)
            {
                sum_labels.push_back(label);
            }
        }

        // 创建输出张量
        Tensor<T> result(output_shape);

        // 获取strides
        std::vector<size_t> stridesA = compute_strides(A.shape());
        std::vector<size_t> stridesB = compute_strides(B.shape());
        std::vector<size_t> stridesOut = compute_strides(output_shape);

        // 准备索引遍历
        // 构建完整的标签列表（输出标签 + 求和标签）
        std::vector<char> all_labels = output_labels;
        all_labels.insert(all_labels.end(), sum_labels.begin(), sum_labels.end());

        // 为每个标签创建索引限制
        std::vector<size_t> index_limits;
        for (char label : all_labels)
        {
            index_limits.push_back(label_to_dim[label]);
        }

        // 初始化索引
        std::vector<size_t> indices(all_labels.size(), 0);

        // 为每个张量构建索引映射
        std::vector<size_t> index_map_A(labelsA.size());
        std::vector<size_t> index_map_B(labelsB.size());
        std::vector<size_t> index_map_out(output_labels.size());

        for (size_t i = 0; i < labelsA.size(); ++i)
        {
            char label = labelsA[i];
            // 在all_labels中找到这个标签的位置
            auto it = std::find(all_labels.begin(), all_labels.end(), label);
            index_map_A[i] = std::distance(all_labels.begin(), it);
        }

        for (size_t i = 0; i < labelsB.size(); ++i)
        {
            char label = labelsB[i];
            auto it = std::find(all_labels.begin(), all_labels.end(), label);
            index_map_B[i] = std::distance(all_labels.begin(), it);
        }

        for (size_t i = 0; i < output_labels.size(); ++i)
        {
            char label = output_labels[i];
            auto it = std::find(all_labels.begin(), all_labels.end(), label);
            index_map_out[i] = std::distance(all_labels.begin(), it);
        }

        // 执行张量收缩
        do
        {
            // 计算A的索引
            std::vector<size_t> indices_A(labelsA.size());
            for (size_t i = 0; i < labelsA.size(); ++i)
            {
                indices_A[i] = indices[index_map_A[i]];
            }

            // 计算B的索引
            std::vector<size_t> indices_B(labelsB.size());
            for (size_t i = 0; i < labelsB.size(); ++i)
            {
                indices_B[i] = indices[index_map_B[i]];
            }

            // 计算输出索引
            std::vector<size_t> indices_out(output_labels.size());
            for (size_t i = 0; i < output_labels.size(); ++i)
            {
                indices_out[i] = indices[index_map_out[i]];
            }

            // 计算线性偏移
            size_t idx_A = compute_index(indices_A, A.shape(), stridesA);
            size_t idx_B = compute_index(indices_B, B.shape(), stridesB);
            size_t idx_out = compute_index(indices_out, output_shape, stridesOut);

            // 累加到输出
            result.data[idx_out] += A.data[idx_A] * B.data[idx_B];

        } while (increment_index(indices, index_limits));

        return opt<T>(result);
    }

    // 常用einsum操作的便捷函数

    // 矩阵乘法
    template <typename T>
    opt<T> matmul(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 2 || B.shape().size() != 2)
        {
            throw std::invalid_argument("matmul requires 2D tensors");
        }
        if (A.shape()[1] != B.shape()[0])
        {
            throw std::invalid_argument("Dimension mismatch for matrix multiplication");
        }
        return einsum<T>("ij,jk->ik", A, B);
    }

    // 向量点积
    template <typename T>
    opt<T> dot(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 1 || B.shape().size() != 1)
        {
            throw std::invalid_argument("dot requires 1D tensors");
        }
        if (A.shape()[0] != B.shape()[0])
        {
            throw std::invalid_argument("Dimension mismatch for dot product");
        }
        return einsum<T>("i,i->", A, B);
    }

    // 逐元素乘法
    template <typename T>
    opt<T> mul(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (!A.is_isomorphic(B))
        {
            throw std::invalid_argument("Tensors must have same shape for element-wise multiplication");
        }
        return einsum<T>("...,...->...", A, B);
    }

    // 外积
    template <typename T>
    opt<T> outer(const Tensor<T> &A, const Tensor<T> &B)
    {
        if (A.shape().size() != 1 || B.shape().size() != 1)
        {
            throw std::invalid_argument("outer requires 1D tensors");
        }
        return einsum<T>("i,j->ij", A, B);
    }
} // namespace TensorN

#endif //!__EINSUM__H__