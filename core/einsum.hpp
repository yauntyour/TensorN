#pragma once
#ifndef __EINSUM__H__
#define __EINSUM__H__

#include "tensor.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cassert>

namespace TensorN
{
    namespace einsum_tools
    {
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

        std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
        {
            std::vector<size_t> strides(shape.size(), 1);
            for (int i = shape.size() - 2; i >= 0; --i)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            return strides;
        }

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

        std::pair<std::vector<std::string>, std::string> encoder(const std::string &exp)
        {
            std::string cleaned;
            for (char c : exp)
            {
                if (!isspace(c))
                    cleaned += c;
            }

            size_t arrow_pos = cleaned.find("->");
            if (arrow_pos == std::string::npos)
            {
                throw std::invalid_argument("Invalid einsum expression: missing '->'");
            }

            std::string input_part = cleaned.substr(0, arrow_pos);
            std::string output_part = cleaned.substr(arrow_pos + 2);

            std::vector<std::string> inputs;
            size_t start = 0;
            size_t comma_pos;

            while ((comma_pos = input_part.find(',', start)) != std::string::npos)
            {
                inputs.push_back(input_part.substr(start, comma_pos - start));
                start = comma_pos + 1;
            }
            inputs.push_back(input_part.substr(start));

            // 验证 ellipsis 的正确性
            for (const auto &input : inputs)
            {
                size_t ellipsis_count = 0;
                for (size_t i = 0; i < input.size(); ++i)
                {
                    if (input[i] == '.')
                    {
                        // 检查是否是连续的三个点
                        if (i + 2 < input.size() && input[i + 1] == '.' && input[i + 2] == '.')
                        {
                            ellipsis_count++;
                            i += 2; // 跳过后续两个点
                        }
                        else
                        {
                            throw std::invalid_argument("Invalid ellipsis in einsum expression");
                        }
                    }
                }
                if (ellipsis_count > 1)
                {
                    throw std::invalid_argument("Multiple ellipsis in single input tensor");
                }
            }

            // 验证输出中的 ellipsis
            size_t ellipsis_count = 0;
            for (size_t i = 0; i < output_part.size(); ++i)
            {
                if (output_part[i] == '.')
                {
                    if (i + 2 < output_part.size() && output_part[i + 1] == '.' && output_part[i + 2] == '.')
                    {
                        ellipsis_count++;
                        i += 2;
                    }
                    else
                    {
                        throw std::invalid_argument("Invalid ellipsis in output expression");
                    }
                }
            }
            if (ellipsis_count > 1)
            {
                throw std::invalid_argument("Multiple ellipsis in output");
            }

            return std::make_pair(inputs, output_part);
        }

        // 解析索引字母到维度的映射
        template <typename T>
        std::unordered_map<char, size_t> index_encoder(
            const std::vector<std::string> &input_labels,
            const std::vector<const Tensor<T> *> &tensors,
            std::vector<std::vector<size_t>> &ellipsis_dims,
            std::vector<std::string> &expanded_labels)
        {
            std::unordered_map<char, size_t> label_to_size;
            ellipsis_dims.clear();
            ellipsis_dims.resize(input_labels.size());
            expanded_labels.resize(input_labels.size());

            // 首先收集所有非ellipsis标签的维度
            for (size_t i = 0; i < input_labels.size(); ++i)
            {
                const std::string &labels = input_labels[i];
                const auto &shape = tensors[i]->shape();

                size_t explicit_dims = 0;
                for (size_t j = 0; j < labels.size(); ++j)
                {
                    if (labels[j] == '.' && j + 2 < labels.size() &&
                        labels[j + 1] == '.' && labels[j + 2] == '.')
                    {
                        // 跳过ellipsis
                        j += 2;
                    }
                    else
                    {
                        explicit_dims++;
                    }
                }

                if (explicit_dims > shape.size())
                {
                    throw std::invalid_argument("Number of explicit labels exceeds tensor dimension");
                }

                size_t ellipsis_ndim = shape.size() - explicit_dims;
                ellipsis_dims[i].push_back(ellipsis_ndim);

                // 构建展开后的标签字符串
                std::string expanded;
                for (size_t j = 0; j < labels.size(); ++j)
                {
                    if (labels[j] == '.' && j + 2 < labels.size() &&
                        labels[j + 1] == '.' && labels[j + 2] == '.')
                    {
                        // 为ellipsis生成匿名标签
                        for (size_t k = 0; k < ellipsis_ndim; ++k)
                        {
                            char anonymous_label = 'A' + k; // 使用大写字母作为匿名标签
                            expanded += anonymous_label;
                            // 为匿名标签设置维度
                            if (label_to_size.find(anonymous_label) == label_to_size.end())
                            {
                                label_to_size[anonymous_label] = shape[explicit_dims + k];
                            }
                            else
                            {
                                // 检查维度一致性
                                if (label_to_size[anonymous_label] != shape[explicit_dims + k])
                                {
                                    throw std::invalid_argument("Inconsistent dimension size in ellipsis");
                                }
                            }
                        }
                        j += 2; // 跳过后续两个点
                    }
                    else
                    {
                        expanded += labels[j];
                        size_t dim_idx = expanded.size() - 1;
                        if (dim_idx >= shape.size())
                        {
                            throw std::invalid_argument("Label index out of bounds");
                        }

                        if (label_to_size.find(labels[j]) != label_to_size.end())
                        {
                            // 检查维度是否一致
                            if (label_to_size[labels[j]] != shape[dim_idx])
                            {
                                throw std::invalid_argument("Inconsistent dimension size for label '" +
                                                            std::string(1, labels[j]) + "'");
                            }
                        }
                        else
                        {
                            label_to_size[labels[j]] = shape[dim_idx];
                        }
                    }
                }

                expanded_labels[i] = expanded;
            }

            return label_to_size;
        }
    } // namespace einsum_tools

    template <typename T, typename... Tensors>
    opt<T> einsum(const std::string &exp, const Tensor<T> &A, const Tensors &...tensors)
    {
        std::vector<const Tensor<T> *> tensor_list{&A, &(tensors)...};
        return einsum_multi(exp, tensor_list);
    }

    template <typename T>
    opt<T> einsum_multi(const std::string &exp, const std::vector<const Tensor<T> *> &tensors)
    {
        using namespace einsum_tools;

        // 1. 解析表达式
        auto parts = encoder(exp);
        if (parts.first.size() < 1 || parts.first.size() != tensors.size())
        {
            throw std::invalid_argument("Invalid number of input tensors in einsum expression");
        }

        // 提取输入标签和输出标签
        std::vector<std::string> &input_labels = parts.first;
        std::string &output_labels = parts.second;

        // 2. 解析标签并检查一致性（现在处理ellipsis）
        std::vector<std::vector<size_t>> ellipsis_dims;
        std::vector<std::string> expanded_input_labels;
        auto label_to_size = index_encoder(input_labels, tensors, ellipsis_dims, expanded_input_labels);

        // 3. 处理输出标签中的ellipsis
        std::string expanded_output_labels;
        size_t output_explicit_dims = 0;
        for (size_t i = 0; i < output_labels.size(); ++i)
        {
            if (output_labels[i] == '.' && i + 2 < output_labels.size() &&
                output_labels[i + 1] == '.' && output_labels[i + 2] == '.')
            {
                // 使用第一个输入张量的ellipsis维度作为参考
                size_t ellipsis_ndim = ellipsis_dims[0].empty() ? 0 : ellipsis_dims[0][0];
                for (size_t k = 0; k < ellipsis_ndim; ++k)
                {
                    char anonymous_label = 'A' + k;
                    expanded_output_labels += anonymous_label;
                    output_explicit_dims++;
                }
                i += 2; // 跳过后续两个点
            }
            else
            {
                expanded_output_labels += output_labels[i];
                output_explicit_dims++;
            }
        }

        // 验证输出标签中的匿名标签是否都在输入中出现过
        for (char label : expanded_output_labels)
        {
            if (label_to_size.find(label) == label_to_size.end())
            {
                throw std::invalid_argument("Output label '" + std::string(1, label) +
                                            "' not found in input labels");
            }
        }

        // 4. 确定输出张量的形状（使用展开后的标签）
        std::vector<size_t> output_shape;
        for (char label : expanded_output_labels)
        {
            auto it = label_to_size.find(label);
            if (it == label_to_size.end())
            {
                throw std::invalid_argument("Output label '" + std::string(1, label) +
                                            "' not found in input labels");
            }
            output_shape.push_back(it->second);
        }

        // 5. 创建输出张量
        Tensor<T> result(output_shape);

        // 6. 预计算每个输入张量的 strides（使用展开后的标签对应的维度）
        std::vector<std::vector<size_t>> input_strides;
        for (const auto &tensor : tensors)
        {
            input_strides.push_back(compute_strides(tensor->shape()));
        }

        // 7. 准备输出 strides
        std::vector<size_t> output_strides = compute_strides(output_shape);

        // 8. 创建所有标签的集合（使用展开后的标签）
        std::vector<char> all_labels;
        for (const auto &pair : label_to_size)
        {
            all_labels.push_back(pair.first);
        }

        // 9. 为每个标签创建维度限制
        std::vector<size_t> label_limits;
        for (char label : all_labels)
        {
            label_limits.push_back(label_to_size[label]);
        }

        // 10. 创建映射：标签 -> 位置索引
        std::unordered_map<char, size_t> label_to_index;
        for (size_t i = 0; i < all_labels.size(); ++i)
        {
            label_to_index[all_labels[i]] = i;
        }

        // 11. 创建映射：标签 -> 是否为求和维度（不在输出中）
        std::unordered_map<char, bool> is_sum_label;
        for (char label : all_labels)
        {
            is_sum_label[label] = (expanded_output_labels.find(label) == std::string::npos);
        }

        // 12. 使用展开后的输入标签进行迭代计算
        std::vector<size_t> indices(all_labels.size(), 0);
        do
        {
            // 计算输出索引
            std::vector<size_t> output_indices;
            for (char label : expanded_output_labels)
            {
                size_t idx = label_to_index[label];
                output_indices.push_back(indices[idx]);
            }

            // 计算输出张量中的位置
            size_t output_pos = 0;
            if (!output_indices.empty())
            {
                output_pos = compute_index(output_indices, output_shape, output_strides);
            }

            // 计算所有输入张量在当前位置的值
            T product = T(1);

            for (size_t i = 0; i < tensors.size(); ++i)
            {
                // 为当前张量构建索引（使用展开后的标签）
                std::vector<size_t> tensor_indices;
                for (char label : expanded_input_labels[i])
                {
                    size_t idx = label_to_index[label];
                    tensor_indices.push_back(indices[idx]);
                }

                // 获取该位置的值
                size_t tensor_pos = compute_index(tensor_indices,
                                                  tensors[i]->shape(),
                                                  input_strides[i]);
                product *= (*tensors[i])[tensor_pos];
            }

            // 累加到输出
            result[output_pos] += product;

        } while (increment_index(indices, label_limits));

        return opt<T>(result);
    }
} // namespace TensorN

#endif //!__EINSUM__H__