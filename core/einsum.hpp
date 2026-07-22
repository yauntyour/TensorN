#pragma once
#ifndef __EINSUM__H__
#define __EINSUM__H__

#include "tensor.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

namespace TensorN
{
    namespace einsum_tools
    {
        const std::string ANON_PREFIX("\x80\x80");

        inline std::string make_anon_label(size_t index)
        {
            return ANON_PREFIX + std::to_string(index);
        }

        inline bool is_anon_label(const std::string &label)
        {
            return label.size() >= 2 &&
                   label[0] == '\x80' &&
                   label[1] == '\x80';
        }

        inline size_t compute_index(const std::vector<size_t> &indices,
                                    const std::vector<size_t> &strides)
        {
            if (indices.empty())
                return 0;
            size_t idx = 0;
            for (size_t i = 0; i < indices.size(); ++i)
            {
                idx += indices[i] * strides[i];
            }
            return idx;
        }

        inline std::vector<size_t> compute_strides(const std::vector<size_t> &shape)
        {
            if (shape.empty())
                return {};
            std::vector<size_t> strides(shape.size(), 1);
            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            return strides;
        }

        inline bool increment_index(std::vector<size_t> &indices,
                             const std::vector<size_t> &limits)
        {
            for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i)
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

        struct EinsumExpr
        {
            std::vector<std::string> input_labels;
            std::string output_labels;
            bool has_arrow;
        };

        inline EinsumExpr parse_expression(const std::string &exp)
        {
            EinsumExpr result;
            result.has_arrow = false;

            std::string cleaned;
            for (char c : exp)
            {
                if (!isspace(static_cast<unsigned char>(c)))
                    cleaned += c;
            }

            if (cleaned.empty())
                TENSOR_THROW("Empty einsum expression");

            size_t arrow_pos = cleaned.find("->");

            std::string input_part;
            std::string output_part;

            if (arrow_pos != std::string::npos)
            {
                result.has_arrow = true;
                input_part = cleaned.substr(0, arrow_pos);
                output_part = cleaned.substr(arrow_pos + 2);
            }
            else
            {
                input_part = cleaned;
            }

            size_t start = 0;
            size_t comma_pos;
            while ((comma_pos = input_part.find(',', start)) != std::string::npos)
            {
                result.input_labels.push_back(input_part.substr(start, comma_pos - start));
                start = comma_pos + 1;
            }
            result.input_labels.push_back(input_part.substr(start));

            if (result.input_labels.empty())
                TENSOR_THROW("No input tensors specified");

            for (const auto &input : result.input_labels)
            {
                size_t count = 0;
                for (size_t i = 0; i < input.size(); ++i)
                {
                    if (input[i] == '.' && i + 2 < input.size() &&
                        input[i + 1] == '.' && input[i + 2] == '.')
                    {
                        count++;
                        i += 2;
                    }
                }
                if (count > 1)
                    TENSOR_THROW("Multiple ellipsis in single input tensor");
            }

            if (result.has_arrow)
            {
                size_t count = 0;
                for (size_t i = 0; i < output_part.size(); ++i)
                {
                    if (output_part[i] == '.' && i + 2 < output_part.size() &&
                        output_part[i + 1] == '.' && output_part[i + 2] == '.')
                    {
                        count++;
                        i += 2;
                    }
                }
                if (count > 1)
                    TENSOR_THROW("Multiple ellipsis in output");

                result.output_labels = output_part;
            }
            else
            {
                std::unordered_map<char, size_t> char_count;
                bool has_ellipsis = false;

                for (const auto &input : result.input_labels)
                {
                    for (size_t i = 0; i < input.size(); ++i)
                    {
                        if (input[i] == '.' && i + 2 < input.size() &&
                            input[i + 1] == '.' && input[i + 2] == '.')
                        {
                            has_ellipsis = true;
                            i += 2;
                        }
                        else
                        {
                            char_count[input[i]]++;
                        }
                    }
                }

                if (has_ellipsis)
                    result.output_labels = "...";

                std::string odd_labels;
                for (const auto &pair : char_count)
                {
                    if (pair.second % 2 != 0)
                        odd_labels += pair.first;
                }
                std::sort(odd_labels.begin(), odd_labels.end());
                result.output_labels += odd_labels;
            }

            return result;
        }

        struct IndexResult
        {
            std::unordered_map<std::string, size_t> label_to_size;
            std::vector<std::string> all_labels;
            std::vector<std::string> expanded_input_labels;
            std::string expanded_output_labels;
            std::vector<size_t> ellipsis_ndims;
        };

        template <typename T>
        IndexResult build_index_mapping(
            const EinsumExpr &expr,
            const std::vector<const Tensor<T> *> &tensors)
        {
            IndexResult result;
            result.ellipsis_ndims.resize(expr.input_labels.size());

            std::vector<std::string> ellipsis_anon_labels;
            bool ellipsis_labels_initialized = false;
            size_t anon_counter = 0;

            auto register_label = [&](const std::string &label, size_t dim_size)
            {
                if (result.label_to_size.count(label))
                {
                    if (result.label_to_size[label] != dim_size)
                    {
                        TENSOR_THROW(
                            "Inconsistent dimension size for label '" + label +
                            "': expected " + std::to_string(result.label_to_size[label]) +
                            ", got " + std::to_string(dim_size));
                    }
                }
                else
                {
                    result.label_to_size[label] = dim_size;
                    result.all_labels.push_back(label);
                }
            };

            for (size_t i = 0; i < expr.input_labels.size(); ++i)
            {
                const std::string &labels = expr.input_labels[i];
                const auto &shape = tensors[i]->shape();

                size_t explicit_count = 0;
                for (size_t j = 0; j < labels.size(); ++j)
                {
                    if (labels[j] == '.' && j + 2 < labels.size() &&
                        labels[j + 1] == '.' && labels[j + 2] == '.')
                    {
                        j += 2;
                    }
                    else
                    {
                        explicit_count++;
                    }
                }

                if (explicit_count > shape.size())
                {
                    TENSOR_THROW(
                        "Number of explicit labels (" + std::to_string(explicit_count) +
                        ") exceeds tensor " + std::to_string(i) +
                        " dimension (" + std::to_string(shape.size()) + ")");
                }

                size_t ndim = shape.size() - explicit_count;
                result.ellipsis_ndims[i] = ndim;

                std::string expanded;

                if (ndim > 0)
                {
                    if (!ellipsis_labels_initialized)
                    {
                        for (size_t k = 0; k < ndim; ++k)
                        {
                            ellipsis_anon_labels.push_back(make_anon_label(anon_counter++));
                        }
                        ellipsis_labels_initialized = true;
                    }
                    else if (ellipsis_anon_labels.size() != ndim)
                    {
                        TENSOR_THROW(
                            "Inconsistent ellipsis dimensions across input tensors");
                    }

                    for (size_t k = 0; k < ndim; ++k)
                    {
                        expanded += ellipsis_anon_labels[k];
                        register_label(ellipsis_anon_labels[k], shape[k]);
                    }
                }

                size_t dim_idx = ndim;
                for (size_t j = 0; j < labels.size(); ++j)
                {
                    if (labels[j] == '.' && j + 2 < labels.size() &&
                        labels[j + 1] == '.' && labels[j + 2] == '.')
                    {
                        j += 2;
                        continue;
                    }

                    std::string label(1, labels[j]);
                    expanded += label;

                    if (dim_idx >= shape.size())
                    {
                        TENSOR_THROW(
                            "Label index out of bounds for tensor " + std::to_string(i));
                    }

                    register_label(label, shape[dim_idx]);
                    dim_idx++;
                }

                result.expanded_input_labels.push_back(expanded);
            }

            // Process output labels
            std::string expanded_output;
            for (size_t j = 0; j < expr.output_labels.size(); ++j)
            {
                if (expr.output_labels[j] == '.' && j + 2 < expr.output_labels.size() &&
                    expr.output_labels[j + 1] == '.' && expr.output_labels[j + 2] == '.')
                {
                    for (const auto &anon : ellipsis_anon_labels)
                    {
                        expanded_output += anon;
                    }
                    j += 2;
                }
                else
                {
                    expanded_output += std::string(1, expr.output_labels[j]);
                }
            }

            // Validate output labels
            for (size_t j = 0; j < expanded_output.size(); ++j)
            {
                // Extract full label (could be anonymous)
                std::string label;
                if (expanded_output[j] == '\x80')
                {
                    // Anonymous label: extract until next non-anon char
                    size_t k = j;
                    while (k < expanded_output.size() && (expanded_output[k] == '\x80' || expanded_output[k] == '\x81'))
                        k++;
                    // Actually, let's extract the full anon label properly
                    // Anonymous labels are: \x80\x80 + digits
                    label = expanded_output.substr(j, 2);
                    j += 1; // Will be incremented by loop
                    // Collect digits
                    while (j + 1 < expanded_output.size() && expanded_output[j + 1] >= '0' && expanded_output[j + 1] <= '9')
                    {
                        label += expanded_output[j + 1];
                        j++;
                    }
                }
                else
                {
                    label = std::string(1, expanded_output[j]);
                }

                if (result.label_to_size.find(label) == result.label_to_size.end())
                {
                    TENSOR_THROW(
                        "Output label not found in input labels");
                }
            }

            result.expanded_output_labels = expanded_output;

            // Check for duplicate output labels
            {
                std::unordered_set<std::string> output_set;
                for (size_t j = 0; j < expanded_output.size(); ++j)
                {
                    std::string label;
                    if (expanded_output[j] == '\x80')
                    {
                        label = expanded_output.substr(j, 2);
                        j += 1;
                        while (j + 1 < expanded_output.size() && expanded_output[j + 1] >= '0' && expanded_output[j + 1] <= '9')
                        {
                            label += expanded_output[j + 1];
                            j++;
                        }
                    }
                    else
                    {
                        label = std::string(1, expanded_output[j]);
                    }

                    if (output_set.count(label))
                    {
                        TENSOR_THROW(
                            "Duplicate label in output");
                    }
                    output_set.insert(label);
                }
            }

            return result;
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

        EinsumExpr expr = parse_expression(exp);

        if (tensors.empty())
            TENSOR_THROW("No input tensors provided");

        if (expr.input_labels.size() != tensors.size())
        {
            TENSOR_THROW(
                "Expected " + std::to_string(expr.input_labels.size()) +
                " input tensors, got " + std::to_string(tensors.size()));
        }

        IndexResult idx = build_index_mapping(expr, tensors);

        // Build output shape from expanded_output_labels
        std::vector<size_t> output_shape;
        {
            const std::string &out = idx.expanded_output_labels;
            for (size_t j = 0; j < out.size(); ++j)
            {
                std::string label;
                if (out[j] == '\x80')
                {
                    label = out.substr(j, 2);
                    j += 1;
                    while (j + 1 < out.size() && out[j + 1] >= '0' && out[j + 1] <= '9')
                    {
                        label += out[j + 1];
                        j++;
                    }
                }
                else
                {
                    label = std::string(1, out[j]);
                }

                auto it = idx.label_to_size.find(label);
                if (it == idx.label_to_size.end())
                {
                    TENSOR_THROW("Output label not found in label map");
                }
                output_shape.push_back(it->second);
            }
        }

        Tensor<T> result(output_shape);

        // Compute strides
        std::vector<std::vector<size_t>> input_strides;
        for (const auto &tensor : tensors)
        {
            input_strides.push_back(compute_strides(tensor->shape()));
        }
        std::vector<size_t> output_strides = compute_strides(output_shape);

        // Build iteration limits and label->iteration-index mapping
        std::vector<size_t> label_limits;
        for (const auto &label : idx.all_labels)
        {
            label_limits.push_back(idx.label_to_size[label]);
        }

        std::unordered_map<std::string, size_t> label_to_iter_idx;
        for (size_t i = 0; i < idx.all_labels.size(); ++i)
        {
            label_to_iter_idx[idx.all_labels[i]] = i;
        }

        // Helper: parse expanded label string into individual labels
        auto parse_expanded = [](const std::string &s) -> std::vector<std::string>
        {
            std::vector<std::string> labels;
            for (size_t i = 0; i < s.size(); ++i)
            {
                if (s[i] == '\x80')
                {
                    std::string label = s.substr(i, 2);
                    i += 1;
                    while (i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9')
                    {
                        label += s[i + 1];
                        i++;
                    }
                    labels.push_back(label);
                }
                else
                {
                    labels.push_back(std::string(1, s[i]));
                }
            }
            return labels;
        };

        // Parse all label strings once
        std::vector<std::vector<std::string>> input_label_lists(idx.expanded_input_labels.size());
        for (size_t i = 0; i < idx.expanded_input_labels.size(); ++i)
        {
            input_label_lists[i] = parse_expanded(idx.expanded_input_labels[i]);
        }
        std::vector<std::string> output_label_list = parse_expanded(idx.expanded_output_labels);

        // Pre-compute iteration index lookups for each input and output
        std::vector<std::vector<size_t>> input_iter_indices(idx.expanded_input_labels.size());
        for (size_t i = 0; i < input_label_lists.size(); ++i)
        {
            input_iter_indices[i].resize(input_label_lists[i].size());
            for (size_t j = 0; j < input_label_lists[i].size(); ++j)
            {
                input_iter_indices[i][j] = label_to_iter_idx[input_label_lists[i][j]];
            }
        }
        std::vector<size_t> output_iter_indices(output_label_list.size());
        for (size_t j = 0; j < output_label_list.size(); ++j)
        {
            output_iter_indices[j] = label_to_iter_idx[output_label_list[j]];
        }

        // Main contraction loop
        std::vector<size_t> iter_indices(idx.all_labels.size(), 0);

        do
        {
            // Compute output position
            std::vector<size_t> output_indices(output_iter_indices.size());
            for (size_t j = 0; j < output_iter_indices.size(); ++j)
            {
                output_indices[j] = iter_indices[output_iter_indices[j]];
            }
            size_t output_pos = compute_index(output_indices, output_strides);

            // Compute product of all input values
            T product = T(1);
            for (size_t i = 0; i < tensors.size(); ++i)
            {
                std::vector<size_t> tensor_indices(input_iter_indices[i].size());
                for (size_t j = 0; j < input_iter_indices[i].size(); ++j)
                {
                    tensor_indices[j] = iter_indices[input_iter_indices[i][j]];
                }
                size_t tensor_pos = compute_index(tensor_indices, input_strides[i]);
                product *= (*tensors[i])[tensor_pos];
            }

            (*result.data)[output_pos] += product;

        } while (increment_index(iter_indices, label_limits));

        return opt<T>(result);
    }
} // namespace TensorN

#endif //!__EINSUM__H__
