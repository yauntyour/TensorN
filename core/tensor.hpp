#pragma once
#ifndef __DATA__H__
#define __DATA__H__

#include <vector>
#include <stdint.h>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace TensorN
{
    template <typename T>
    class opt;

    template <typename T>
    class Tensor
    {
    private:
        size_t _size = 1;
        std::vector<size_t> _shape;

        void format_recursive(std::ostream &os, const std::vector<T> &data,
                              const std::vector<size_t> &shape,
                              std::vector<size_t> &indices, size_t dim,
                              size_t offset, int indent = 0) const
        {
            if (dim == shape.size())
            {
                os << data[offset];
                return;
            }

            if (dim == shape.size() - 1)
            {
                os << "[";
                for (size_t i = 0; i < shape[dim]; ++i)
                {
                    indices[dim] = i;
                    size_t new_offset = offset + i;
                    format_recursive(os, data, shape, indices, dim + 1, new_offset, indent);
                    if (i < shape[dim] - 1)
                        os << ", ";
                }
                os << "]";
            }
            else
            {
                os << "[";
                if (shape[dim] > 1 && dim < shape.size() - 2)
                    os << "\n"
                       << std::string(indent + 2, ' ');

                size_t stride = std::accumulate(shape.begin() + dim + 1,
                                                shape.end(), 1,
                                                std::multiplies<size_t>());

                for (size_t i = 0; i < shape[dim]; ++i)
                {
                    indices[dim] = i;
                    size_t new_offset = offset + i * stride;

                    format_recursive(os, data, shape, indices, dim + 1, new_offset, indent + 2);

                    if (i < shape[dim] - 1)
                    {
                        os << ",";
                        if (dim < shape.size() - 2)
                            os << "\n"
                               << std::string(indent + 2, ' ');
                        else
                            os << " ";
                    }
                }

                if (shape[dim] > 1 && dim < shape.size() - 2)
                    os << "\n"
                       << std::string(indent, ' ');
                os << "]";
            }
        }

    public:
        std::vector<T> data;

        Tensor() = default;
        Tensor(const Tensor<T> &other) : _size(other._size), _shape(other._shape), data(other.data) {}
        Tensor(Tensor<T> &&other) noexcept : _size(other._size), _shape(std::move(other._shape)), data(std::move(other.data))
        {
            other._size = 0;
        }
        Tensor(const std::vector<size_t> &shape) : _shape(shape)
        {
            for (auto &e : _shape)
            {
                _size *= e;
            }
            data.resize(_size);
        }
        Tensor(const std::vector<size_t> &shape, const std::vector<T> &data_vec) : _shape(shape), data(data_vec)
        {
            _size = 1;
            for (auto &e : _shape)
                _size *= e;
            if (_size != data.size())
            {
                throw std::invalid_argument("Shape does not match data size.");
            }
        }

        ~Tensor() = default;

        size_t size() const
        {
            return _size;
        }
        // Iterators for STL compatibility
        typename std::vector<T>::iterator begin() { return data.begin(); }
        typename std::vector<T>::iterator end() { return data.end(); }

        typename std::vector<T>::const_iterator begin() const { return data.begin(); }
        typename std::vector<T>::const_iterator end() const { return data.end(); }
        typename std::vector<T>::const_iterator cbegin() const { return data.cbegin(); }
        typename std::vector<T>::const_iterator cend() const { return data.cend(); }

        const std::vector<size_t> &shape() const
        {
            return _shape;
        }

        bool is_isomorphic(const Tensor<T> &B) const
        {
            return B._shape == _shape;
        }

        bool operator==(const Tensor<T> &B) const
        {
            if (!is_isomorphic(B))
                return false;
            return data == B.data;
        }

        bool check_indices(const std::vector<size_t> &indices) const
        {
            for (size_t i = 0; i < indices.size(); ++i)
            {
                if (indices[i] >= _shape[i])
                {
                    return false;
                }
            }
            return true;
        }

        size_t flat_index(const std::vector<size_t> &indices) const
        {
            size_t idx = 0;
            size_t stride = 1;
            for (int i = _shape.size() - 1; i >= 0; --i)
            {
                idx += indices[i] * stride;
                stride *= _shape[i];
            }
            return idx;
        }

        Tensor<T> &operator+=(const Tensor<T> &B)
        {
            if (!is_isomorphic(B))
            {
                throw std::invalid_argument("A and B are not isomorphic Tensors.");
            }
            for (size_t i = 0; i < _size; i++)
            {
                data[i] += B.data[i];
            }
            return *this;
        }
        Tensor<T> &operator-=(const Tensor<T> &B)
        {
            if (!is_isomorphic(B))
            {
                throw std::invalid_argument("A and B are not isomorphic Tensors.");
            }
            for (size_t i = 0; i < _size; i++)
            {
                data[i] -= B.data[i];
            }
            return *this;
        }
        Tensor<T> &operator*=(const Tensor<T> &B)
        {
            if (!is_isomorphic(B))
            {
                throw std::invalid_argument("A and B are not isomorphic Tensors.");
            }
            for (size_t i = 0; i < _size; i++)
            {
                data[i] *= B.data[i];
            }
            return *this;
        }
        Tensor<T> &operator/=(const Tensor<T> &B)
        {
            if (!is_isomorphic(B))
            {
                throw std::invalid_argument("A and B are not isomorphic Tensors.");
            }
            for (size_t i = 0; i < _size; i++)
            {
                data[i] /= B.data[i];
            }
            return *this;
        }

        opt<T> operator+(const Tensor<T> &B) const
        {
            opt<T> oper(*this);
            oper.tensor += B;
            return oper;
        }
        opt<T> operator-(const Tensor<T> &B) const
        {
            opt<T> oper(*this);
            oper.tensor -= B;
            return oper;
        }
        opt<T> operator*(const Tensor<T> &B) const
        {
            opt<T> oper(*this);
            oper.tensor *= B;
            return oper;
        }

        Tensor<T> &operator+=(T B)
        {
            for (size_t i = 0; i < _size; i++)
            {
                data[i] += B;
            }
            return *this;
        }
        Tensor<T> &operator-=(T B)
        {
            for (size_t i = 0; i < _size; i++)
            {
                data[i] -= B;
            }
            return *this;
        }
        Tensor<T> &operator*=(T B)
        {
            for (size_t i = 0; i < _size; i++)
            {
                data[i] *= B;
            }
            return *this;
        }
        Tensor<T> &operator/=(T B)
        {
            for (size_t i = 0; i < _size; i++)
            {
                data[i] /= B;
            }
            return *this;
        }

        opt<T> operator+(T B) const
        {
            opt<T> oper(*this);
            oper.tensor += B;
            return oper;
        }
        opt<T> operator-(T B) const
        {
            opt<T> oper(*this);
            oper.tensor -= B;
            return oper;
        }
        opt<T> operator*(T B) const
        {
            opt<T> oper(*this);
            oper.tensor *= B;
            return oper;
        }
        opt<T> operator/(T B) const
        {
            opt<T> oper(*this);
            oper.tensor /= B;
            return oper;
        }

        T &operator[](const std::vector<size_t> &indices)
        {
            if (indices.size() != _shape.size())
            {
                throw std::invalid_argument("Number of indices must match tensor dimension.");
            }
            if (!check_indices(indices))
            {
                throw std::out_of_range("Index out of range.");
            }
            return data[flat_index(indices)];
        }
        T &operator[](size_t index)
        {
            return data[index];
        }
        const T &operator[](const std::vector<size_t> &indices) const
        {
            if (indices.size() != _shape.size())
            {
                throw std::invalid_argument("Number of indices must match tensor dimension.");
            }
            if (!check_indices(indices))
            {
                throw std::out_of_range("Index out of range.");
            }
            return data[flat_index(indices)];
        }
        const T &operator[](size_t index) const
        {
            return data[index];
        }
        friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor)
        {
            if (tensor._size == 0)
            {
                os << "Tensor[]";
            }
            else if (tensor._shape.empty())
            {
                os << tensor.data[0];
            }
            else
            {
                std::vector<size_t> indices(tensor._shape.size(), 0);
                tensor.format_recursive(os, tensor.data, tensor._shape, indices, 0, 0);
            }
            return os;
        }
    };

    template <typename T>
    class opt
    {
    public:
        Tensor<T> tensor;

        opt(const Tensor<T> &tensor) : tensor(tensor)
        {
        }

        opt(std::vector<size_t> shape)
        {
            tensor = Tensor<T>(shape);
        }

        opt<T> &operator+(const Tensor<T> &B)
        {
            tensor += B;
            return *this;
        }
        opt<T> &operator-(const Tensor<T> &B)
        {
            tensor -= B;
            return *this;
        }
        opt<T> &operator*(const Tensor<T> &B)
        {
            tensor *= B;
            return *this;
        }

        opt<T> &operator+(T B)
        {
            tensor += B;
            return *this;
        }
        opt<T> &operator-(T B)
        {
            tensor -= B;
            return *this;
        }
        opt<T> &operator*(T B)
        {
            tensor *= B;
            return *this;
        }
        opt<T> &operator/(T B)
        {
            tensor /= B;
            return *this;
        }

        operator Tensor<T>() const
        {
            return tensor;
        }

        friend std::ostream &operator<<(std::ostream &os, const opt<T> &opt_tensor)
        {
            os << opt_tensor.tensor;
            return os;
        }
    };
    template <typename T>
    Tensor<T> zeros(const std::vector<size_t> &shape)
    {
        Tensor<T> tensor(shape);
        std::fill(tensor.data.begin(), tensor.data.end(), T(0));
        return tensor;
    }

    template <typename T>
    Tensor<T> ones(const std::vector<size_t> &shape)
    {
        Tensor<T> tensor(shape);
        std::fill(tensor.data.begin(), tensor.data.end(), T(1));
        return tensor;
    }

    template <typename T>
    Tensor<T> eye(size_t n)
    {
        Tensor<T> tensor({n, n});
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                tensor[{i, j}] = (i == j) ? T(1) : T(0);
            }
        }
        return tensor;
    }

    template <typename T>
    Tensor<T> arange(T start, T stop, T step = 1)
    {
        size_t n = static_cast<size_t>((stop - start) / step);
        std::vector<T> values;
        for (T val = start; val < stop; val += step)
        {
            values.push_back(val);
        }
        return Tensor<T>({n}, values);
    }
}

#endif // !__DATA__H__