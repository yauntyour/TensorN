#pragma once
#ifndef __CUDA_TENSOR_HPP__
#define __CUDA_TENSOR_HPP__

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <memory>

namespace TensorN
{
    template <typename T>
    class CudaTensor
    {
    private:
        size_t _size = 0;
        std::vector<size_t> _shape;
        T* d_data = nullptr;  // Device pointer
        bool owns_memory = true;

        void allocate()
        {
            if (_size > 0)
            {
                cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_data), _size * sizeof(T));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
                }
            }
        }

        void deallocate()
        {
            if (d_data && owns_memory)
            {
                cudaFree(d_data);
                d_data = nullptr;
            }
        }

    public:
        CudaTensor() = default;
        
        CudaTensor(const std::vector<size_t>& shape) : _shape(shape)
        {
            _size = 1;
            for (auto& e : _shape)
                _size *= e;
            allocate();
        }

        CudaTensor(const std::vector<size_t>& shape, const T* host_data) : _shape(shape)
        {
            _size = 1;
            for (auto& e : _shape)
                _size *= e;
            allocate();
            copyFromHost(host_data, _size);
        }

        CudaTensor(const Tensor<T>& cpu_tensor) : _shape(cpu_tensor.shape())
        {
            _size = cpu_tensor.size();
            allocate();
            copyFromHost(cpu_tensor.data.data(), _size);
        }

        ~CudaTensor()
        {
            deallocate();
        }

        // Copy constructor
        CudaTensor(const CudaTensor& other) : _size(other._size), _shape(other._shape)
        {
            allocate();
            if (d_data && other.d_data)
            {
                cudaMemcpy(d_data, other.d_data, _size * sizeof(T), cudaMemcpyDeviceToDevice);
            }
        }

        // Move constructor
        CudaTensor(CudaTensor&& other) noexcept 
            : _size(other._size), _shape(std::move(other._shape)), 
              d_data(other.d_data), owns_memory(other.owns_memory)
        {
            other.d_data = nullptr;
            other._size = 0;
            other.owns_memory = false;
        }

        // Copy assignment
        CudaTensor& operator=(const CudaTensor& other)
        {
            if (this != &other)
            {
                deallocate();
                _size = other._size;
                _shape = other._shape;
                allocate();
                if (d_data && other.d_data)
                {
                    cudaMemcpy(d_data, other.d_data, _size * sizeof(T), cudaMemcpyDeviceToDevice);
                }
            }
            return *this;
        }

        // Move assignment
        CudaTensor& operator=(CudaTensor&& other) noexcept
        {
            if (this != &other)
            {
                deallocate();
                _size = other._size;
                _shape = std::move(other._shape);
                d_data = other.d_data;
                owns_memory = other.owns_memory;
                other.d_data = nullptr;
                other._size = 0;
                other.owns_memory = false;
            }
            return *this;
        }

        size_t size() const { return _size; }
        const std::vector<size_t>& shape() const { return _shape; }
        T* device_ptr() { return d_data; }
        const T* device_ptr() const { return d_data; }

        void copyFromHost(const T* host_data, size_t count)
        {
            if (count != _size)
            {
                throw std::invalid_argument("Data size mismatch");
            }
            if (d_data && host_data)
            {
                cudaError_t err = cudaMemcpy(d_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("CUDA memcpy H2D failed: " + std::string(cudaGetErrorString(err)));
                }
            }
        }

        void copyToHost(T* host_data, size_t count) const
        {
            if (count != _size)
            {
                throw std::invalid_argument("Data size mismatch");
            }
            if (host_data && d_data)
            {
                cudaError_t err = cudaMemcpy(host_data, d_data, count * sizeof(T), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("CUDA memcpy D2H failed: " + std::string(cudaGetErrorString(err)));
                }
            }
        }

        Tensor<T> toTensor() const
        {
            Tensor<T> result(_shape);
            copyToHost(result.data.data(), _size);
            return result;
        }

        static CudaTensor fromTensor(const Tensor<T>& cpu_tensor)
        {
            return CudaTensor(cpu_tensor);
        }
    };

    // Helper function to check CUDA errors
    inline void checkCudaError(cudaError_t err, const char* file, int line)
    {
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("CUDA error at ") + file + ":" + std::to_string(line) + 
                ": " + std::string(cudaGetErrorString(err)));
        }
    }

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

} // namespace TensorN

#endif // __CUDA_TENSOR_HPP__