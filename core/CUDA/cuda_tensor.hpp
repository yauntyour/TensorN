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
        T* d_data = nullptr;
        bool owns_memory = true;
        bool is_pinned_ = false;

        void allocate()
        {
            if (_size > 0)
            {
                cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_data), _size * sizeof(T));
                if (err != cudaSuccess)
                {
                    TENSOR_THROW("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
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
            copyFromHost(cpu_tensor.data->data(), _size);
        }

        ~CudaTensor()
        {
            deallocate();
        }

        CudaTensor(const CudaTensor& other) : _size(other._size), _shape(other._shape)
        {
            allocate();
            if (d_data && other.d_data)
            {
                cudaMemcpy(d_data, other.d_data, _size * sizeof(T), cudaMemcpyDeviceToDevice);
            }
        }

        CudaTensor(CudaTensor&& other) noexcept 
            : _size(other._size), _shape(std::move(other._shape)), 
              d_data(other.d_data), owns_memory(other.owns_memory),
              is_pinned_(other.is_pinned_)
        {
            other.d_data = nullptr;
            other._size = 0;
            other.owns_memory = false;
            other.is_pinned_ = false;
        }

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

        CudaTensor& operator=(CudaTensor&& other) noexcept
        {
            if (this != &other)
            {
                deallocate();
                _size = other._size;
                _shape = std::move(other._shape);
                d_data = other.d_data;
                owns_memory = other.owns_memory;
                is_pinned_ = other.is_pinned_;
                other.d_data = nullptr;
                other._size = 0;
                other.owns_memory = false;
                other.is_pinned_ = false;
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
                TENSOR_THROW("Data size mismatch");
            }
            if (d_data && host_data)
            {
                cudaError_t err = cudaMemcpy(d_data, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    TENSOR_THROW("CUDA memcpy H2D failed: " + std::string(cudaGetErrorString(err)));
                }
            }
        }

        void copyToHost(T* host_data, size_t count) const
        {
            if (count != _size)
            {
                TENSOR_THROW("Data size mismatch");
            }
            if (host_data && d_data)
            {
                cudaError_t err = cudaMemcpy(host_data, d_data, count * sizeof(T), cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    TENSOR_THROW("CUDA memcpy D2H failed: " + std::string(cudaGetErrorString(err)));
                }
            }
        }

        void copyFromHostAsync(const T* host_data, size_t count, cudaStream_t stream)
        {
            if (count != _size) TENSOR_THROW("Data size mismatch");
            if (d_data && host_data)
            {
                cudaError_t err = cudaMemcpyAsync(d_data, host_data, count * sizeof(T),
                                                   cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess)
                    TENSOR_THROW("CUDA async H2D failed");
            }
        }

        void copyToHostAsync(T* host_data, size_t count, cudaStream_t stream) const
        {
            if (count != _size) TENSOR_THROW("Data size mismatch");
            if (host_data && d_data)
            {
                cudaError_t err = cudaMemcpyAsync(host_data, d_data, count * sizeof(T),
                                                   cudaMemcpyDeviceToHost, stream);
                if (err != cudaSuccess)
                    TENSOR_THROW("CUDA async D2H failed");
            }
        }

        void copyFromDeviceAsync(const T* src, size_t count, cudaStream_t stream)
        {
            if (count != _size) TENSOR_THROW("Data size mismatch");
            if (d_data && src)
            {
                cudaError_t err = cudaMemcpyAsync(d_data, src, count * sizeof(T),
                                                   cudaMemcpyDeviceToDevice, stream);
                if (err != cudaSuccess)
                    TENSOR_THROW("CUDA async D2D failed");
            }
        }

        Tensor<T> toTensor() const
        {
            Tensor<T> result(_shape);
            copyToHost(result.data->data(), _size);
            return result;
        }

        Tensor<T> toTensorAsync(cudaStream_t stream) const
        {
            Tensor<T> result(_shape);
            copyToHostAsync(result.data->data(), _size, stream);
            return result;
        }

        static CudaTensor fromTensor(const Tensor<T>& cpu_tensor)
        {
            return CudaTensor(cpu_tensor);
        }

        static CudaTensor fromTensorAsync(const Tensor<T>& cpu_tensor, cudaStream_t stream)
        {
            CudaTensor result(cpu_tensor.shape());
            result.copyFromHostAsync(cpu_tensor.data->data(), cpu_tensor.size(), stream);
            return result;
        }

        static CudaTensor fromPinned(const Tensor<T>& cpu_tensor, cudaStream_t stream)
        {
            CudaTensor result;
            result._shape = cpu_tensor.shape();
            result._size = cpu_tensor.size();
            result.allocate();

            void* pinned = nullptr;
            cudaMallocHost(&pinned, result._size * sizeof(T));
            std::memcpy(pinned, cpu_tensor.data->data(), result._size * sizeof(T));
            cudaMemcpyAsync(result.d_data, pinned, result._size * sizeof(T),
                           cudaMemcpyHostToDevice, stream);
            cudaFreeHost(pinned);
            return result;
        }

        CudaTensor<T> view() const
        {
            CudaTensor<T> result;
            result._size = _size;
            result._shape = _shape;
            result.d_data = d_data;
            result.owns_memory = false;
            return result;
        }

        CudaTensor<T> reshape(const std::vector<size_t>& new_shape) const
        {
            size_t new_size = 1;
            for (auto& e : new_shape) new_size *= e;
            if (new_size != _size)
                TENSOR_THROW("Reshape: total size must match");
            CudaTensor<T> result = view();
            result._shape = new_shape;
            return result;
        }

        void memset_zero()
        {
            if (d_data && _size > 0)
                cudaMemset(d_data, 0, _size * sizeof(T));
        }

        void memset_zero_async(cudaStream_t stream)
        {
            if (d_data && _size > 0)
                cudaMemsetAsync(d_data, 0, _size * sizeof(T), stream);
        }
    };

    inline void checkCudaError(cudaError_t err, const char* file, int line)
    {
        if (err != cudaSuccess)
        {
            throw TensorException(
                std::string("CUDA error: ") + std::string(cudaGetErrorString(err)),
                file, "TensorN::cuda::checkCudaError", line);
        }
    }

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

} // namespace TensorN

#endif // __CUDA_TENSOR_HPP__