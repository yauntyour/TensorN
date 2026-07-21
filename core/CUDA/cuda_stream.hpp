#pragma once
#ifndef __CUDA_STREAM_HPP__
#define __CUDA_STREAM_HPP__

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace TensorN
{
    class CudaStream
    {
    private:
        cudaStream_t stream_ = nullptr;
        bool owns_ = true;

    public:
        CudaStream()
        {
            cudaError_t err = cudaStreamCreate(&stream_);
            if (err != cudaSuccess)
                throw std::runtime_error("Failed to create CUDA stream");
        }

        explicit CudaStream(cudaStream_t external) : stream_(external), owns_(false) {}

        ~CudaStream()
        {
            if (owns_ && stream_)
                cudaStreamDestroy(stream_);
        }

        CudaStream(CudaStream&& o) noexcept : stream_(o.stream_), owns_(o.owns_)
        {
            o.stream_ = nullptr;
            o.owns_ = false;
        }

        CudaStream& operator=(CudaStream&& o) noexcept
        {
            if (this != &o)
            {
                if (owns_ && stream_) cudaStreamDestroy(stream_);
                stream_ = o.stream_;
                owns_ = o.owns_;
                o.stream_ = nullptr;
                o.owns_ = false;
            }
            return *this;
        }

        CudaStream(const CudaStream&) = delete;
        CudaStream& operator=(const CudaStream&) = delete;

        cudaStream_t get() const { return stream_; }
        operator cudaStream_t() const { return stream_; }

        void synchronize() const
        {
            cudaStreamSynchronize(stream_);
        }

        void wait_event(cudaEvent_t event) const
        {
            cudaStreamWaitEvent(stream_, event, 0);
        }
    };

    class CudaEvent
    {
    private:
        cudaEvent_t event_ = nullptr;

    public:
        CudaEvent()
        {
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
        }

        ~CudaEvent()
        {
            if (event_) cudaEventDestroy(event_);
        }

        CudaEvent(CudaEvent&& o) noexcept : event_(o.event_) { o.event_ = nullptr; }
        CudaEvent& operator=(CudaEvent&& o) noexcept
        {
            if (this != &o)
            {
                if (event_) cudaEventDestroy(event_);
                event_ = o.event_;
                o.event_ = nullptr;
            }
            return *this;
        }

        CudaEvent(const CudaEvent&) = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;

        cudaEvent_t get() const { return event_; }

        void record(cudaStream_t stream = nullptr)
        {
            cudaEventRecord(event_, stream);
        }

        void synchronize()
        {
            cudaEventSynchronize(event_);
        }
    };

    class CudaStreamPool
    {
    private:
        std::vector<std::unique_ptr<CudaStream>> streams_;
        size_t next_ = 0;
        std::mutex mutex_;

    public:
        static CudaStreamPool& instance(size_t count = 4)
        {
            static CudaStreamPool pool(count);
            return pool;
        }

        explicit CudaStreamPool(size_t count)
        {
            for (size_t i = 0; i < count; ++i)
                streams_.push_back(std::make_unique<CudaStream>());
        }

        cudaStream_t acquire()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& s = streams_[next_ % streams_.size()];
            next_++;
            return s->get();
        }

        void synchronize_all()
        {
            for (auto& s : streams_)
                s->synchronize();
        }
    };

    class CudaMemoryPool
    {
    private:
        struct Block
        {
            void* ptr;
            size_t size;
            bool in_use;
        };

        std::unordered_map<size_t, std::vector<Block>> buckets_;
        std::mutex mutex_;
        size_t cached_bytes_ = 0;
        size_t max_cached_bytes_ = 512ULL * 1024 * 1024;

        static size_t bucket_key(size_t bytes)
        {
            size_t k = 512;
            while (k < bytes) k <<= 1;
            return k;
        }

    public:
        static CudaMemoryPool& instance()
        {
            static CudaMemoryPool pool;
            return pool;
        }

        void* acquire(size_t bytes)
        {
            if (bytes == 0) return nullptr;
            size_t key = bucket_key(bytes);
            std::lock_guard<std::mutex> lock(mutex_);

            auto it = buckets_.find(key);
            if (it != buckets_.end())
            {
                for (auto& block : it->second)
                {
                    if (!block.in_use)
                    {
                        block.in_use = true;
                        return block.ptr;
                    }
                }
            }

            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, key);
            if (err != cudaSuccess)
                throw std::runtime_error("CudaMemoryPool: cudaMalloc failed");

            buckets_[key].push_back({ptr, key, true});
            cached_bytes_ += key;
            return ptr;
        }

        void release(void* ptr)
        {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [key, blocks] : buckets_)
            {
                for (auto& block : blocks)
                {
                    if (block.ptr == ptr)
                    {
                        block.in_use = false;
                        return;
                    }
                }
            }
            cudaFree(ptr);
        }

        void purge()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [key, blocks] : buckets_)
            {
                for (auto& block : blocks)
                {
                    if (!block.in_use)
                    {
                        cudaFree(block.ptr);
                        cached_bytes_ -= block.size;
                    }
                }
                blocks.erase(
                    std::remove_if(blocks.begin(), blocks.end(),
                        [](const Block& b) { return !b.in_use; }),
                    blocks.end());
            }
        }

        ~CudaMemoryPool()
        {
            for (auto& [key, blocks] : buckets_)
                for (auto& block : blocks)
                    cudaFree(block.ptr);
        }

        CudaMemoryPool(const CudaMemoryPool&) = delete;
        CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    private:
        CudaMemoryPool() = default;
    };

    class PinnedMemoryPool
    {
    private:
        struct Block
        {
            void* ptr;
            size_t size;
            bool in_use;
        };

        std::unordered_map<size_t, std::vector<Block>> buckets_;
        std::mutex mutex_;

        static size_t bucket_key(size_t bytes)
        {
            size_t k = 4096;
            while (k < bytes) k <<= 1;
            return k;
        }

    public:
        static PinnedMemoryPool& instance()
        {
            static PinnedMemoryPool pool;
            return pool;
        }

        void* acquire(size_t bytes)
        {
            if (bytes == 0) return nullptr;
            size_t key = bucket_key(bytes);
            std::lock_guard<std::mutex> lock(mutex_);

            auto it = buckets_.find(key);
            if (it != buckets_.end())
            {
                for (auto& block : it->second)
                {
                    if (!block.in_use)
                    {
                        block.in_use = true;
                        return block.ptr;
                    }
                }
            }

            void* ptr = nullptr;
            cudaError_t err = cudaMallocHost(&ptr, key);
            if (err != cudaSuccess)
                throw std::runtime_error("PinnedMemoryPool: cudaMallocHost failed");

            buckets_[key].push_back({ptr, key, true});
            return ptr;
        }

        void release(void* ptr)
        {
            if (!ptr) return;
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [key, blocks] : buckets_)
            {
                for (auto& block : blocks)
                {
                    if (block.ptr == ptr)
                    {
                        block.in_use = false;
                        return;
                    }
                }
            }
            cudaFreeHost(ptr);
        }

        void purge()
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& [key, blocks] : buckets_)
            {
                for (auto& block : blocks)
                {
                    if (!block.in_use)
                        cudaFreeHost(block.ptr);
                }
                blocks.erase(
                    std::remove_if(blocks.begin(), blocks.end(),
                        [](const Block& b) { return !b.in_use; }),
                    blocks.end());
            }
        }

        ~PinnedMemoryPool()
        {
            for (auto& [key, blocks] : buckets_)
                for (auto& block : blocks)
                    cudaFreeHost(block.ptr);
        }

        PinnedMemoryPool(const PinnedMemoryPool&) = delete;
        PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

    private:
        PinnedMemoryPool() = default;
    };

    inline void async_copy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t stream)
    {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    }

    inline void async_copy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t stream)
    {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    }

    inline void async_copy_d2d(void* dst, const void* src, size_t bytes, cudaStream_t stream)
    {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    }

    class CudaBlasStreamHandle
    {
    private:
        cublasHandle_t handle_ = nullptr;
        cudaStream_t bound_stream_ = nullptr;

    public:
        CudaBlasStreamHandle()
        {
            cublasCreate(&handle_);
        }

        ~CudaBlasStreamHandle()
        {
            if (handle_) cublasDestroy(handle_);
        }

        CudaBlasStreamHandle(CudaBlasStreamHandle&& o) noexcept
            : handle_(o.handle_), bound_stream_(o.bound_stream_)
        {
            o.handle_ = nullptr;
        }

        CudaBlasStreamHandle(const CudaBlasStreamHandle&) = delete;
        CudaBlasStreamHandle& operator=(const CudaBlasStreamHandle&) = delete;

        cublasHandle_t get() { return handle_; }

        void set_stream(cudaStream_t stream)
        {
            if (stream != bound_stream_)
            {
                cublasSetStream(handle_, stream);
                bound_stream_ = stream;
            }
        }
    };

    inline CudaBlasStreamHandle& get_stream_blas_handle()
    {
        thread_local CudaBlasStreamHandle h;
        return h;
    }

} // namespace TensorN

#endif // __CUDA_STREAM_HPP__
