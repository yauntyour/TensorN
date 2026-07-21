#pragma once
#ifndef __MEMORY_POOL_HPP__
#define __MEMORY_POOL_HPP__

#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <algorithm>
#include <cstddef>

namespace TensorN
{
    class MemoryPool
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
        size_t max_cached_bytes_ = 256 * 1024 * 1024;
        size_t cached_bytes_ = 0;

        static size_t bucket_key(size_t bytes)
        {
            size_t k = 256;
            while (k < bytes) k <<= 1;
            return k;
        }

    public:
        static MemoryPool& instance()
        {
            static MemoryPool pool;
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

            void* ptr = ::operator new(key);
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
            ::operator delete(ptr);
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
                        ::operator delete(block.ptr);
                        cached_bytes_ -= block.size;
                    }
                }
                blocks.erase(
                    std::remove_if(blocks.begin(), blocks.end(),
                        [](const Block& b) { return !b.in_use; }),
                    blocks.end());
            }
        }

        ~MemoryPool()
        {
            for (auto& [key, blocks] : buckets_)
                for (auto& block : blocks)
                    ::operator delete(block.ptr);
        }

        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;

    private:
        MemoryPool() = default;
    };

    template <typename T>
    class PooledAllocator
    {
    public:
        using value_type = T;

        PooledAllocator() noexcept = default;

        template <typename U>
        PooledAllocator(const PooledAllocator<U>&) noexcept {}

        T* allocate(size_t n)
        {
            return static_cast<T*>(MemoryPool::instance().acquire(n * sizeof(T)));
        }

        void deallocate(T* p, size_t) noexcept
        {
            MemoryPool::instance().release(p);
        }

        bool operator==(const PooledAllocator&) const noexcept { return true; }
        bool operator!=(const PooledAllocator&) const noexcept { return false; }
    };

    template <typename T>
    using PooledVector = std::vector<T, PooledAllocator<T>>;

} // namespace TensorN

#endif // __MEMORY_POOL_HPP__
