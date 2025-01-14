//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <memory> // For std::allocator_traits
#include <new>    // For placement new

namespace einsums {

class MemoryAllocator {
  public:
    virtual void *allocate_raw(size_t bytes)              = 0;
    virtual void  deallocate_raw(void *ptr, size_t bytes) = 0;

    template <typename T>
    T *allocate(size_t count = 1) {
        void *memory       = allocate_raw(sizeof(T) * count);
        T    *typed_memory = static_cast<T *>(memory);

        // Use std::allocator_traits to construct objects
        for (size_t i = 0; i < count; ++i) {
            std::allocator_traits<std::allocator<T>>::construct(std::allocator<T>(), typed_memory + i);
        }
        return typed_memory;
    }

    template <typename T>
    void deallocate(T *ptr, size_t count = 1) {
        if (ptr != nullptr) {
            // Use std::allocator_traits to destroy objects
            for (size_t i = 0; i < count; ++i) {
                std::allocator_traits<std::allocator<T>>::destroy(std::allocator<T>(), ptr + i);
            }

            deallocate_raw(static_cast<void *>(ptr), sizeof(T) * count);
        }
    }

    virtual ~MemoryAllocator() = default;
};

} // namespace einsums
