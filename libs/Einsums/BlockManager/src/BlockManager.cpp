//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BlockManager/BlockManager.hpp>

#include <memory>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUMemory/GPUAllocator.hpp>
#    include <Einsums/GPUMemory/GPUPointer.hpp>
#endif

namespace einsums {

#ifdef EINSUMS_COMPUTE_CODE
GPUBlock::GPUBlock(gpu::GPUAllocator<uint8_t> &alloc, size_t bytes) : size{bytes}, gpu_pointer{alloc.allocate(bytes)} {
}

GPUBlock::~GPUBlock() {
    gpu::GPUAllocator<uint8_t> gpu_alloc;
    gpu_alloc.deallocate(gpu_pointer, size);
}
#endif

EINSUMS_SINGLETON_IMPL(BlockManager);

std::weak_ptr<uint8_t[]> BlockManager::request_block(size_t bytes) {
    // Try to clean up stale blocks.
    for (int i = 0; i < 10 && _alloc.available_size() < bytes; i++) {
        bool erased = false;
        {
            auto lock = std::lock_guard(*this);

            for (auto it = _block_list.begin(); it != _block_list.end(); it++) {
                if (it->use_count() == 1) {
                    _block_list.erase(it);
                    erased = true;
                    break;
                }
            }
        }

        // Wait a bit to see if any blocks are freed up.
        if (!erased) {
            std::this_thread::yield();
        }
    }

    if (_alloc.available_size() < bytes) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "Could not allocate enough memory for the requested block, even after stale blocks were removed! "
                                "Requested {} bytes, but there are only {} bytes available.",
                                bytes, _alloc.available_size());
    }

    auto lock = std::lock_guard(*this);

    _block_list.push_back(std::allocate_shared<uint8_t[]>(_alloc, bytes));

    return std::weak_ptr<uint8_t[]>(_block_list.back());
}

#ifdef EINSUMS_COMPUTE_CODE
std::weak_ptr<GPUBlock> BlockManager::request_gpu_block(size_t bytes) { // Try to clean up stale blocks.
    for (int i = 0; i < 10 && _gpu_alloc.available_size() < bytes; i++) {
        bool erased = false;
        {
            auto lock = std::lock_guard(*this);

            for (auto it = _gpu_block_list.begin(); it != _gpu_block_list.end(); it++) {
                if (it->use_count() == 1) {
                    _gpu_block_list.erase(it);
                    erased = true;
                    break;
                }
            }
        }

        // Wait a bit to see if any blocks are freed up.
        if (!erased) {
            std::this_thread::yield();
        }
    }

    if (_gpu_alloc.available_size() < bytes) {
        EINSUMS_THROW_EXCEPTION(
            std::runtime_error,
            "Could not allocate enough memory on the GPU for the requested block, even after stale blocks were removed! "
            "Requested {} bytes, but there are only {} bytes available.",
            bytes, _gpu_alloc.available_size());
    }

    auto lock = std::lock_guard(*this);

    _gpu_block_list.push_back(std::make_shared<GPUBlock>(_gpu_alloc, bytes));

    return std::weak_ptr<GPUBlock>(_gpu_block_list.back());
}
#endif

} // namespace einsums