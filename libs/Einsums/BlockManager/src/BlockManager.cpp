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

BlockManager::BlockManager() {
    _block_list = new BufferList<BlockType>();
    _alloc      = new BufferAllocator<uint8_t>();

#ifdef EINSUMS_COMPUTE_CODE
    _gpu_block_list = new BufferList<std::shared_ptr<GPUBlock>>();
    _gpu_alloc      = new gpu::GPUAllocator<uint8_t>();
#endif
}

BlockManager::~BlockManager() {
    std::fprintf(stderr, "Finalizing the BlockManager.\n");

    std::fprintf(stderr, "Freeing the block list.\n");
    if (_block_list) {
        delete _block_list;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }

    std::fprintf(stderr, "Freeing the allocator.\n");
    if (_alloc) {
        delete _alloc;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }

#ifdef EINSUMS_COMPUTE_CODE
    std::fprintf(stderr, "Freeing the GPU block list.");
    if (_gpu_block_list) {
        delete _gpu_block_list;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }

    std::fprintf(stderr, "Freeing the GPU allocator.");
    if (_gpu_alloc) {
        delete _gpu_alloc;
        std::fprintf(stderr, "Done.\n");
    } else {
        std::fprintf(stderr, "No work to do.\n");
    }
#endif
}

std::weak_ptr<uint8_t[]> BlockManager::request_block(size_t bytes) {
    // Try to clean up stale blocks.
    for (int i = 0; i < 10 && _alloc->available_size() < bytes; i++) {
        bool erased = false;
        {
            auto lock = std::lock_guard(*this);

            for (auto it = _block_list->begin(); it != _block_list->end(); it++) {
                if (it->use_count() == 1) {
                    _block_list->erase(it);
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

    if (_alloc->available_size() < bytes) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "Could not allocate enough memory for the requested block, even after stale blocks were removed! "
                                "Requested {} bytes, but there are only {} bytes available.",
                                bytes, _alloc->available_size());
    }

    auto lock = std::lock_guard(*this);

    _block_list->push_back(std::allocate_shared_for_overwrite<uint8_t[]>(*_alloc, (size_t)bytes));

    return std::weak_ptr<uint8_t[]>(_block_list->back());
}

#ifdef EINSUMS_COMPUTE_CODE
std::weak_ptr<GPUBlock> BlockManager::request_gpu_block(size_t bytes) { // Try to clean up stale blocks.
    for (int i = 0; i < 10 && _gpu_alloc->available_size() < bytes; i++) {
        bool erased = false;
        {
            auto lock = std::lock_guard(*this);

            for (auto it = _gpu_block_list->begin(); it != _gpu_block_list->end(); it++) {
                if (it->use_count() == 1) {
                    _gpu_block_list->erase(it);
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

    if (_gpu_alloc->available_size() < bytes) {
        EINSUMS_THROW_EXCEPTION(
            std::runtime_error,
            "Could not allocate enough memory on the GPU for the requested block, even after stale blocks were removed! "
            "Requested {} bytes, but there are only {} bytes available.",
            bytes, _gpu_alloc->available_size());
    }

    auto lock = std::lock_guard(*this);

    _gpu_block_list->push_back(std::make_shared<GPUBlock>(*_gpu_alloc, bytes));

    return std::weak_ptr<GPUBlock>(_gpu_block_list->back());
}
#endif

} // namespace einsums