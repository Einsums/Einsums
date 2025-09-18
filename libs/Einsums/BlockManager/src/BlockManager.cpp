//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BlockManager/BlockManager.hpp>

namespace einsums {

EINSUMS_SINGLETON_IMPL(BlockManager);

#ifdef EINSUMS_COMPUTE_CODE
BlockManager::BlockManager() : _block_list(), _alloc(), _gpu_block_list(), _gpu_alloc() {
}
#else
BlockManager::BlockManager() : _block_list(), _alloc() {
}
#endif

std::weak_ptr<uint8_t[]> BlockManager::request_block(size_t bytes) {
    // Try to clean up stale blocks.
    for (int i = 0; i < 10 && _alloc.available_size() < bytes; i++) {
        bool erased = false;
        {
            auto lock = std::lock_guard(*this);

            for (auto it = _block_list.begin(); it != _block_list.end(); it++) {
                if (it->unique()) {
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
std::weak_ptr<uint8_t[]> BlockManager::request_gpu_block(size_t bytes) { // Try to clean up stale blocks.
    for (int i = 0; i < 10 && _gpu_alloc.available_size() < bytes; i++) {
        bool erased = false;
        {
            auto lock = std::lock_guard(*this);

            for (auto it = _gpu_block_list.begin(); it != _gpu_block_list.end(); it++) {
                if (it->unique()) {
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

    _gpu_block_list.push_back(std::allocate_shared<uint8_t[]>(_gpu_alloc, bytes));

    return std::weak_ptr<uint8_t[]>(_gpu_block_list.back());
}
#endif

} // namespace einsums