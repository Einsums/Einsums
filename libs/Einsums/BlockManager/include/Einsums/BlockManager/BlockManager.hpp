//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUMemory/GPUAllocator.hpp>
#    include <Einsums/GPUMemory/GPUPointer.hpp>
#endif

#include <mutex>
#include <stdexcept>

namespace einsums {

#ifdef EINSUMS_COMPUTE_CODE
/**
 * @struct GPUBlock
 *
 * @brief Represents a block of data on the GPU.
 *
 * @versionadded{2.0.0}
 */
struct EINSUMS_EXPORT GPUBlock {
    GPUBlock(gpu::GPUAllocator<uint8_t> &alloc, size_t size);
    ~GPUBlock();
    size_t   size; //< The size of the block.
    gpu::GPUPointer<uint8_t> gpu_pointer;   //< A smart pointer for accessing the GPU data.
};
#endif

/**
 * @struct BlockManager
 *
 * @brief Handles allocations to make sure they don't exceed the program limits.
 *
 * @versionadded{2.0.0}
 */
struct EINSUMS_EXPORT BlockManager final : public design_pats::Lockable<std::mutex> {
    EINSUMS_SINGLETON_DEF(BlockManager)
  public:
    ~BlockManager() = default;

    /**
     * @typedef BlockType
     *
     * @brief Pointer to a block of data.
     *
     * @versionadded{2.0.0}
     */
    using BlockType = std::shared_ptr<uint8_t[]>;

    /**
     * @brief Request a block of data with a given number of bytes.
     *
     * @param bytes The number of bytes being requested.
     *
     * @return A weak pointer to the block of data. As long as the weak pointer is locked, it can not be deallocated.
     *
     * @throws std::runtime_error If there was not enough memory to allocate the data, or not enough memory could be freed to accomodate.
     *
     * @versionadded{2.0.0}
     */
    std::weak_ptr<uint8_t[]> request_block(size_t bytes);

#ifdef EINSUMS_COMPUTE_CODE
    /**
     * @brief Request a block of data with a given number of bytes on the GPU.
     *
     * @param bytes The number of bytes being requested.
     *
     * @return A weak pointer to the block of GPU data. As long as the weak pointer is locked, it can not be deallocated.
     *
     * @throws std::runtime_error If there was not enough memory to allocate the data, or not enough memory could be freed to accomodate.
     *
     * @versionadded{2.0.0}
     */
    std::weak_ptr<GPUBlock> request_gpu_block(size_t bytes);
#endif

  private:
    BlockManager() = default;

    BufferList<BlockType> _block_list; //< Contains the list of currently allocated blocks.

    BufferAllocator<uint8_t> _alloc; //< Allocator for allocating blocks.

#ifdef EINSUMS_COMPUTE_CODE
    BufferList<std::shared_ptr<GPUBlock>> _gpu_block_list; //< Contains the list of currently allocated GPU blocks.

    gpu::GPUAllocator<uint8_t> _gpu_alloc; //< Allocator for allocating GPU blocks.
#endif
};

} // namespace einsums