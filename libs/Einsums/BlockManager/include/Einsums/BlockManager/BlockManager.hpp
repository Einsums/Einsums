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
    size_t   size;  /// The size of the block in bytes.
    gpu::GPUPointer<uint8_t> gpu_pointer;   /// A smart pointer to the block of memory.
};
#endif

/**
 * @struct BlockManager
 *
 * @brief Manages allocations to ensure they don't go over the program limits.
 *
 * @versionadded{2.0.0}
 */
struct EINSUMS_EXPORT BlockManager final : public design_pats::Lockable<std::mutex> {
    EINSUMS_SINGLETON_DEF(BlockManager)

    /**
     * @fn get_singleton()
     *
     * Gets a reference to the single unique implementation of this class.
     *
     * @versionadded{2.0.0}
     */
  public:
    ~BlockManager() = default;

    /**
     * @typedef BlockType
     *
     * @brief The type used to represent blocks of data.
     *
     * @versionadded{2.0.0}
     */
    using BlockType = std::shared_ptr<uint8_t[]>;

    /**
     * @brief Requests a block of data with a given size in bytes.
     *
     * @param bytes The number of bytes requested.
     *
     * @return A weak pointer to the block. As long as the weak pointer is not locked, the block may be deallocated for something else to take its place.
     *
     * @versionadded{2.0.0}
     */
    std::weak_ptr<uint8_t[]> request_block(size_t bytes);

#ifdef EINSUMS_COMPUTE_CODE
    /**
     * @brief Requests a block of data with a given size in bytes on the GPU.
     *
     * @param bytes The number of bytes requested.
     *
     * @return A weak pointer to the block. As long as the weak pointer is not locked, the block may be deallocated for something else to take its place.
     *
     * @versionadded{2.0.0}
     */
    std::weak_ptr<GPUBlock> request_gpu_block(size_t bytes);
#endif

  private:
    BlockManager() = default;

    /**
     * @var _block_list
     *
     * @brief The list of blocks currently allocated.
     *
     * @versionadded{2.0.0}
     */
    BufferList<BlockType> _block_list;

    /**
     * @var _alloc
     *
     * @brief The allocator to use to allocate the blocks.
     *
     * @versionadded{2.0.0}
     */
    BufferAllocator<uint8_t> _alloc;

#ifdef EINSUMS_COMPUTE_CODE

    /**
     * @var _gpu_block_list
     *
     * @brief The list of blocks currently allocated on the GPU.
     *
     * @versionadded{2.0.0}
     */
    BufferList<std::shared_ptr<GPUBlock>> _gpu_block_list;

    /**
     * @var _gpu_alloc
     *
     * @brief The allocator to use to allocate the blocks on the GPU.
     *
     * @versionadded{2.0.0}
     */
    gpu::GPUAllocator<uint8_t> _gpu_alloc;
#endif
};

} // namespace einsums