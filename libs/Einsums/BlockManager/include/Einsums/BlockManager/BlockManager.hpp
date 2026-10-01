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
struct EINSUMS_EXPORT GPUBlock {
    GPUBlock(gpu::GPUAllocator<uint8_t> &alloc, size_t size);
    ~GPUBlock();
    size_t   size;
    gpu::GPUPointer<uint8_t> gpu_pointer;
};
#endif

struct EINSUMS_EXPORT BlockManager final : public design_pats::Lockable<std::mutex> {
    EINSUMS_SINGLETON_DEF(BlockManager)
  public:
    ~BlockManager() = default;
    using BlockType = std::shared_ptr<uint8_t[]>;

    std::weak_ptr<uint8_t[]> request_block(size_t bytes);

#ifdef EINSUMS_COMPUTE_CODE
    std::weak_ptr<GPUBlock> request_gpu_block(size_t bytes);
#endif

  private:
    BlockManager() = default;

    BufferList<BlockType> _block_list;

    BufferAllocator<uint8_t> _alloc;

#ifdef EINSUMS_COMPUTE_CODE
    BufferList<std::shared_ptr<GPUBlock>> _gpu_block_list;

    gpu::GPUAllocator<uint8_t> _gpu_alloc;
#endif
};

} // namespace einsums