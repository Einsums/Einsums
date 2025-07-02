//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/GPUMemory/GPUAllocator.hpp>
#include <Einsums/GPUMemory/GPUPointer.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <list>
#include <mutex>
namespace einsums {
namespace gpu {

struct GPUMemoryTracker final : design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(GPUMemoryTracker)

    /**
     * @brief Finds memory that can be deallocated and deallocates it.
     *
     * This goes through the list of allocations, oldest first, and checks if
     * it can be deleted. Then, it deletes them.
     *
     * @param until The amount of memory we are trying to free. If we free more than this,
     * then this function will exit early.
     *
     * @return The amount of memory we freed.
     */
    size_t cleanup_allocations(size_t until = std::numeric_limits<size_t>::max());

    /**
     * @brief Checks to see if all the memory allocations are sticky.
     *
     * This is important to check, since having a bunch of sticky allocations can cause
     * issues when allocating temporary buffers.
     */
    bool is_all_sticky();

    /**
     * @brief Searches for a handle or any of its parents.
     *
     * If the pointer does not exist, return nullptr.
     */
    GPUPointer<void> get_pointer_for_handle_no_alloc(size_t handle);

    /**
     * @brief Get the pointer associated with the given handle.
     *
     * If the pointer does not exist, then reallocate it. If the handle is a view of another handle,
     * then this will look for parents of the view as well.
     *
     * @param handle The handle to query.
     * @param nelems The number of elements to allocate if needed.
     *
     * @returns The pointer associated with the handle and whether the pointer needed to be allocated.
     */
    template <typename T>
    std::tuple<GPUPointer<T>, bool> get_pointer(size_t handle, size_t nelems) {
        auto out = get_pointer_for_handle_no_alloc(handle);

        if (!out) {
            // We couldn't find the pointer or any parent, so we need to allocate the pointer.
            auto data = allocate_pointer(nelems, false, handle);

            return {GPUPointer<T>(data.ptr), true};
        } else {
            // We found the pointer, return it.
            return {GPUPointer<T>(out), false};
        }
    }

    /**
     * @brief Allocates a sticky pointer.
     *
     * A sticky pointer is one that won't go away until it is explicitly freed.
     *
     * @return The handle for the pointer and the pointer itself.
     */
    template <typename T>
    std::tuple<size_t, GPUPointer<T>> create_sticky_pointer(size_t nelems) {
        size_t handle = create_handle();

        auto data = allocate_pointer(nelems, true, handle);

        return {handle, GPUPointer<T>(data.ptr)};
    }

    /**
     * @brief Creates a new handle.
     *
     * This does not allocate any memory. It just creates the handle. Pass this into get_pointer to actually
     * allocate the memory.
     */
    size_t create_handle();

    /**
     * @brief Tells the tracker that a handle is a contiguous view of another.
     *
     * @param handle The view handle.
     * @param parent The handle of the view's parent.
     * @param offset The position in the buffer that the view starts.
     */
    void handle_view(size_t handle, size_t parent, size_t offset);

    /**
     * @brief Remove a view from the view list.
     */
    void remove_view(size_t handle);

    /**
     * @brief Releases a hold on a handle.
     *
     * No memory is actually freed when this is called. If the use count for a handle goes to zero,
     * the memory still remains. If the handle is acquired again, the pointer will be the same.
     * However, if the use count of the handle goes to zero, then the memory associated with that
     * handle may be freed to make way for other buffers. In this case, the pointer returned by get_pointer
     * will not necessarily be the same as what it was. If the handle is sticky, this function will not
     * mark that handle as not sticky unless it is told to do so. A sticky handle is not freed during
     * clean up operations. By default, this does not remove the sticky property.
     *
     * @param handle The handle to release.
     * @param remove_sticky If true, sets a sticky handle to be not sticky, allowing it to be removed
     * during clean up operations.
     */
    void release_handle(size_t handle, bool remove_sticky = false);

    /**
     * @brief Check to see if a handle already has a pointer allocated.
     */
    bool handle_is_allocated(size_t handle);

  private:
    GPUMemoryTracker() = default;

    ~GPUMemoryTracker();

    struct MemoryData {
        size_t  handle;
        void   *ptr;
        size_t  size;
        int32_t use_count;
        int32_t is_sticky;
    };

    struct HandleRelations {
        size_t handle;
        size_t offset;
        size_t contiguous_parent_handle;
    };

    MemoryData allocate_pointer(size_t nelems, bool sticky, size_t handle);

    GPUAllocator<void> alloc_;

    std::list<MemoryData> memory_list_;

    std::list<HandleRelations> view_list_;

    /**
     * @var serialize_
     *
     * @brief Holds a value to help generate unique handles.
     *
     * These start at 1 so that uninitialized handles can be zero.
     */
    size_t serialize_{1};
};

} // namespace gpu
} // namespace einsums