#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUMemory/GPUMemoryTracker.hpp>

namespace einsums {
namespace gpu {

EINSUMS_SINGLETON_IMPL(GPUMemoryTracker)

size_t GPUMemoryTracker::cleanup_allocations(size_t until) {
    auto   lock         = std::lock_guard(*this);
    size_t freed_memory = 0;
    if (memory_list_.size() == 0) {
        return freed_memory;
    }

    auto curr_pos = memory_list_.begin();

    while (curr_pos != memory_list_.end()) {
        auto next_pos = std::next(curr_pos);

        if (curr_pos->use_count <= 0 && !curr_pos->is_sticky) {
            freed_memory += curr_pos->size;
            alloc_.deallocate(GPUPointer<void>(curr_pos->ptr), curr_pos->size);
            memory_list_.erase(curr_pos);

            if (freed_memory >= until) {
                return freed_memory;
            }
        }

        curr_pos = next_pos;
    }

    return freed_memory;
}

bool GPUMemoryTracker::is_all_sticky() {
    auto lock = std::lock_guard(*this);

    for (auto &elem : memory_list_) {
        if (!elem.is_sticky) {
            return false;
        }
    }

    return true;
}

GPUMemoryTracker::~GPUMemoryTracker() {
    for (auto &element : memory_list_) {
        alloc_.deallocate(GPUPointer<void>(element.ptr), element.size);
    }

    memory_list_.clear();
}

GPUMemoryTracker::MemoryData GPUMemoryTracker::allocate_pointer(size_t nelems, bool sticky, size_t handle) {
    if (handle == 0) {
        EINSUMS_THROW_EXCEPTION(
            uninitialized_error,
            "The handle used to request an allocation was 0. This means that it is uninitialized. Please acquire a handle first.");
    }
    // No pointer found. Create a new pointer with this handle.
    // First, check to see if we can allocate that many elements at all.
    if (nelems > alloc_.max_size()) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "Requested allocation size is bigger than the total allowed buffer size! Got {}, maximum size is {}.",
                                nelems, alloc_.max_size());
    }

    // Now, try to allocate until we can.
    GPUPointer<void> out = alloc_.try_allocate(nelems);
    MemoryData       data;

    if (out != nullptr) {
        auto lock      = std::lock_guard(*this);
        data.handle    = handle;
        data.is_sticky = sticky;
        data.ptr       = (void *)out;
        data.size      = nelems;
        data.use_count = 1;

        memory_list_.push_back(data);

        return data;
    }

    // While we can't allocate, we need to clean up the memory list and check to see if there are too many
    // sticky allocations.
    int tries = 0;
    while (!out && tries < 100) {
        if (is_all_sticky()) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not allocate GPU memory! Too many permanent allocations are clogging up "
                                                        "the pipeline. Fix your code to remove some of these permanent allocations.");
        }

        size_t freed = cleanup_allocations(nelems);

        if (freed >= nelems) {
            auto             lock = std::lock_guard(*this);
            GPUPointer<void> out  = alloc_.try_allocate(nelems);

            if (out != nullptr) {
                data.handle    = handle;
                data.is_sticky = sticky;
                data.ptr       = (void *)out;
                data.size      = nelems;
                data.use_count = 1;

                memory_list_.push_back(data);
            }
        }

        tries++;

        // Wait a bit. Maybe some memory will be freed.
        std::this_thread::yield();
    }

    if (!out) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "Could not allocate GPU memory! Something is using too much memory and not giving it back. Check to "
                                "see that you are not allocating large buffers in the same thread as this call before this call.");
    }

    return data;
}

size_t GPUMemoryTracker::create_handle() {
    auto lock = std::lock_guard(*this);

    size_t out = serialize_;
    serialize_++;

    return out;
}

void GPUMemoryTracker::release_handle(size_t handle, bool remove_sticky) {
    auto lock = std::lock_guard(*this);

    for (auto it = memory_list_.begin(); it != memory_list_.end(); it++) {
        if (it->handle == handle) {
            it->use_count--;

            if (remove_sticky) {
                it->is_sticky = false;
            }

            return;
        }
    }

    // Look for the parent.
    for (auto relation : view_list_) {
        if (relation.handle == handle) {
            release_handle(relation.handle, false); // We only want to remove the current stickiness, not the parent's.
        }
    }
}

void GPUMemoryTracker::handle_view(size_t handle, size_t parent, size_t offset) {
    auto lock = std::lock_guard(*this);
    view_list_.emplace_back(HandleRelations{handle, offset, parent});
}

void GPUMemoryTracker::remove_view(size_t handle) {
    auto lock = std::lock_guard(*this);

    // First, find the item that matches this handle.
    HandleRelations relation;
    bool            found = false;

    for (auto it = view_list_.begin(); it != view_list_.end(); it++) {
        if (it->handle == handle) {
            found    = true;
            relation = *it;
            view_list_.erase(it);
            break;
        }
    }

    // If not found, then search through and look for children. Delete them, which will mark them as top-level.
    if (!found) {
        auto curr_pos = view_list_.begin();
        while (curr_pos != view_list_.end()) {
            auto next_pos = std::next(curr_pos);

            if (curr_pos->contiguous_parent_handle == handle) {
                view_list_.erase(curr_pos);
            }

            curr_pos = next_pos;
        }
    } else {
        // Otherwise, search through for all this handle's children and set their parents to this handle's parent.
        for (auto &elem : view_list_) {
            if (elem.contiguous_parent_handle == handle) {
                elem.contiguous_parent_handle = relation.contiguous_parent_handle;
            }
        }
    }
}

GPUPointer<void> GPUMemoryTracker::get_pointer_for_handle_no_alloc(size_t handle) {
    auto lock = std::lock_guard(*this);

    for (auto it = memory_list_.begin(); it != memory_list_.end(); it++) {
        if (it->handle == handle) {
            // Move the item to the end of the list. This will mean it is checked last when looking
            // for memory to free.
            MemoryData temp = *it;
            temp.use_count++;

            memory_list_.erase(it);

            memory_list_.push_back(temp);

            return GPUPointer<void>(temp.ptr);
        }
    }

    // Look for the parent.
    for (auto relation : view_list_) {
        if (relation.handle == handle) {
            // Look for the parent pointer
            auto ptr = get_pointer_for_handle_no_alloc(relation.contiguous_parent_handle);

            // No parent pointer.
            if (ptr == nullptr) {
                return ptr;
            }

            // Offset the parent pointer to get the view pointer.
            return GPUPointer<void>(GPUPointer<uint8_t>((void *)ptr) + relation.offset);
        }
    }

    // No pointer.
    return nullptr;
}

bool GPUMemoryTracker::handle_is_allocated(size_t handle) {
    return get_pointer_for_handle_no_alloc(handle) != nullptr;
}

} // namespace gpu
} // namespace einsums