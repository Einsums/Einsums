#include "einsums/Print.hpp"
#include "einsums/STL.hpp"

#include <cassert>
#include <cstdlib>

namespace einsums::detail {

auto allocate_aligned_memory(size_t align, size_t size) -> void * {
    assert(align >= sizeof(void *));
    assert(align && !(align & (align - 1))); // Align should be a power of 2 but disallow 0.

    if (size == 0) {
        return nullptr;
    }

    void *ptr{nullptr};
#if defined(_WIN32) || defined(_WIN64)
    ptr = malloc(size);
#else
    int rc = posix_memalign(&ptr, align, size);

    if (rc != 0) {
        println("posix_memalign returned non-zero!");
        return nullptr;
    }
#endif

    return ptr;
}

void deallocate_aligned_memory(void *ptr) noexcept {
    return free(ptr);
}

} // namespace einsums::detail