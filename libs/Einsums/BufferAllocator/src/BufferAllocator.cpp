//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Logging.hpp>

#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
#    include <mimalloc-new-delete.h>
#    include <mimalloc.h>
#elif defined(EINSUMS_WINDOWS)
#    include <malloc.h>
#else
#    include <cstdlib>
#endif

namespace einsums::detail {

void *allocate(size_t n) {
    void *ptr = nullptr;

    constexpr size_t alignment = 64;

    size_t const modified_n = n + alignment - 1;
    size_t const rounded_n  = modified_n & ~(alignment - 1);

    EINSUMS_LOG_TRACE("Allocating {} bytes, modified to {}, and rounded up to {} bytes.", n, modified_n, rounded_n);

#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
    ptr = mi_malloc_aligned(rounded_n, alignment);
#elif defined(EINSUMS_WINDOWS)
    if (n != 0) {
        ptr = _aligned_malloc(rounded_n, alignment);
    }
#else
    ptr = std::aligned_alloc(alignment, rounded_n);
#endif
    return ptr;
}

void deallocate(void *p) {
#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
    mi_free(p);
#elif defined(EINSUMS_WINDOWS)
    _aligned_free(p);
#else
    free(static_cast<void *>(p));
#endif
}

} // namespace einsums::detail