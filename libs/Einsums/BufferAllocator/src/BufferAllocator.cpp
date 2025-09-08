//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/BufferAllocator.hpp>

#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
#    include <mimalloc.h>
#endif

namespace einsums::detail {

void* allocate(size_t n) {
    void *ptr = nullptr;

#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
    ptr = mi_malloc_aligned(n, 64);
#elif defined(_ISOC11_SOURCE) || (__STDC_VERSION__ >= 201112L)
    ptr = std::aligned_alloc(64, n );
#else
    // returns zero on success, or an error value. On Linux (and other systems), p is not modified on failure.
    if (posix_memalign(&ptr, 64, n) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}

void deallocate(void* p) {
#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
    mi_free(p);
#else
    free(static_cast<void *>(p));
#endif
}

}