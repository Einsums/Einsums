//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Logging.hpp>

#if __cpp_lib_int_pow2 >= 202002L
#    include <bit>
#endif

#ifdef EINSUMS_WINDOWS
#    include <Windows.h>
#endif

#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
#    include <mimalloc.h>
#else
#    include <cstdlib>
#endif

namespace einsums::detail {

void *allocate(size_t n) {
    void *ptr = nullptr;

    constexpr size_t alignment = 64;
    constexpr size_t mask      = alignment - 1;
    size_t const     remainder = n & mask;
    size_t const     rounding  = (remainder == 0) ? 0 : alignment;
    size_t const     rounded_n = (n & ~mask) + rounding;

#if __cpp_lib_int_pow2 >= 202002L
    static_assert(std::has_single_bit(alignment));
#endif

#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
    ptr = mi_malloc_aligned(n, alignment);
#elif !defined(EINSUMS_WINDOWS)
    ptr = std::aligned_alloc(alignment, rounded_n);
#elif defined(EINSUMS_WINDOWS)
    if (rounded_n != 0) { // Windows will throw errors if n = 0. Everyone else returns null.
        ptr = _aligned_malloc(rounded_n, alignment);
    }
#endif

    if (n != 0 && ptr == nullptr) {
        EINSUMS_LOG_WARN("Requested {} bytes, rounded to {} bytes, but allocator returned null!", n, rounded_n);
    }

    return ptr;
}

void deallocate(void *p) {
#if defined(EINSUMS_HAVE_MALLOC_MIMALLOC)
    mi_free(p);
#elif !defined(EINSUMS_WINDOWS)
    free(p);
#elif defined(EINSUMS_WINDOWS)
    _aligned_free(p);
#endif
}

} // namespace einsums::detail