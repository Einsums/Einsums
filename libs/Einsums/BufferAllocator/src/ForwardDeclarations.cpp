//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------


#include <Einsums/BufferAllocator/BufferAllocator.hpp>

using namespace einsums;

namespace einsums {

static void check_requirements() {
    constexpr bool requirement = requires(BufferAllocator<int> alloc, size_t n) {
        { *alloc.allocate(n) } -> std::same_as<typename BufferAllocator<int>::value_type &>;
        { alloc.deallocate(alloc.allocate(n), n) };
    } && std::copy_constructible<BufferAllocator<int>> && std::equality_comparable<BufferAllocator<int>>;

    static_assert(requirement);
}

#ifndef WINDOWS

template struct BufferAllocator<float>;
template struct BufferAllocator<double>;
template struct BufferAllocator<std::complex<float>>;
template struct BufferAllocator<std::complex<double>>;

#endif

} // namespace einsums