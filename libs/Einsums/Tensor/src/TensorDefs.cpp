//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

#include <H5Lpublic.h>
#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace einsums {

TENSOR_DEFINE_RANK(BlockTensor, 2)
TENSOR_DEFINE_RANK(BlockTensor, 3)
TENSOR_DEFINE_RANK(BlockTensor, 4)

TENSOR_DEFINE(DiskTensor)
TENSOR_DEFINE(DiskView)

TENSOR_DEFINE_ALLOC_RANK(GeneralTensor, 0, std::allocator)
TENSOR_ALLOC_DEFINE(GeneralTensor, std::allocator)
TENSOR_ALLOC_DEFINE(GeneralTensor, BufferAllocator)
TENSOR_DEFINE(TensorView)

TENSOR_DEFINE(TiledTensor)
TENSOR_DEFINE(TiledTensorView)

#ifndef EINSUMS_WINDOWS
template class GeneralRuntimeTensor<float, std::allocator<float>>;
template class GeneralRuntimeTensor<double, std::allocator<double>>;
template class GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>;
template class GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>;

template class GeneralRuntimeTensor<float, BufferAllocator<float>>;
template class GeneralRuntimeTensor<double, BufferAllocator<double>>;
template class GeneralRuntimeTensor<std::complex<float>, BufferAllocator<std::complex<float>>>;
template class GeneralRuntimeTensor<std::complex<double>, BufferAllocator<std::complex<double>>>;

template class RuntimeTensorView<float>;
template class RuntimeTensorView<double>;
template class RuntimeTensorView<std::complex<float>>;
template class RuntimeTensorView<std::complex<double>>;
#endif

static bool verify_path(std::string const &path) {
    if (path.size() == 0) {
        return true;
    }

    if (path[0] != '/' && path[0] != '.') {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "The format of the disk tensor name \"{}\" was invalid! It must be formatted as a path.", path);
    }

    return true;
}

namespace detail {

std::string temp_tensor_name() {
    constexpr char base_64_conv[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                     'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                                     'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                     'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-'};

    auto &singleton = Einsums_Tensor_vars::get_singleton();

    std::ostringstream out;

    out << "temp_";

    // Safely get the serial number and increase the counter.
    auto const value = singleton.temp_counter.fetch_add(1);

    // Now, mess it up. The FNV algorithm looks good.
    constexpr uint64_t FNV_offset = 0xcbf29ce484222325;
    constexpr uint64_t FNV_prime  = 0x00000100000001b3;

    union {
        uint64_t qword;
        uint8_t  bytes[sizeof(uint64_t)];
    } map;

    map.qword = value;

    uint64_t hash = FNV_offset;

#pragma unroll
    for (unsigned int i = 0; i < sizeof(uint64_t); i++) {
        hash *= FNV_prime;

        hash ^= static_cast<uint64_t>(map.bytes[i]);
    }

    // Then convert into base-64 but with safe options for 62 and 63.
    // The digits will be in reverse order, but I don't really care. This is just to make it so it doesn't look like a number.
#pragma unroll
    for (unsigned int i = 0; i < (sizeof(uint64_t) * 8) / 6 + 1; i++) {
        // Calculate the remainder and quotient when dividing by 64.
        uint64_t const remainder = hash & 0x3f;
        hash >>= 6;

        // Get the next character.
        out << base_64_conv[remainder];
    }

    return out.str();
}

bool verify_exists(hid_t loc_id, std::string const &path, hid_t lapl_id) {
    if (!verify_path(path)) {
        return false;
    }
    if (path.length() == 0) {
        return false;
    }

    std::string temp_path;

    temp_path.reserve(path.length());

    for (auto ch : path) {
        if (ch == '/' && temp_path.length() > 0) {
            auto res = H5Lexists(loc_id, temp_path.c_str(), lapl_id);

            if (res <= 0) {
                return false;
            }
        }
        temp_path.push_back(ch);
    }

    return H5Lexists(loc_id, temp_path.c_str(), lapl_id) > 0;
}
} // namespace detail

} // namespace einsums
