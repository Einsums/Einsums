//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <version>

#include <Einsums/Config/ExportDefinitions.hpp>

#if defined(__x86__) || defined(__x86_64__)
#    include <immintrin.h>
#endif

namespace hptt {

constexpr char endian_char() {
    if constexpr (std::endian::native == std::endian::big) {
        return 'B';
    } else if constexpr (std::endian::native == std::endian::little) {
        return 'L';
    } else {
        throw std::runtime_error("Mixed endian systems are not supported.");
    }
}

template <std::integral T>
constexpr T byteswap(T value) noexcept {
#if defined(__cpp_lib_byteswap) && __cpp_lib_byteswap >= 202110L
    return std::byteswap(value);
#else
    auto bytes = std::bit_cast<std::array<uint8_t, sizeof(T)>>(value);
    std::ranges::reverse(bytes);
    return std::bit_cast<T>(bytes);
#endif
}

template <>
constexpr uint8_t byteswap<uint8_t>(uint8_t value) noexcept {
    return value;
}

template <>
constexpr uint16_t byteswap<uint16_t>(uint16_t value) noexcept {
    union {
        uint16_t whole;
        uint8_t  bytes[2];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[1]);
    return convert.whole;
}

template <>
constexpr uint32_t byteswap<uint32_t>(uint32_t value) noexcept {
    union {
        uint32_t whole;
        uint8_t  bytes[4];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[3]);
    std::swap(convert.bytes[1], convert.bytes[2]);
    return convert.whole;
}

template <>
constexpr int32_t byteswap<int32_t>(int32_t value) noexcept {
    union {
        int32_t whole;
        uint8_t bytes[4];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[3]);
    std::swap(convert.bytes[1], convert.bytes[2]);
    return convert.whole;
}

typedef struct NodeConstants {
    ptrdiff_t start, end, inc, offDiffAB;
    size_t    lda, ldb;
    uint16_t  indexA, indexB, has_next;
    uint16_t  pad;
} NodeConstants;

typedef struct TransposeConstants {
    int32_t dim;
    int32_t numThreads;
    size_t  innerStrideA;
    size_t  innerStrideB;
    int32_t selectedParallelStrategy;
    int32_t selectedLoopOrderId;
    int32_t conjA;
    int32_t pad;
} TransposeConstants;

/**
 * File header specification for transpose files.
 */
typedef struct FileHeader {
    char magic[4];
    char version[4];

    uint32_t checksum;
} FileHeader;

void EINSUMS_EXPORT setupFile(std::FILE *fp);

uint32_t EINSUMS_EXPORT computeChecksum(std::FILE *fp);

int EINSUMS_EXPORT verifyFile(std::FILE *fp);

} // namespace hptt