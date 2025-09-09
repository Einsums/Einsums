#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
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
    auto bytes = std::bit_cast<std::array<uint8_t, sizeof(T)>>(value);
    std::ranges::reverse(bytes);
    return std::bit_cast<T>(bytes);
}

template <>
constexpr uint8_t byteswap(uint8_t value) noexcept {
    return value;
}

template <>
constexpr uint16_t byteswap(uint16_t value) noexcept {
    union {
        uint16_t whole;
        uint8_t  bytes[2];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[1]);
    return convert.whole;
}

template <>
constexpr uint32_t byteswap(uint32_t value) noexcept {
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
constexpr int32_t byteswap(int32_t value) noexcept {
    union {
        int32_t whole;
        uint8_t bytes[4];
    } convert;

    convert.whole = value;
    std::swap(convert.bytes[0], convert.bytes[3]);
    std::swap(convert.bytes[1], convert.bytes[2]);
    return convert.whole;
}

struct NodeConstants {
    ptrdiff_t start, end, inc, offDiffAB;
    size_t    lda, ldb;
    uint16_t      indexA, indexB, has_next;
};

struct TransposeConstants {
    int    dim;
    int    numThreads;
    size_t innerStrideA;
    size_t innerStrideB;
    int    selectedParallelStrategy;
    int    selectedLoopOrderId;
    bool   conjA;
};

/**
 * File header specification for transpose files.
 */
struct FileHeader {
    char magic[4];
    char version[4];

    uint32_t checksum;
};

void setupFile(std::FILE *fp);

uint32_t computeChecksum(std::FILE *fp);

int verifyFile(std::FILE *fp);

} // namespace hptt