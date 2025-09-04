#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
namespace hptt {

struct NodeConstants {
    ptrdiff_t start, end, inc, offDiffAB;
    size_t    lda, ldb;
    uint32_t  offsetNext;
    bool      indexA, indexB;
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