/*
  Copyright 2018 Paul Springer

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Utils.hpp>

#include <stdexcept>

#include "Einsums/HPTT/Files.hpp"

namespace hptt {

Plan::Plan(std::vector<int> loopOrder, std::vector<int> numThreadsAtLoop)
    : rootNodes_(), loopOrder_(loopOrder), numThreadsAtLoop_(numThreadsAtLoop) {
    numTasks_ = 1;
    for (auto nt : numThreadsAtLoop)
        numTasks_ *= nt;
    rootNodes_.resize(numTasks_);
}

ComputeNode const *Plan::getRootNode(int threadId) const {
    return &rootNodes_.at(threadId);
}
ComputeNode *Plan::getRootNode(int threadId) {
    return &rootNodes_.at(threadId);
}

void Plan::print() const {
    printVector(loopOrder_, "LoopOrder");
    printVector(numThreadsAtLoop_, "Parallelization");
}

int Plan::getNumTasks() const {
    return rootNodes_.size();
}

void Plan::writeToFile(std::FILE *fp) const {
    size_t error = fwrite(&numTasks_, sizeof(int), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    std::fflush(fp);

    size_t size = loopOrder_.size();

    error = fwrite(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    std::fflush(fp);

    error = fwrite(loopOrder_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    std::fflush(fp);

    size = numThreadsAtLoop_.size();

    error = fwrite(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    std::fflush(fp);

    error = fwrite(numThreadsAtLoop_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    std::fflush(fp);

    // Offsets for the root nodes.
    std::vector<uint32_t> offsets(numTasks_);

    long offset_table_pos = ftell(fp);

    int error2 = fseek(fp, (long)numTasks_ * sizeof(uint32_t), SEEK_CUR);

    if (error2 != 0) {
        perror("Error seeking in file!");
        throw std::runtime_error("IO error.");
    }

    // Write the root nodes.
    for (int i = 0; i < numTasks_; i++) {
        auto *curr_node = &rootNodes_[i];
        // Get the offset of the root node.
        offsets[i] = ftell(fp);

        while (curr_node != nullptr) {
            NodeConstants constants{.start     = curr_node->start,
                                    .end       = curr_node->end,
                                    .inc       = curr_node->inc,
                                    .offDiffAB = curr_node->offDiffAB,
                                    .lda       = curr_node->lda,
                                    .ldb       = curr_node->ldb,
                                    .indexA    = curr_node->indexA,
                                    .indexB    = curr_node->indexB,
                                    .has_next  = (uint16_t)((curr_node->next == nullptr) ? 0 : 1),
                                    .pad       = 0};
            error = fwrite(&constants, sizeof(NodeConstants), 1, fp);

            if (error < 1) {
                perror("Error writing to file!");
                throw std::runtime_error("IO error");
            }

            std::fflush(fp);

            curr_node = curr_node->next;
        }
    }

    error2 = fseek(fp, offset_table_pos, SEEK_SET);

    if (error2 != 0) {
        perror("Error seeking in file!");
        throw std::runtime_error("IO error.");
    }

    error = fwrite(offsets.data(), sizeof(uint32_t), numTasks_, fp);

    std::fflush(fp);

    if (error < numTasks_) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }
}

Plan::Plan(std::FILE *fp, bool swap_endian) {
    size_t error = fread(&numTasks_, sizeof(int), 1, fp);

    if (error < 1) {
        perror("Error reading from file!");
        throw std::runtime_error("IO error.");
    }

    if (swap_endian) {
        numTasks_ = byteswap(numTasks_);
    }

    size_t size;

    error = fread(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error reading from file!");
        throw std::runtime_error("IO error.");
    }

    if (swap_endian) {
        size = byteswap(size);
    }

    loopOrder_.resize(size);

    error = fread(loopOrder_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error reading from file!");
        throw std::runtime_error("IO error.");
    }

    if (swap_endian) {
        for (int i = 0; i < loopOrder_.size(); i++) {
            loopOrder_[i] = byteswap(loopOrder_[i]);
        }
    }

    error = fread(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error reading from file!");
        throw std::runtime_error("IO error.");
    }

    if (swap_endian) {
        size = byteswap(size);
    }

    numThreadsAtLoop_.resize(size);

    error = fread(numThreadsAtLoop_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error reading from file!");
        throw std::runtime_error("IO error.");
    }

    if (swap_endian) {
        for (int i = 0; i < numThreadsAtLoop_.size(); i++) {
            numThreadsAtLoop_[i] = byteswap(numThreadsAtLoop_[i]);
        }
    }

    // Get the offsets for each root node.
    std::vector<uint32_t> offsets;
    offsets.resize(numTasks_);

    error = fread(offsets.data(), sizeof(uint32_t), numTasks_, fp);

    if (error < numTasks_) {
        perror("Error reading from file!");
        throw std::runtime_error("IO error");
    }

    if (swap_endian) {
        for (int i = 0; i < numTasks_; i++) {
            offsets[i] = byteswap(offsets[i]);
        }
    }

    rootNodes_.resize(numTasks_);

    if (swap_endian) {
        for (int i = 0; i < numTasks_; i++) {
            NodeConstants constants;
            auto         *curr_node = &rootNodes_[i];

            int error2 = fseek(fp, offsets[i], SEEK_SET);

            if (error2 < 0) {
                perror("Error while seeking!");
                throw std::runtime_error("IO error");
            }

            do {
                error = fread(&constants, sizeof(NodeConstants), 1, fp);

                if (error < 1) {
                    perror("Error reading from file!");
                    throw std::runtime_error("IO error");
                }

                curr_node->start     = byteswap(constants.start);
                curr_node->end       = byteswap(constants.end);
                curr_node->inc       = byteswap(constants.inc);
                curr_node->lda       = byteswap(constants.lda);
                curr_node->ldb       = byteswap(constants.ldb);
                curr_node->indexA    = byteswap(constants.indexA);
                curr_node->indexB    = byteswap(constants.indexB);
                curr_node->offDiffAB = byteswap(constants.offDiffAB);

                if (constants.has_next) {
                    curr_node->next = new ComputeNode;
                } else {
                    curr_node->next = nullptr;
                }
                curr_node = curr_node->next;
            } while (constants.has_next);
        }
    } else {
        for (int i = 0; i < numTasks_; i++) {
            NodeConstants constants;
            auto         *curr_node = &rootNodes_[i];

            int error2 = fseek(fp, offsets[i], SEEK_SET);

            if (error2 < 0) {
                perror("Error while seeking!");
                throw std::runtime_error("IO error");
            }

            do {
                error = fread(&constants, sizeof(NodeConstants), 1, fp);

                if (error < 1) {
                    perror("Error reading from file!");
                    throw std::runtime_error("IO error");
                }

                curr_node->start     = constants.start;
                curr_node->end       = constants.end;
                curr_node->inc       = constants.inc;
                curr_node->lda       = constants.lda;
                curr_node->ldb       = constants.ldb;
                curr_node->indexA    = constants.indexA;
                curr_node->indexB    = constants.indexB;
                curr_node->offDiffAB = constants.offDiffAB;

                if (constants.has_next) {
                    curr_node->next = new ComputeNode;
                } else {
                    curr_node->next = nullptr;
                }
                curr_node = curr_node->next;
            } while (constants.has_next);
        }
    }
}
} // namespace hptt
