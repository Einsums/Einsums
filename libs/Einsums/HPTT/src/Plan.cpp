//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

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

    size_t size = loopOrder_.size();

    error = fwrite(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    error = fwrite(loopOrder_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    size = numThreadsAtLoop_.size();

    error = fwrite(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    error = fwrite(numThreadsAtLoop_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    std::vector<uint32_t> offsets(numTasks_);

    long offset_table_pos = ftell(fp);

    int error2 = fseek(fp, (long)numTasks_ * sizeof(uint32_t), SEEK_CUR);

    if (error2 != 0) {
        perror("Error seeking in file!");
        throw std::runtime_error("IO error.");
    }

    for (int i = 0; i < numTasks_; i++) {
        auto          curr_node = rootNodes_[i];
        NodeConstants constants{.start      = curr_node.start,
                                .end        = curr_node.end,
                                .inc        = curr_node.inc,
                                .offDiffAB  = curr_node.offDiffAB,
                                .lda        = curr_node.lda,
                                .ldb        = curr_node.ldb,
                                .offsetNext = 0,
                                .indexA     = curr_node.indexA,
                                .indexB     = curr_node.indexB};
        offsets[i] = ftell(fp);
        error      = fwrite(&constants, sizeof(NodeConstants), 1, fp);

        if (error < 1) {
            perror("Error writing to file!");
            throw std::runtime_error("IO error");
        }
    }

    error2 = fseek(fp, offset_table_pos, SEEK_SET);

    if (error2 != 0) {
        perror("Error seeking in file!");
        throw std::runtime_error("IO error.");
    }

    error = fwrite(offsets.data(), sizeof(uint32_t), numTasks_, fp);

    if (error < numTasks_) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    for (int i = 0; i < numTasks_; i++) {
        auto curr_node = rootNodes_[i];

        int index = -1;

        for (int j = 0; j < numTasks_; j++) {
            if (curr_node.next == &rootNodes_[j]) {
                index = j;
                break;
            }
        }

        error2 = fseek(fp, (long)offsets[i] + offsetof(NodeConstants, offsetNext), SEEK_SET);

        if (error2 != 0) {
            perror("Error seeking in file!");
            throw std::runtime_error("IO error.");
        }

        if (index != -1) {
            error = fwrite(&offsets[index], sizeof(uint32_t), 1, fp);

            if (error < numTasks_) {
                perror("Error writing to file!");
                throw std::runtime_error("IO error.");
            }
        }
    }
}

Plan::Plan(std::FILE *fp) {
    size_t error = fread(&numTasks_, sizeof(int), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    size_t size;

    error = fread(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    loopOrder_.resize(size);

    error = fread(loopOrder_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    error = fread(&size, sizeof(size_t), 1, fp);

    if (error < 1) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    numThreadsAtLoop_.resize(size);

    error = fread(numThreadsAtLoop_.data(), sizeof(int), size, fp);

    if (error < size) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    std::vector<uint32_t> offsets;
    offsets.reserve(numTasks_);

    long offset_table_pos = ftell(fp);

    int error2 = fseek(fp, (long)numTasks_ * sizeof(uint32_t), SEEK_CUR);

    if (error2 != 0) {
        perror("Error seeking in file!");
        throw std::runtime_error("IO error.");
    }

    rootNodes_.resize(numTasks_);

    for (int i = 0; i < numTasks_; i++) {
        NodeConstants constants;
        error = fread(&constants, sizeof(NodeConstants), 1, fp);

        rootNodes_[i].start     = constants.start;
        rootNodes_[i].end       = constants.end;
        rootNodes_[i].inc       = constants.inc;
        rootNodes_[i].lda       = constants.lda;
        rootNodes_[i].ldb       = constants.ldb;
        rootNodes_[i].indexA    = constants.indexA;
        rootNodes_[i].indexB    = constants.indexB;
        rootNodes_[i].offDiffAB = constants.offDiffAB;

        if (error < 1) {
            perror("Error writing to file!");
            throw std::runtime_error("IO error");
        }
    }

    error2 = fseek(fp, offset_table_pos, SEEK_SET);

    if (error2 != 0) {
        perror("Error seeking in file!");
        throw std::runtime_error("IO error.");
    }

    error = fread(offsets.data(), sizeof(uint32_t), numTasks_, fp);

    if (error < numTasks_) {
        perror("Error writing to file!");
        throw std::runtime_error("IO error.");
    }

    for (int i = 0; i < numTasks_; i++) {
        auto curr_node = rootNodes_[i];

        int index = -1;

        for (int j = 0; j < numTasks_; j++) {
            if (curr_node.next == &rootNodes_[j]) {
                index = j;
                break;
            }
        }

        error2 = fseek(fp, (long)offsets[i] + offsetof(NodeConstants, offsetNext), SEEK_SET);

        if (error2 != 0) {
            perror("Error seeking in file!");
            throw std::runtime_error("IO error.");
        }

        if (index != -1) {
            error = fwrite(&offsets[index], sizeof(uint32_t), 1, fp);

            if (error < numTasks_) {
                perror("Error writing to file!");
                throw std::runtime_error("IO error.");
            }
        }
    }
}
} // namespace hptt
