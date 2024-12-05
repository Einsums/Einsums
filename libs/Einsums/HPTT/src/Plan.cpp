

//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Utils.hpp>

namespace hptt {

Plan::Plan(std::vector<int> loopOrder, std::vector<int> numThreadsAtLoop)
    : rootNodes_(nullptr), loopOrder_(loopOrder), numThreadsAtLoop_(numThreadsAtLoop) {
    numTasks_ = 1;
    for (auto nt : numThreadsAtLoop)
        numTasks_ *= nt;
    rootNodes_ = new ComputeNode[numTasks_];
}

Plan::~Plan() {
    if (rootNodes_ != nullptr)
        delete[] rootNodes_;
}

ComputeNode const *Plan::getRootNode_const(int threadId) const {
    return &rootNodes_[threadId];
}
ComputeNode *Plan::getRootNode(int threadId) const {
    return &rootNodes_[threadId];
}

void Plan::print() const {
    printVector(loopOrder_, "LoopOrder");
    printVector(numThreadsAtLoop_, "Parallelization");
}
} // namespace hptt
