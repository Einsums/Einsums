

//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Utils.hpp>

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
} // namespace hptt
