//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/HPTT/ComputeNode.hpp>

#include <vector>

namespace hptt {

/**
 * @class Plan
 *
 * \brief A plan encodes the execution of a tensor transposition.
 *
 * It stores the loop order and parallelizes each loop.
 */
class Plan {
  public:
    Plan() : rootNodes_(), numTasks_(0) {}

    /**
     * Construct a plan. Initialize the loop order and the number of threads for each level of loop.
     *
     * @param loopOrder Holds the indices in the order they are to be updated.
     * @param numThreadsAtLoop Contains the number of threads for each level of loop. Used to find the number of tasks as well.
     */
    Plan(std::vector<int> loopOrder, std::vector<int> numThreadsAtLoop);

    ~Plan() = default;

    /**
     * Get the root node, but get a const pointer to it.
     */
    ComputeNode const *getRootNode(int threadId) const;

    /**
     * Get the root node, but get a non-const pointer to it.
     */
    ComputeNode *getRootNode(int threadId);

    /**
     * Get the number of tasks for the plan.
     */
    int getNumTasks() const;

    /**
     * Print the loop order and number of threads for each level of loop.
     */
    void print() const;

  private:
    /**
     * @var numTasks_
     *
     * The number of tasks to spawn for the calculation.
     */
    int numTasks_;

    /**
     * @var loopOrder_
     *
     * Holds the indices in the order that they will be used.
     *
     * For example, if \f$ B_{1,0,2} \gets A_{0,1,2}\f$. loopOrder_ = {1,0,2} denotes that B is
     * traversed in a linear fashion.
     */
    std::vector<int> loopOrder_;

    /**
     * @var numThreadsAtLoop_
     *
     * Holds the number of threads for each level of the loop.
     */
    std::vector<int> numThreadsAtLoop_;

    /**
     * @var rootNodes_
     *
     * Holds the nodes used to compute the transposition.
     */
    std::vector<ComputeNode> rootNodes_;
};

} // namespace hptt
