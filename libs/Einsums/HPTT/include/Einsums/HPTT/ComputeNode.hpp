//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <limits>

namespace hptt {

/**
 * \brief A ComputNode encodes a loop.
 */
class ComputeNode {
  public:
    ComputeNode()
        : start(-1), end(-1), inc(0), lda(0), ldb(0), indexA(false), indexB(false), offDiffAB(std::numeric_limits<ptrdiff_t>::min()),
          next(nullptr) {}

    ~ComputeNode() {
        if (next != nullptr)
            delete next;
    }

    ptrdiff_t    start;     //!< start index for at the current loop
    ptrdiff_t    end;       //!< end index for at the current loop
    ptrdiff_t    inc;       //!< increment for at the current loop
    size_t       lda;       //!< stride of A w.r.t. the loop index
    size_t       ldb;       //!< stride of B w.r.t. the loop index
    bool         indexA;    //!< true if index of A is innermost (0)
    bool         indexB;    //!< true if index of B is innermost (0)
    ptrdiff_t    offDiffAB; //!< difference in offset A and B (i.e., A - B) at the current loop
    ComputeNode *next;      //!< next ComputeNode, this might be another loop or 'nullptr'
                            //!< (i.e., indicating that the macro-kernel should be called)
};

} // namespace hptt
