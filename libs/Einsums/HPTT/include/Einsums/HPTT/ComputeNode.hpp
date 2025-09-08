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
