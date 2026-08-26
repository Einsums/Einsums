//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

namespace einsums::linear_algebra {
namespace detail {

/// @todo This class can be freely changed. It is provided as a starting point for your convenience. If not needed, it may be removed.

struct EINSUMS_EXPORT Einsums_LinearAlgebra_vars final : design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_LinearAlgebra_vars)

    // Put module-global variables here.

    unsigned int get_gemm_chunk() const;

    unsigned int get_gemv_chunk() const;

    unsigned int get_ger_chunk() const;

    void set_gemm_chunk(unsigned int chunk_dim);

    void set_gemv_chunk(unsigned int chunk_dim);

    void set_ger_chunk(unsigned int chunk_dim);

  private:
    explicit Einsums_LinearAlgebra_vars() = default;

    unsigned int gemm_chunk{500}, gemv_chunk{500}, ger_chunk{500};
};

} // namespace detail
} // namespace einsums