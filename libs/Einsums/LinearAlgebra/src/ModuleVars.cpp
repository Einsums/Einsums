//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/LinearAlgebra/ModuleVars.hpp>

namespace einsums::linear_algebra::detail {

EINSUMS_SINGLETON_IMPL(Einsums_LinearAlgebra_vars)

unsigned int Einsums_LinearAlgebra_vars::get_gemm_chunk() const {
    std::lock_guard lock(*this);

    return gemm_chunk;
}

unsigned int Einsums_LinearAlgebra_vars::get_gemv_chunk() const {
    std::lock_guard lock(*this);

    return gemv_chunk;
}

unsigned int Einsums_LinearAlgebra_vars::get_ger_chunk() const {
    std::lock_guard lock(*this);

    return ger_chunk;
}

void Einsums_LinearAlgebra_vars::set_gemm_chunk(unsigned int chunk_dim) {
    std::lock_guard lock(*this);

    gemm_chunk = chunk_dim;
}

void Einsums_LinearAlgebra_vars::set_gemv_chunk(unsigned int chunk_dim) {
    std::lock_guard lock(*this);

    gemv_chunk = chunk_dim;
}

void Einsums_LinearAlgebra_vars::set_ger_chunk(unsigned int chunk_dim) {
    std::lock_guard lock(*this);

    ger_chunk = chunk_dim;
}

} // namespace einsums::detail