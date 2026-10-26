//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Tensor/ModuleVars.hpp>

namespace einsums::detail {

EINSUMS_SINGLETON_IMPL(Einsums_Tensor_vars)

Einsums_Tensor_vars::~Einsums_Tensor_vars() {
    std::fprintf(stderr, "Destroying the Einsums_Tensor_vars.\n");
}
} // namespace einsums::detail