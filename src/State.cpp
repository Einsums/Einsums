//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/State.hpp"

namespace einsums::state {
namespace {
h5::fd_t s_Data;
}

EINSUMS_EXPORT h5::fd_t &data() {
    return s_Data;
}

} // namespace einsums::state
