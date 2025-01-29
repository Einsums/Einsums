//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>
#include <string>

#if defined(EINSUMS_HAVE_BACKTRACES)

namespace einsums::util {

EINSUMS_EXPORT std::string backtrace(std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH);

} // namespace einsums::util

#else

namespace einsums::util {

inline std::string backtrace(std::size_t frames_no = 0) {
    return "";
}

} // namespace einsums::util

#endif
