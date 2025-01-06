//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Debugging/Backtrace.hpp>

#if defined(EINSUMS_HAVE_BACKTRACES)
#    include <cpptrace/cpptrace.hpp>

namespace einsums::util {

std::string backtrace(std::size_t frames_no) {
    return cpptrace::generate_trace(1, frames_no).to_string(true);
}

} // namespace einsums::util
#endif
