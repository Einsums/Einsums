//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>
#include <string>

#if defined(EINSUMS_HAVE_BACKTRACES)

#include <cpptrace/cpptrace.hpp>

namespace einsums::util {

/**
 * @brief Generate a backtrace.
 */
EINSUMS_EXPORT std::string backtrace(std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH);

template <typename Ostream>
void print_backtrace(Ostream &stream, std::size_t frames_no = EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH) {
    auto trace = cpptrace::generate_trace(1, frames_no);

    stream << trace;
}

} // namespace einsums::util

#else

namespace einsums::util {

/**
 * @brief Generate a backtrace.
 */
inline std::string backtrace(std::size_t frames_no = 0) {
    return "";
}

template <typename Ostream>
void print_backtrace(Ostream &stream, std::size_t frames_no = 0) {
    ; // Noop
}

} // namespace einsums::util

#endif
