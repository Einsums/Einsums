//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>

#ifdef EINSUMS_WINDOWS
#    include <process.h>
#    include <stdlib.h>
#else
#    include <unistd.h>
#endif

namespace einsums {
#ifdef EINSUMS_WINDOWS

[[nodiscard]] inline int getpid() {
    return ::_getpid();
}

[[nodiscard]] int EINSUMS_EXPORT getppid();

// Because longs are 32 bit on Windows for whatever reason.
[[nodiscard]] inline int fseek(std::FILE *fp, std::ptrdiff_t offset, int origin) {
    return _fseeki64(fp, offset, origin);
}

[[nodiscard]] inline std::ptrdiff_t ftell(std::FILE *fp) {
    return _ftelli64(fp);
}

#else

[[nodiscard]] inline int getpid() {
    return ::getpid();
}

[[nodiscard]] inline int getppid() {
    return ::getppid();
}

[[nodiscard]] inline int fseek(std::FILE *fp, std::ptrdiff_t offset, int origin) {
    return std::fseek(fp, offset, origin);
}

[[nodiscard]] inline std::ptrdiff_t ftell(std::FILE *fp) {
    return std::ftell(fp);
}

#endif
} // namespace einsums
