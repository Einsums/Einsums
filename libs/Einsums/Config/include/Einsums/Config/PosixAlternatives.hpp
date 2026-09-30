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

#else

[[nodiscard]] inline int getpid() {
    return ::getpid();
}

[[nodiscard]] inline int getppid() {
    return ::getppid();
}

#endif
} // namespace einsums