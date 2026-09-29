//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#ifdef EINSUMS_WINDOWS
// This needs to be before everything.

// #    ifdef _M_AMD64
// #        define _AMD64_
// #    elif defined(_M_ARM)
// #        define _ARM_
// #    endif

#    include <Windows.h>
#    include <basetsd.h>
#    include <windef.h>
#    include <winnt.h>
#endif

#include <Einsums/Config/PosixAlternatives.hpp>

#include <fmt/format.h>

#include <array>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace einsums {

#ifdef EINSUMS_WINDOWS

#    ifndef EINSUMS_WINDOWS_HAS_TYPES
#        include "windows_types.h"

extern "C" BOOL WINAPI CloseHandle(HANDLE hObject);

#    endif

// #    include <errhandlingapi.h>
// #    include <handleapi.h>
#    include <stdexcept>
#    include <tlhelp32.h>
#endif

#ifdef EINSUMS_WINDOWS

[[nodiscard]] int getppid() {
    int pid = _getpid();

    HANDLE         snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    PROCESSENTRY32 process_entry;

    // Error checking.
    if (snapshot == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Einsums: Couldn't get process handle for getppid");
    }

    // Get the first process entry.
    if (!Process32First(snapshot, &process_entry)) {
        throw std::runtime_error("Einsums: Couldn't get first process entry for getppid");
    }

    do {
        if (process_entry.th32ProcessID == pid) {
            CloseHandle(snapshot);
            return process_entry.th32ParentProcessID;
        }
    } while (Process32Next(snapshot, &process_entry));

    CloseHandle(snapshot);

    throw std::runtime_error(
        fmt::format("Einsums: Couldn't find a process with PID that matches {} (current PID), so no parent was found.", pid));
}

#endif

} // namespace einsums
