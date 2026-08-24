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

#include <Einsums/Config/StdAlternatives.hpp>

#include <fmt/format.h>

#include <array>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstring>

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

/*
 * To keep in line with the underlying library functions, we should never store 0 in errno.
 */

static constexpr std::array<char const *, 7> day_names{"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};

static constexpr std::array<char const *, 12> month_names{"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};

namespace einsums {

namespace detail {
[[nodiscard]] errno_t validate_timestruct(struct ::std::tm const *time_ptr) noexcept {
    if (time_ptr->tm_sec < 0 || time_ptr->tm_sec > 60) { // 60 because leap seconds are a thing.
        errno = EINVAL;
        return EINVAL;
    }

    if (time_ptr->tm_min < 0 || time_ptr->tm_min > 59) {
        errno = EINVAL;
        return EINVAL;
    }

    if (time_ptr->tm_hour < 0 || time_ptr->tm_hour > 23) {
        errno = EINVAL;
        return EINVAL;
    }

    if (time_ptr->tm_mday < 1 || time_ptr->tm_mday > 31) {
        errno = EINVAL;
        return EINVAL;
    }

    if (time_ptr->tm_mon < 0 || time_ptr->tm_mon > 11) {
        errno = EINVAL;
        return EINVAL;
    }

    // No validation for year. Can be positive or negative since it's years since 1900.

    if (time_ptr->tm_wday < 0 || time_ptr->tm_wday > 6) {
        errno = EINVAL;
        return EINVAL;
    }

    if (time_ptr->tm_yday < 0 || time_ptr->tm_yday > 365) {
        errno = EINVAL;
        return EINVAL;
    }

    return 0; // Success!
}

[[nodiscard]] errno_t asctime_convert(char *out_buffer, struct ::std::tm const *time_ptr) {
    errno_t prev_err = errno;
    errno            = 0;
    int ret =
        ::std::snprintf(out_buffer, 26, "%3s %3s %2d %.2d:%.2d:%.2d %4d\n", day_names[time_ptr->tm_wday], month_names[time_ptr->tm_mon],
                        time_ptr->tm_mday, time_ptr->tm_hour, time_ptr->tm_min, time_ptr->tm_sec, time_ptr->tm_year + 1900);

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}
} // namespace detail

#ifndef EINSUMS_WINDOWS
[[nodiscard]] errno_t asctime_s(char *out_buffer, ::std::size_t number_of_elements, struct ::std::tm const *time_ptr) {
    if (out_buffer == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (number_of_elements == 0) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if ((0 < number_of_elements && number_of_elements < 26) || time_ptr == nullptr) {
        ::std::memset(out_buffer, 0, number_of_elements);
        errno = EINVAL;
        std::raise(SIGSEGV);

        return EINVAL;
    }

    errno_t validation = detail::validate_timestruct(time_ptr);

    if (validation != 0) {
        ::std::memset(out_buffer, 0, number_of_elements);
        errno = validation;
        return validation;
    }

    errno_t out = detail::asctime_convert(out_buffer, time_ptr);

    return out;
}

[[nodiscard]] void *bsearch_s(void const *key, void const *base, ::std::size_t number, ::std::size_t width, safe_compare compare,
                              void *context) {
    // Essentially CS 101 binary search. It's not that hard to implement.
    if (number == 0) {
        return nullptr;
    }

    if (number == 1) {
        int low_compare = compare(context, key, base);

        if (low_compare == 0) {
            return const_cast<void *>(base);
        } else {
            return nullptr;
        }
    }

    ::std::size_t lower_bound = 0, upper_bound = number - 1, midpoint;

    char const *char_base = reinterpret_cast<char const *>(base);

    // Check the endpoints and exit early if they don't work.

    {
        int low_compare = compare(context, key, base);
        if (low_compare == 0) {
            return const_cast<void *>(base);
        }

        if (low_compare < 0) {
            return nullptr;
        }
    }

    {
        int high_compare = compare(context, reinterpret_cast<void const *>(char_base + (number - 1) * width), key);

        if (high_compare == 0) {
            return const_cast<void *>(reinterpret_cast<void const *>(char_base + (number - 1) * width));
        }

        if (high_compare < 0) {
            return nullptr;
        }
    }

    while (upper_bound - lower_bound > 1) {
        midpoint = (upper_bound + lower_bound) / 2;

        void const *curr = reinterpret_cast<void const *>(char_base + midpoint * width);

        int mid_compare = compare(context, key, curr);

        if (mid_compare == 0) {
            return const_cast<void *>(curr);
        } else if (mid_compare < 0) {
            upper_bound = midpoint;
        } else {
            lower_bound = midpoint;
        }
    }

    int low_compare = compare(context, key, reinterpret_cast<void const *>(char_base + lower_bound * width));
    if (low_compare == 0) {
        return const_cast<void *>(reinterpret_cast<void const *>(char_base + lower_bound * width));
    } else {
        int high_compare = compare(context, key, reinterpret_cast<void const *>(char_base + upper_bound * width));
        if (high_compare == 0) {
            return const_cast<void *>(reinterpret_cast<void const *>(char_base + upper_bound * width));
        }
    }
    return nullptr;
}

[[nodiscard]] errno_t clearerr_s(::std::FILE *fp) {
    if (fp == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    errno_t prev_err = errno;
    errno            = 0;

    std::clearerr(fp);

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t fopen_s(std::FILE **fp, char const *filename, char const *mode) {
    if (fp == nullptr) {
        errno = EINVAL;
        // Throw a segfault.
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (filename == nullptr) {
        errno = EINVAL;
        // Throw a segfault.
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (mode == nullptr) {
        errno = EINVAL;
        // Throw a segfault.
        std::raise(SIGSEGV);
        return EINVAL;
    }

    errno_t prev_err = errno;
    errno            = 0;

    *fp = std::fopen(filename, mode);

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(printf, 2, 3) int fprintf_s(std::FILE *fp, char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = std::vfprintf(fp, format, args);

    va_end(args);

    return out;
}

[[nodiscard]] std::size_t fread_s(void *buffer, std::size_t buffer_size, std::size_t element_size, std::size_t count, std::FILE *fp) {
    if (buffer_size == 0 || count == 0 || element_size == 0) {
        return 0;
    }

    std::size_t actual_count = count;

    if (count * element_size > buffer_size) {
        actual_count = buffer_size / element_size;
    }

    if (actual_count == 0) {
        return 0;
    }

    if (buffer == nullptr || fp == nullptr) {
        errno = EINVAL;
        return 0;
    }

    return std::fread(buffer, element_size, actual_count, fp);
}

[[nodiscard]] errno_t freopen_s(std::FILE **fp, char const *filename, char const *mode, std::FILE *old_fp) {
    if (old_fp == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (fp == nullptr) {
        // Setting this to zero is fine here since it gets overwritten in both code paths that follow.
        errno = 0;
        std::fclose(old_fp);

        if (errno != 0) {
            return errno;
        } else {
            errno = EINVAL;
            return EINVAL;
        }
    }

    if (filename == nullptr || mode == nullptr) {
        errno = 0;
        std::fclose(old_fp);

        if (errno != 0) {
            return errno;
        } else {
            errno = EINVAL;
            return EINVAL;
        }
    }

    errno_t prev_errno = errno;
    errno              = 0;
    *fp                = std::freopen(filename, mode, old_fp);

    if (errno == 0) {
        errno = prev_errno;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 3) int fscanf_s(std::FILE *fp, char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = std::vfscanf(fp, format, args);

    va_end(args);

    return out;
}

[[nodiscard]] errno_t getenv_s(std::size_t *needed_size, char *buffer, std::size_t buffer_size, char const *var_name) {
    if (needed_size == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (buffer == nullptr) {
        if (buffer_size > 0) {
            errno = EINVAL;
            std::raise(SIGSEGV);
            return EINVAL;
        }
    }

    if (var_name == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    errno_t prev_err = errno;
    errno            = 0;

    char *env_var = std::getenv(var_name);

    if (env_var == nullptr) {
        *needed_size = 0;

        if (errno == 0) {
            errno = prev_err;
            return 0;
        } else {
            return errno;
        }
    }

    *needed_size = std::strlen(env_var) + 1;

    if (buffer != nullptr && buffer_size < *needed_size) {
        errno = ERANGE;
        return ERANGE;
    } else if (buffer != nullptr) {
        std::strncpy(buffer, env_var, *needed_size);
    }

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t gmtime_s(struct std::tm *tm_out, std::time_t const *time) {
    if (tm_out == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (time == nullptr || *time < 0) {
        *tm_out = {.tm_sec   = -1,
                   .tm_min   = -1,
                   .tm_hour  = -1,
                   .tm_mday  = -1,
                   .tm_mon   = -1,
                   .tm_year  = -1,
                   .tm_wday  = -1,
                   .tm_yday  = -1,
                   .tm_isdst = -1};

#    ifdef __GNU__
        tm_out->tm_gmtoff = -1;
        tm_out->tm_zone   = nullptr;
#    endif

        errno = EINVAL;
        if(time == nullptr) {
            std::raise(SIGSEGV);
        }
        return EINVAL;
    }

    struct std::tm *temp = std::gmtime(time);

    errno_t prev_err = errno;
    errno            = 0;

    std::memcpy(tm_out, temp, sizeof(struct std::tm));

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t localtime_s(struct std::tm *tm_out, std::time_t const *time) {
    if (tm_out == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (time == nullptr) {
        *tm_out = {.tm_sec   = -1,
                   .tm_min   = -1,
                   .tm_hour  = -1,
                   .tm_mday  = -1,
                   .tm_mon   = -1,
                   .tm_year  = -1,
                   .tm_wday  = -1,
                   .tm_yday  = -1,
                   .tm_isdst = -1};

#    ifdef __GNU__
        tm_out->tm_gmtoff = -1;
        tm_out->tm_zone   = nullptr;
#    endif

        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    // This is allowed since the tm object is internal and handled by the C runtime, not allocated on the heap.
    errno_t prev_err = errno;
    errno            = 0;
    *tm_out          = *std::localtime(time);

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t memcpy_s(void *dest, std::size_t dest_size, void const *src, std::size_t count) {
    if (count != 0 && (dest == nullptr || src == nullptr)) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (dest_size < count) {
        errno = ERANGE;
        return ERANGE;
    }

    errno_t prev_errno = errno;
    errno              = 0;

    std::memcpy(dest, src, count);

    if (errno == 0) {
        errno = prev_errno;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t memmove_s(void *dest, std::size_t dest_size, void const *src, std::size_t count) {
    if (count != 0 && (dest == nullptr || src == nullptr)) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (dest_size < count) {
        errno = ERANGE;
        return ERANGE;
    }

    errno_t prev_errno = errno;
    errno              = 0;

    std::memmove(dest, src, count);

    if (errno == 0) {
        errno = prev_errno;
        return 0;
    } else {
        return errno;
    }
}

EINSUMS_CHECK_FORMAT(printf, 1, 2) int printf_s(char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = std::vprintf(format, args);

    va_end(args);

    return out;
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 1, 2) int scanf_s(char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = std::vscanf(format, args);

    va_end(args);

    return out;
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 3) int sscanf_s(char const *buffer, char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = std::vsscanf(buffer, format, args);

    va_end(args);

    return out;
}

[[nodiscard]] errno_t strcat_s(char *dest, std::size_t dest_size, char const *src) {
    if (dest == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (src == nullptr) {
        dest[0] = 0; // I don't know why Windows does this. It would be better to leave the string unmodified.
        errno   = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (dest_size == 0 || dest_size < std::strlen(dest) + std::strlen(src) + 1) {
        dest[0] = 0; // Again, it would be better to leave the string unmodified, but Windows has to do something weird.
        errno   = ERANGE;
        return ERANGE;
    }

    errno_t prev_errno = errno;
    errno              = 0;

    std::strcat(dest, src);

    if (errno == 0) {
        errno = prev_errno;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t strcpy_s(char *dest, std::size_t dest_size, char const *src) {
    if (dest == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return errno;
    }

    if (src == nullptr) {
        dest[0] = 0;
        errno   = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (dest_size == 0 || dest_size < std::strlen(src)) {
        dest[0] = 0;
        errno   = ERANGE;
        return ERANGE;
    }

    errno_t prev_errno = errno;
    errno              = 0;

    std::strcpy(dest, src);

    if (errno == 0) {
        errno = prev_errno;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t strerror_s(char *buffer, std::size_t buff_size, errno_t error_code) {
    if (buffer == nullptr) {
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (buff_size == 0) {
        return ERANGE;
    }

    errno_t prev_err = errno;
    errno            = 0;

    char const *error_str = std::strerror(error_code);

    // strncpy doesn't do the same thing that we need. We need simple truncation, strncpy backfills with zeros.
    for (std::size_t i = 0; i < buff_size; i++) {

        buffer[i] = error_str[i];
        if (i == buff_size - 1) {
            buffer[i] = 0;
        }

        if (buffer[i] == 0) {
            break;
        }
    }

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t strncat_s(char *dest, std::size_t dest_size, char const *src, std::size_t count) {
    if (dest == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (src == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (dest_size == 0 || dest_size < std::strlen(dest) + count) {
        errno = ERANGE;
        return ERANGE;
    }

    errno_t prev_errno = errno;
    errno              = 0;

    std::strncat(dest, src, count);

    if (errno == 0) {
        errno = prev_errno;
        return 0;
    } else {
        return errno;
    }
}

[[nodiscard]] errno_t strncpy_s(char *dest, std::size_t dest_size, char const *src, std::size_t count) {
    if (dest == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return errno;
    }

    if (src == nullptr) {
        dest[0] = 0;
        errno   = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    if (dest_size == 0 || dest_size < count) {
        dest[0] = 0;
        errno   = ERANGE;
        return ERANGE;
    }

    errno_t prev_errno = errno;
    errno              = 0;

    std::strncpy(dest, src, count);

    if (errno == 0) {
        errno = prev_errno;
        return 0;
    } else {
        return errno;
    }
}

// We'll need to write our own.
[[nodiscard]] char *strtok_s(char *str, char const *delimiters, StrtokContext *context) {
    if (context == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return nullptr;
    }

    if (str == nullptr && *context == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return nullptr;
    }

    if (delimiters == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return nullptr;
    }

    if (str != nullptr) {
        *context = str;
    }

    char *out = static_cast<char *>(*context);

    if (out == nullptr) {
        return out;
    }

    std::size_t prefix = std::strspn(static_cast<char *>(*context), delimiters);

    *context += prefix;

    if (**context == 0) {
        return nullptr;
    }

    out = static_cast<char *>(*context);

    std::size_t tok_len = std::strcspn(static_cast<char *>(*context), delimiters);

    *context += tok_len;

    if (**context != 0) {
        **context = 0;
        (*context)++;
    }

    return out;
}

[[nodiscard]] errno_t tmpfile_s(std::FILE **fp) {
    if (fp == nullptr) {
        errno = EINVAL;
        std::raise(SIGSEGV);
        return EINVAL;
    }

    errno_t prev_err = errno;
    errno            = 0;

    *fp = std::tmpfile();

    if (errno == 0) {
        errno = prev_err;
        return 0;
    } else {
        return errno;
    }
}

#else

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

[[nodiscard]] EINSUMS_CHECK_FORMAT(printf, 2, 3) int fprintf_s(std::FILE *fp, char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = ::vfprintf_s(fp, format, args);

    va_end(args);

    return out;
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 3) int fscanf_s(std::FILE *fp, char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = ::vfscanf_s(fp, format, args);

    va_end(args);

    return out;
}

EINSUMS_CHECK_FORMAT(printf, 1, 2) int printf_s(char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = ::vprintf_s(format, args);

    va_end(args);

    return out;
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 1, 2) int scanf_s(char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = ::vscanf_s(format, args);

    va_end(args);

    return out;
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 3) int sscanf_s(char const *buffer, char const *format, ...) {
    std::va_list args;

    va_start(args, format);

    int out = ::vsscanf_s(buffer, format, args);

    va_end(args);

    return out;
}

#endif

} // namespace einsums
