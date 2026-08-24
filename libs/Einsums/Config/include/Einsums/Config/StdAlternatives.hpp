//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/*
 * Windows doesn't like certain library functions. We want to use library functions. This header provides wrappers for library functions
 * that don't throw errors on Windows.
 */

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/format.h>
#include <fmt/xchar.h>

#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <string>
#include <type_traits>

#ifdef EINSUMS_WINDOWS
#    include <process.h>
#    include <stdlib.h>
#else
#    include <unistd.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define EINSUMS_CHECK_FORMAT(archetype, string_idx, first_arg) [[gnu::format(archetype, string_idx, first_arg)]]
#else
#    define EINSUMS_CHECK_FORMAT(archetype, string_idx, first_arg)
#endif

namespace einsums {
namespace detail {

// fmtlib doesn't have formatted_size for styled text. I'll just make my own.
template <typename... T>
FMT_NODISCARD FMT_INLINE auto formatted_size(fmt::text_style ts, fmt::format_string<T...> fmt, T &&...args) -> size_t {
    auto buf = fmt::detail::counting_buffer<>();
    fmt::detail::vformat_to(buf, ts, fmt.str, fmt::vargs<T...>{{args...}});
    return buf.count();
}

// Windows thinks fmtlib does heap bashing. I don't think it does, but it still causes segfaults.
template <typename... Args>
inline std::basic_string<char> corrected_format(std::basic_string_view<char> const &format, Args &&...args) {
    std::basic_string<char> out;

    auto runtime_format = fmt::runtime(format);

    size_t out_size = fmt::formatted_size(runtime_format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), runtime_format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
inline std::basic_string<wchar_t> corrected_format(std::basic_string_view<wchar_t> const &format, Args &&...args) {
    std::basic_string<wchar_t> out;

    auto runtime_format = fmt::runtime(format);

    size_t out_size = fmt::formatted_size(runtime_format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), runtime_format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
std::basic_string<char> corrected_format(fmt::text_style style, std::basic_string_view<char> const &format, Args &&...args) {
    std::basic_string<char> out;

    auto runtime_format = fmt::runtime(format);

    size_t out_size = detail::formatted_size(style, runtime_format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), style, runtime_format, std::forward<Args>(args)...);

    return out;
}

template <typename... Args>
std::basic_string<wchar_t> corrected_format(fmt::text_style style, std::basic_string_view<wchar_t> const &format, Args &&...args) {
    std::basic_string<wchar_t> out;

    auto runtime_format = fmt::runtime(format);

    size_t out_size = detail::formatted_size(style, runtime_format, std::forward<Args>(args)...);

    out.resize(out_size);

    fmt::format_to(out.begin(), style, runtime_format, std::forward<Args>(args)...);

    return out;
}
} // namespace detail

using StrtokContext = char *;

namespace detail {

template <typename T, typename Else, typename = void>
struct type_or_else_if_defined {
    using type = Else;
};

template <typename T, typename Else>
struct type_or_else_if_defined<T, Else, std::void_t<decltype(sizeof(T))>> {
    using type = T;
};

template <typename T, typename Else>
using type_or_else_if_defined_t = type_or_else_if_defined<T, Else>::type;
} // namespace detail

#ifndef EINSUMS_WINDOWS
using errno_t      = int;
using safe_compare = int (*)(void *context, void const *key, void const *datum);
#else
using safe_compare = int(__cdecl *)(void *context, void const *key, void const *datum);
#endif

// Just going in alphabetical order, looking for things with an _s in the Windows C runtime.

namespace detail {
[[nodiscard]] EINSUMS_EXPORT errno_t validate_timestruct(struct ::std::tm const *time_ptr) noexcept;

// The actual asctime is really bad. Write our own since it's probably going to be deprecated and it's not too hard.
// Dealing with locales would be harder.
// Also, this does no validation. Validation is done by the callers.
[[nodiscard]] EINSUMS_EXPORT errno_t asctime_convert(char *out_buffer, struct ::std::tm const *time_ptr);

} // namespace detail

// Yes, these are nodiscard. Everyone always forgets that these functions have return values that need to be checked.
// Except printf_s. Really, what are our options if printf fails? If that's the case something has gone really wrong.
[[nodiscard]] EINSUMS_CHECK_FORMAT(printf, 2, 3) EINSUMS_EXPORT int fprintf_s(std::FILE *fp, char const *format, ...);

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 3) EINSUMS_EXPORT int fscanf_s(std::FILE *fp, char const *format, ...);

EINSUMS_CHECK_FORMAT(printf, 1, 2) EINSUMS_EXPORT int printf_s(char const *format, ...);

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 1, 2) EINSUMS_EXPORT int scanf_s(char const *format, ...);

// No sprintf. We shouldn't be using it anyways. No snprintf since it's already safe. The _snprintf_s function is considered optional, it
// seems. This is due to the fact that _snprintf isn't standards conformant. Standards-conformant snprintf is essentially the same as
// _snprintf_s.

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 3) EINSUMS_EXPORT int sscanf_s(char const *buffer, char const *format, ...);

#ifdef EINSUMS_WINDOWS
[[nodiscard]] inline errno_t asctime_s(char *out_buffer, ::std::size_t number_of_elements, struct ::std::tm const *time_ptr) noexcept {
    return ::asctime_s(out_buffer, number_of_elements, time_ptr);
}

[[nodiscard]] inline void *bsearch_s(void const *key, void const *base, ::std::size_t number, ::std::size_t width, safe_compare compare,
                                     void *context) {
    return ::bsearch_s(key, base, number, width, compare, context);
}

[[nodiscard]] inline errno_t clearerr_s(::std::FILE *stream) {
    return ::clearerr_s(stream);
}

[[nodiscard]] inline errno_t ctime_s(char *buffer, ::std::size_t size, ::std::time_t const *source_time) {
    return ::ctime_s(buffer, size, source_time);
}

[[nodiscard]] inline errno_t fopen_s(std::FILE **fp, char const *filename, char const *mode) {
    return ::fopen_s(fp, filename, mode);
}

[[nodiscard]] inline std::size_t fread_s(void *buffer, std::size_t buffer_size, std::size_t element_size, std::size_t count,
                                         std::FILE *fp) {
    return ::fread_s(buffer, buffer_size, element_size, count, fp);
}

[[nodiscard]] inline errno_t freopen_s(std::FILE **stream, char const *file_name, char const *mode, std::FILE *old_fp) {
    return ::freopen_s(stream, file_name, mode, old_fp);
}

[[nodiscard]] inline errno_t getenv_s(std::size_t *needed_size, char *buffer, std::size_t buffer_size, char const *var_name) {
    return ::getenv_s(needed_size, buffer, buffer_size, var_name);
}

// The Windows standard says that buffer size should be size_t. The C++ standard says that fgets takes int.
// Prefer int in this case.
[[nodiscard]] inline char *gets_s(char *buffer, int buffer_size) {
    return ::gets_s(buffer, buffer_size);
}

[[nodiscard]] inline errno_t putenv_s(char const *var_name, char const *value) {
    return ::_putenv_s(var_name, value);
}

[[nodiscard]] inline int getpid() {
    return ::_getpid();
}

[[nodiscard]] int EINSUMS_EXPORT getppid();

[[nodiscard]] inline errno_t gmtime_s(struct std::tm *tm_out, std::time_t const *time) {
    return ::gmtime_s(tm_out, time);
}

[[nodiscard]] inline errno_t localtime_s(struct std::tm *tm_out, std::time_t const *time) {
    return ::localtime_s(tm_out, time);
}

[[nodiscard]] inline errno_t mbsrtowcs_s(std::size_t *ret_val, wchar_t *dst, std::size_t dest_size, char const **src, std::size_t count,
                                         std::mbstate_t *state) {
    return ::mbsrtowcs_s(ret_val, dst, dest_size, src, count, state);
}

[[nodiscard]] inline errno_t memcpy_s(void *dest, std::size_t dest_size, void const *src, std::size_t count) {
    return ::memcpy_s(dest, dest_size, src, count);
}

[[nodiscard]] inline errno_t memmove_s(void *dest, std::size_t dest_size, void const *src, std::size_t count) {
    return ::memmove_s(dest, dest_size, src, count);
}

inline void qsort_s(void *base, std::size_t elements, std::size_t width, safe_compare compare, void *context) {
    return ::qsort_s(base, elements, width, compare, context);
}

[[nodiscard]] inline errno_t strcat_s(char *dest, std::size_t dest_size, char const *src) {
    return ::strcat_s(dest, dest_size, src);
}

[[nodiscard]] inline errno_t strcpy_s(char *dest, std::size_t dest_size, char const *src) {
    return ::strcpy_s(dest, dest_size, src);
}

[[nodiscard]] inline errno_t strerror_s(char *buffer, std::size_t buff_size, errno_t error_code) {
    return ::strerror_s(buffer, buff_size, error_code);
}

[[nodiscard]] inline errno_t strncat_s(char *dest, std::size_t dest_size, char const *src, std::size_t count) {
    return ::strncat_s(dest, dest_size, src, count);
}

[[nodiscard]] inline errno_t strncpy_s(char *dest, std::size_t dest_size, char const *src, std::size_t count) {
    return ::strncpy_s(dest, dest_size, src, count);
}

[[nodiscard]] inline char *strtok_s(char *str, char const *delimiters, StrtokContext *context) {
    return ::strtok_s(str, delimiters, static_cast<char **>(context));
}

[[nodiscard]] inline errno_t tmpfile_s(std::FILE **fp) {
    return ::tmpfile_s(fp);
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(printf, 2, 0) inline int vfprintf_s(std::FILE *fp, char const *format, std::va_list args) {
    return ::vfprintf_s(fp, format, args);
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 0) inline int vfscanf_s(std::FILE *fp, char const *format, std::va_list args) {
    return ::vfscanf_s(fp, format, args);
}

EINSUMS_CHECK_FORMAT(printf, 1, 0) inline int vprintf_s(char const *format, std::va_list args) {
    return ::vprintf_s(format, args);
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 1, 0) inline int vscanf_s(char const *format, std::va_list args) {
    return ::vscanf_s(format, args);
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 0) inline int vsscanf_s(char const *buffer, char const *format, std::va_list args) {
    return ::vsscanf_s(buffer, format, args);
}

#else
[[nodiscard]] EINSUMS_EXPORT errno_t asctime_s(char *out_buffer, ::std::size_t number_of_elements, struct ::std::tm const *time_ptr);

[[nodiscard]] EINSUMS_EXPORT void *bsearch_s(void const *key, void const *base, ::std::size_t number, ::std::size_t width,
                                             safe_compare compare, void *context);

[[nodiscard]] EINSUMS_EXPORT errno_t clearerr_s(::std::FILE *stream);

[[nodiscard]] inline errno_t ctime_s(char *out_buffer, ::std::size_t size, ::std::time_t const *source_time) {
    return asctime_s(out_buffer, size, std::localtime(source_time));
}

[[nodiscard]] EINSUMS_EXPORT errno_t fopen_s(std::FILE **fp, char const *filename, char const *mode);

[[nodiscard]] EINSUMS_EXPORT std::size_t fread_s(void *buffer, std::size_t buffer_size, std::size_t element_size, std::size_t count,
                                                 std::FILE *fp);

[[nodiscard]] EINSUMS_EXPORT errno_t freopen_s(std::FILE **fp, char const *filename, char const *mode, std::FILE *old_fp);

[[nodiscard]] EINSUMS_EXPORT errno_t getenv_s(std::size_t *needed_size, char *buffer, std::size_t buffer_size, char const *var_name);

[[nodiscard]] inline char *gets_s(char *buffer, int buffer_size) {
    return std::fgets(buffer, buffer_size, stdin);
}

[[nodiscard]] inline int getpid() {
    return ::getpid();
}

[[nodiscard]] inline int getppid() {
    return ::getppid();
}

[[nodiscard]] EINSUMS_EXPORT errno_t gmtime_s(struct std::tm *tm_out, std::time_t const *time);

[[nodiscard]] EINSUMS_EXPORT errno_t localtime_s(struct std::tm *tm_out, std::time_t const *time);

[[nodiscard]] EINSUMS_EXPORT errno_t memcpy_s(void *dest, std::size_t dest_size, void const *src, std::size_t count);

[[nodiscard]] EINSUMS_EXPORT errno_t memmove_s(void *dest, std::size_t dest_size, void const *src, std::size_t count);

EINSUMS_EXPORT void qsort_s(void *base, std::size_t elements, std::size_t width, safe_compare compare, void *context);

[[nodiscard]] EINSUMS_EXPORT errno_t strcat_s(char *dest, std::size_t dest_size, char const *src);

[[nodiscard]] EINSUMS_EXPORT errno_t strcpy_s(char *dest, std::size_t dest_size, char const *src);

[[nodiscard]] EINSUMS_EXPORT errno_t strerror_s(char *buffer, std::size_t buff_size, errno_t error_code);

[[nodiscard]] EINSUMS_EXPORT errno_t strncat_s(char *dest, std::size_t dest_size, char const *src, std::size_t count);

[[nodiscard]] EINSUMS_EXPORT errno_t strncpy_s(char *dest, std::size_t dest_size, char const *src, std::size_t count);

[[nodiscard]] EINSUMS_EXPORT char *strtok_s(char *str, char const *delimiters, StrtokContext *context);

[[nodiscard]] EINSUMS_EXPORT errno_t tmpfile_s(std::FILE **fp);

[[nodiscard]] EINSUMS_CHECK_FORMAT(printf, 2, 0) inline int vfprintf_s(std::FILE *fp, char const *format, std::va_list args) {
    return std::vfprintf(fp, format, args);
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 0) inline int vfscanf_s(std::FILE *fp, char const *format, std::va_list args) {
    return std::vfscanf(fp, format, args);
}

EINSUMS_CHECK_FORMAT(printf, 1, 0) inline int vprintf_s(char const *format, std::va_list args) {
    return std::vprintf(format, args);
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 1, 0) inline int vscanf_s(char const *format, std::va_list args) {
    return std::vscanf(format, args);
}

[[nodiscard]] EINSUMS_CHECK_FORMAT(scanf, 2, 0) inline int vsscanf_s(char const *buffer, char const *format, std::va_list args) {
    return std::vsscanf(buffer, format, args);
}

#endif

} // namespace einsums
