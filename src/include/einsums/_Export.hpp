#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#define _EXPORT __declspec(dllexport)
#else
#define _EXPORT __attribute__((visibility("default")))
#endif

#if defined(EINSUMS_LIBRARY)
#define EINSUMS_EXPORT _EXPORT
#elif defined(EINSUMS_STATIC_LIBRARY)
#define EINSUMS_EXPORT
#else
#define EINSUMS_EXPORT _EXPORT
#endif
