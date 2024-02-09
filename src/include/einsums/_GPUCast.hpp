#pragma once

#include "einsums/_Common.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::gpu)

template<typename To, typename From>
struct HipCast {
};

template<>
struct HipCast<float, float> {
    __host__ __device__
    static inline float cast(float from) {
        return from;
    }
};

template<>
struct HipCast<double, float> {
    __host__ __device__
    static inline double cast(float from) {
        return (double) from;
    }
};

template<>
struct HipCast<std::complex<float>, float> {
    __host__ __device__
    static inline std::complex<float> cast(float from) {
        return std::complex<float>(from);
    }
};

template<>
struct HipCast<std::complex<double>, float> {
    __host__ __device__
    static inline std::complex<double> cast(float from) {
        return std::complex<double>((double) from);
    }
};

template<>
struct HipCast<hipComplex, float> {
    __host__ __device__
    static inline hipComplex cast(float from) {
        return make_hipComplex(from, 0.0f);
    }
};

template<>
struct HipCast<hipDoubleComplex, float> {
    __host__ __device__
    static inline hipDoubleComplex cast(float from) {
        return make_hipDoubleComplex((double) from, 0.0);
    }
};

template<>
struct HipCast<float, double> {
    __host__ __device__
    static inline float cast(double from) {
        return (float) from;
    }
};

template<>
struct HipCast<double, double> {
    __host__ __device__
    static inline double cast(double from) {
        return from;
    }
};

template<>
struct HipCast<std::complex<float>, double> {
    __host__ __device__
    static inline std::complex<float> cast(double from) {
        return std::complex<float>((float) from);
    }
};

template<>
struct HipCast<std::complex<double>, double> {
    __host__ __device__
    static inline std::complex<double> cast(double from) {
        return std::complex<double>(from);
    }
};

template<>
struct HipCast<hipComplex, double> {
    __host__ __device__
    static inline hipComplex cast(double from) {
        return make_hipComplex((float) from, 0.0f);
    }
};

template<>
struct HipCast<hipDoubleComplex, double> {
    __host__ __device__
    static inline hipDoubleComplex cast(double from) {
        return make_hipDoubleComplex(from, 0.0);
    }
};

template<>
struct HipCast<float, std::complex<float>> {
    __host__ __device__
    static inline float cast(std::complex<float> from) {
        return from.real();
    }
};

template<>
struct HipCast<double, std::complex<float>> {
    __host__ __device__
    static inline double cast(std::complex<float> from) {
        return (double) from.real();
    }
};

template<>
struct HipCast<std::complex<float>, std::complex<float>> {
    __host__ __device__
    static inline std::complex<float> cast(std::complex<float> from) {
        return from;
    }
};

template<>
struct HipCast<std::complex<double>, std::complex<float>> {
    __host__ __device__
    static inline std::complex<double> cast(std::complex<float> from) {
        return std::complex<double>((double) from.real(), (double) from.imag());
    }
};

template<>
struct HipCast<hipComplex, std::complex<float>> {
    __host__ __device__
    static inline hipComplex cast(std::complex<float> from) {
        return *reinterpret_cast<hipComplex *>(&from);
    }
};

template<>
struct HipCast<hipDoubleComplex, std::complex<float>> {
    __host__ __device__
    static inline hipDoubleComplex cast(std::complex<float> from) {
        return make_hipDoubleComplex((double) from.real(), (double) from.imag());
    }
};

template<>
struct HipCast<float, std::complex<double>> {
    __host__ __device__
    static inline float cast(std::complex<double> from) {
        return (float) from.real();
    }
};

template<>
struct HipCast<double, std::complex<double>> {
    __host__ __device__
    static inline double cast(std::complex<double> from) {
        return from.real();
    }
};

template<>
struct HipCast<std::complex<float>, std::complex<double>> {
    __host__ __device__
    static inline std::complex<float> cast(std::complex<double> from) {
        return std::complex<float>((float) from.real(), (float) from.imag());
    }
};

template<>
struct HipCast<std::complex<double>, std::complex<double>> {
    __host__ __device__
    static inline std::complex<double> cast(std::complex<double> from) {
        return from;
    }
};

template<>
struct HipCast<hipComplex, std::complex<double>> {
    __host__ __device__
    static inline hipComplex cast(std::complex<double> from) {
        return make_hipComplex((float) from.real(), (float) from.imag());
    }
};

template<>
struct HipCast<hipDoubleComplex, std::complex<double>> {
    __host__ __device__
    static inline hipDoubleComplex cast(std::complex<double> from) {
        return *reinterpret_cast<hipDoubleComplex *>(&from);
    }
};

template<>
struct HipCast<float, hipComplex> {
    __host__ __device__
    static inline float cast(hipComplex from) {
        return from.x;
    }
};

template<>
struct HipCast<double, hipComplex> {
    __host__ __device__
    static inline double cast(hipComplex from) {
        return (double) from.x;
    }
};

template<>
struct HipCast<std::complex<float>, hipComplex> {
    __host__ __device__
    static inline std::complex<float> cast(hipComplex from) {
        return *reinterpret_cast<std::complex<float> *>(&from);
    }
};

template<>
struct HipCast<std::complex<double>, hipComplex> {
    __host__ __device__
    static inline std::complex<double> cast(hipComplex from) {
        return std::complex<double>((double) from.x, (double) from.y);
    }
};

template<>
struct HipCast<hipComplex, hipComplex> {
    __host__ __device__
    static inline hipComplex cast(hipComplex from) {
        return from;
    }
};

template<>
struct HipCast<hipDoubleComplex, hipComplex> {
    __host__ __device__
    static inline hipDoubleComplex cast(hipComplex from) {
        return make_hipDoubleComplex((double) from.x, (double) from.x);
    }
};

template<>
struct HipCast<float, hipDoubleComplex> {
    __host__ __device__
    static inline float cast(hipDoubleComplex from) {
        return (float) from.x;
    }
};

template<>
struct HipCast<double, hipDoubleComplex> {
    __host__ __device__
    static inline double cast(hipDoubleComplex from) {
        return from.x;
    }
};

template<>
struct HipCast<std::complex<float>, hipDoubleComplex> {
    __host__ __device__
    static inline std::complex<float> cast(hipDoubleComplex from) {
        return std::complex<float>((float) from.x, (float) from.y);
    }
};

template<>
struct HipCast<std::complex<double>, hipDoubleComplex> {
    __host__ __device__
    static inline std::complex<double> cast(hipDoubleComplex from) {
        return *reinterpret_cast<std::complex<double> *>(&from);
    }
};

template<>
struct HipCast<hipComplex, hipDoubleComplex> {
    __host__ __device__
    static inline hipComplex cast(hipDoubleComplex from) {
        return make_hipComplex((float) from.x, (float) from.y);
    }
};

template<>
struct HipCast<hipDoubleComplex, hipDoubleComplex> {
    __host__ __device__
    static inline hipDoubleComplex cast(hipDoubleComplex from) {
        return from;
    }
};

END_EINSUMS_NAMESPACE_HPP(einsums::gpu)