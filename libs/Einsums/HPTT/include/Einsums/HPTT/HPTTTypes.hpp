#pragma once

#include <complex>
#include <complex.h>

#define REGISTER_BITS 256 // AVX
#ifdef __aarch64__
#undef REGISTER_BITS 
#define REGISTER_BITS 128 // ARM
#endif

namespace hptt {

/**
 * \brief Determines the duration of the auto-tuning process.
 *
 * * ESTIMATE: 0 seconds (i.e., no auto-tuning)
 * * MEASURE: 10 seconds
 * * PATIENT: 60 seconds
 * * CRAZY : 3600 seconds
 */
enum SelectionMethod { ESTIMATE, MEASURE, PATIENT, CRAZY };

/** 
 * @typedef FloatComplex
 *
 * @brief Alias for std::complex<float>.
 */
using FloatComplex = std::complex<float>;

/**
 * @typedef DoubleComplex
 *
 * @brief Alias for std::complex<double>.
 */
using DoubleComplex = std::complex<double>;

}

