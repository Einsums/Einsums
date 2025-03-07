/*
  Copyright 2018 Paul Springer
  
  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  
  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  
  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  
  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
  
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * @author: Paul Springer (springer@aices.rwth-aachen.de)
 */

#pragma once

#include <Einsums/HPTT/HPTTTypes.hpp>

#include <cstdint>
#include <iostream>
#include <list>
#include <vector>

namespace hptt {

/**
 * Find the conjugate of the given type.
 */
template <typename floatType>
static floatType conj(floatType x) {
    return std::conj(x);
}

/// @copydoc conj<floatType>(floatType)
template <>
float conj(float x) {
    return x;
}

/// @copydoc conj<floatType>(floatType)
template <>
double conj(double x) {
    return x;
}

/**
 * Get the value that we consider to be zero.
 */
template <typename floatType>
static double getZeroThreshold();

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double getZeroThreshold<double>() {
    return 1e-16;
}

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double getZeroThreshold<DoubleComplex>() {
    return 1e-16;
}

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double getZeroThreshold<float>() {
    return 1e-6;
}

/// @copydoc getZeroThreshold<floatType>()
template <>
constexpr inline double getZeroThreshold<FloatComplex>() {
    return 1e-6;
}

/**
 * @todo Figure this out.
 */
void trashCache(double *A, double *B, int n);

/**
 * Check whether a vector contains an item.
 */
template <typename t>
int hasItem(std::vector<t> const &vec, t value) {
    return (std::find(vec.begin(), vec.end(), value) != vec.end());
}

/**
 * Print a vector to stdout.
 */
template <typename t>
void printVector(std::vector<t> const &vec, char const *label) {
    std::cout << label << ": ";
    for (auto a : vec)
        std::cout << a << ", ";
    std::cout << "\n";
}

/**
 * Print a list to stdout.
 */
template <typename t>
void printVector(std::list<t> const &vec, char const *label) {
    std::cout << label << ": ";
    for (auto a : vec)
        std::cout << a << ", ";
    std::cout << "\n";
}

/**
 * Gets the prime factors of a number. If zero or one is passed, the prime
 * factors output will be empty.
 *
 * @param n The number to factor.
 * @param primeFactors The list of factors.
 */
template <typename T>
void getPrimeFactors(T n, std::list<T> &primeFactors);

#ifndef DOXYGEN
template <>
void getPrimeFactors(std::uint8_t n, std::list<std::uint8_t> &primeFactors);
template <>
void getPrimeFactors(std::uint16_t n, std::list<std::uint16_t> &primeFactors);
template <>
void getPrimeFactors(std::uint32_t n, std::list<std::uint32_t> &primeFactors);

template <>
void getPrimeFactors(std::int8_t n, std::list<std::int8_t> &primeFactors);
template <>
void getPrimeFactors(std::int16_t n, std::list<std::int16_t> &primeFactors);
template <>
void getPrimeFactors(std::int32_t n, std::list<std::int32_t> &primeFactors);
#endif

/**
 * Find where a value is in an array.
 */
template <typename t>
int findPos(t value, std::vector<t> const &array) {
    for (int i = 0; i < array.size(); ++i)
        if (array[i] == value)
            return i;
    return -1;
}

/**
 * Find where a value is in an array.
 *
 * @param value The value to find.
 * @param array The array of values.
 * @param n The maximum index to check.
 *
 * @return The position of the value, or -1 if not found.
 */
int findPos(int value, int const *array, int n);

/**
 * Calculate the factorial of a number. Can only take factorials of numbers up to but not
 * including 21.
 *
 * @param n The number to take the factorial of.
 *
 * @return The factorial of that number.
 *
 * @throws std::overflow_error Throws this if the result would be too high to represent.
 */
std::uint64_t factorial(std::uint8_t n);

/**
 * Reorders parameters to swap between row-major and column-major forms.
 */
void accountForRowMajor(int const *sizeA, int const *outerSizeA, int const *outerSizeB, const int *offsetA, const int *offsetB, int const *perm,
                        int *tmpSizeA, int *tmpOuterSizeA, int *tmpouterSizeB, int *tmpOffsetA, int *tmpOffsetB, int *tmpPerm, int const dim, 
                        bool const useRowMajor);
} // namespace hptt
