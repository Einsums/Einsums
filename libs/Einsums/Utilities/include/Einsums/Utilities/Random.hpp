//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <numbers>
#include <random>

namespace einsums {

namespace detail {

/**
 * @struct circle_distribution
 *
 * @brief A uniformly random distribution on a circle with a center and radius.
 *
 * For real numbers, this will give a uniform distribution on an interval. For complex numbers,
 * the distribution will be such that the probability of a point being within a subregion
 * is proportional to the area of that subregion.
 */
template <typename T>
struct circle_distribution {};

#ifndef DOXYGEN
template <>
struct circle_distribution<float> {
  public:
    circle_distribution(float center, float radius) : mag_dist_(center - radius, center + radius) {}

    ~circle_distribution() = default;

    template <typename Generator>
    float operator()(Generator &generator) {
        return mag_dist_(generator);
    }

  private:
    std::uniform_real_distribution<float> mag_dist_;
};

template <>
struct circle_distribution<double> {
  public:
    circle_distribution(double center, double radius) : mag_dist_(center - radius, center + radius) {}

    ~circle_distribution() = default;

    template <typename Generator>
    double operator()(Generator &generator) {
        return mag_dist_(generator);
    }

  private:
    std::uniform_real_distribution<double> mag_dist_;
};

// For this case, we can just use the normal uniform distribution. The boundary of the region will not be included.
template <typename T>
struct circle_distribution<std::complex<T>> {
  public:
    circle_distribution(std::complex<T> center, T radius)
        : center_{center}, mag_dist_(0, radius), angle_dist_(0, 2 * std::numbers::pi_v<T>) {}

    ~circle_distribution() = default;

    template <typename Generator>
    std::complex<T> operator()(Generator &generator) {
        T mag = mag_dist_(generator), angle = angle_dist_(generator);

        return std::complex<T>{mag * std::cos(angle), mag * std::sin(angle)} + center_;
    }

  private:
    std::complex<T>                   center_;
    std::uniform_real_distribution<T> mag_dist_, angle_dist_;
};
#endif

/**
 * @struct unit_circle_distribution
 *
 * @brief A uniformly random distribution on the unit circle.
 *
 * For real numbers, this will give a uniform distribution on (-1, 1). The endpoints are not included.
 * For complex numbers, the distribution will be such that the probability of a point being within a subregion
 * is proportional to the area of that subregion. The region will be the unit disc without its boundary.
 */
template <typename T>
struct unit_circle_distribution {};

#ifndef DOXYGEN
template <>
struct unit_circle_distribution<float> {
  public:
    unit_circle_distribution() : mag_dist_(-1.0f, 1.0f) {}

    ~unit_circle_distribution() = default;

    template <typename Generator>
    float operator()(Generator &generator) {
        return mag_dist_(generator);
    }

  private:
    std::uniform_real_distribution<float> mag_dist_;
};

template <>
struct unit_circle_distribution<double> {
  public:
    unit_circle_distribution() : mag_dist_(-1.0, 1.0) {}

    ~unit_circle_distribution() = default;

    template <typename Generator>
    double operator()(Generator &generator) {
        return mag_dist_(generator);
    }

  private:
    std::uniform_real_distribution<double> mag_dist_;
};

// For this case, we can just use the normal uniform distribution. The boundary of the region will not be included.
template <typename T>
struct unit_circle_distribution<std::complex<T>> {
  public:
    unit_circle_distribution() : mag_dist_(0, 1), angle_dist_(0, 2 * std::numbers::pi_v<T>) {}

    ~unit_circle_distribution() = default;

    template <typename Generator>
    std::complex<T> operator()(Generator &generator) {
        T mag = mag_dist_(generator), angle = angle_dist_(generator);

        return std::complex<T>{std::cos(angle), std::sin(angle)};
    }

  private:
    std::uniform_real_distribution<T> mag_dist_, angle_dist_;
};
#endif

} // namespace detail

/**
 * @property random_engine
 *
 * @brief The global random engine for random number generation.
 */
EINSUMS_EXPORT extern std::default_random_engine random_engine;

/**
 * @brief Set the seed of the random number generator.
 *
 * @param seed The new seed for the random number generator.
 */
EINSUMS_EXPORT void seed_random(std::default_random_engine::result_type seed);

} // namespace einsums
