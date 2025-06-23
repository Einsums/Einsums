//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

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

template <>
struct circle_distribution<float> {
  public:
    circle_distribution(float center, float radius)
        : center_{center}, radius_{radius}, mag_dist_(0, std::numeric_limits<uint32_t>::max()) {}

    ~circle_distribution() = default;

    template <typename Generator>
    float operator()(Generator &generator) {
        union {
            uint32_t integer;
            float    floating;
        } bitmanip;

        bitmanip.integer = mag_dist_(generator);

        // Clear the exponent.
        bitmanip.integer &= 0x807fffffU;

        // Set the exponent so that these numbers range from (-2,-1] and [1,2).
        bitmanip.integer |= 0x3f800000U;

        // Now, generate a second random number for backfilling.
        uint32_t backfill = mag_dist_(generator);

        // Compute the number of bits we will need.
        uint32_t temp = bitmanip.integer;
        // Clear the sign and exponent.
        temp &= 0x007fffff;
        // Now, count the number of leading zeros in the new value.
        int fill_bits = std::countl_zero(temp);
        // Shift a bitmask.
        uint32_t mask = 0x007fffffU >> (31 - fill_bits);

        // Now, mask the backfill.
        backfill &= mask;

        // Now, when we perform the subtraction, we can backfill these values in.
        if (bitmanip.floating < 0.0f) {
            bitmanip.floating += 1.0f;
        } else {
            bitmanip.floating -= 1.0f;
        }

        // Backfill. Use xor just to be fancy.
        if ((bitmanip.integer & 0x7fffffU) != 0) {
            // Backfill. Use xor just to be fancy.
            bitmanip.integer ^= backfill;
        }

        // Now, scale and recenter the value.
        return bitmanip.floating * radius_ + center_;
    }

  private:
    float                                   center_, radius_;
    std::uniform_int_distribution<uint32_t> mag_dist_;
};

template <>
struct circle_distribution<double> {
  public:
    circle_distribution(double center, double radius)
        : center_{center}, radius_{radius}, mag_dist_(0, std::numeric_limits<uint64_t>::max()) {}

    ~circle_distribution() = default;

    template <typename Generator>
    double operator()(Generator &generator) {
        union {
            uint64_t integer;
            double   floating;
        } bitmanip;

        bitmanip.integer = mag_dist_(generator);

        // Clear the exponent.
        bitmanip.integer &= 0x800fffffffffffffUL;
        // Set the exponent so that these numbers range from (-2,-1] and [1,2).
        bitmanip.integer |= 0x3ff0000000000000UL;

        // Now, generate a second random number for backfilling.
        uint64_t backfill = mag_dist_(generator);

        // Compute the number of bits we will need.
        uint64_t temp = bitmanip.integer;
        // Clear the sign and exponent.
        temp &= 0x000fffffffffffffUL;
        // Now, count the number of leading zeros in the new value.
        int fill_bits = std::countl_zero(temp);
        // Shift a bitmask.
        uint32_t mask = 0x000fffffffffffffUL >> (63 - fill_bits);
        // Now, mask the backfill.
        backfill &= mask;
        // Now, when we perform the subtraction, we can backfill these values in.
        if (bitmanip.floating < 0.0) {
            bitmanip.floating += 1.0;
        } else {
            bitmanip.floating -= 1.0;
        }

        if ((bitmanip.integer & 0x7fffffffffffffUL) != 0) {
            // Backfill. Use xor just to be fancy.
            bitmanip.integer ^= backfill;
        }

        // Now, scale and recenter the value.
        return bitmanip.floating * radius_ + center_;
    }

  private:
    double                                  center_, radius_;
    std::uniform_int_distribution<uint64_t> mag_dist_;
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

template <>
struct unit_circle_distribution<float> {
  public:
    unit_circle_distribution() : mag_dist_(0, std::numeric_limits<uint32_t>::max()) {}

    ~unit_circle_distribution() = default;

    template <typename Generator>
    float operator()(Generator &generator) {
        union {
            uint32_t integer;
            float    floating;
        } bitmanip;

        bitmanip.integer = mag_dist_(generator);

        // Clear the exponent.
        bitmanip.integer &= 0x807fffffU;

        // Set the exponent so that these numbers range from (-2,-1] and [1,2).
        bitmanip.integer |= 0x3f800000U;

        // Now, generate a second random number for backfilling.
        uint32_t backfill = mag_dist_(generator);

        // Compute the number of bits we will need.
        uint32_t temp = bitmanip.integer;
        // Clear the sign and exponent.
        temp &= 0x007fffff;
        // Now, count the number of leading zeros in the new value.
        int fill_bits = std::countl_zero(temp);
        // Shift a bitmask.
        uint32_t mask = 0x007fffffU >> (31 - fill_bits);

        // Now, mask the backfill.
        backfill &= mask;

        // Now, when we perform the subtraction, we can backfill these values in.
        if (bitmanip.floating < 0.0f) {
            bitmanip.floating += 1.0f;
        } else {
            bitmanip.floating -= 1.0f;
        }

        // Backfill. Use xor just to be fancy.
        if ((bitmanip.integer & 0x7fffffU) != 0) {
            // Backfill. Use xor just to be fancy.
            bitmanip.integer ^= backfill;
        }

        // Now, scale and recenter the value.
        return bitmanip.floating;
    }

  private:
    std::uniform_int_distribution<uint32_t> mag_dist_;
};

template <>
struct unit_circle_distribution<double> {
  public:
    unit_circle_distribution() : mag_dist_(0, std::numeric_limits<uint64_t>::max()) {}

    ~unit_circle_distribution() = default;

    template <typename Generator>
    double operator()(Generator &generator) {
        union {
            uint64_t integer;
            double   floating;
        } bitmanip;

        bitmanip.integer = mag_dist_(generator);

        // Clear the exponent.
        bitmanip.integer &= 0x800fffffffffffffUL;
        // Set the exponent so that these numbers range from (-2,-1] and [1,2).
        bitmanip.integer |= 0x3ff0000000000000UL;

        // Now, generate a second random number for backfilling.
        uint64_t backfill = mag_dist_(generator);

        // Compute the number of bits we will need.
        uint64_t temp = bitmanip.integer;
        // Clear the sign and exponent.
        temp &= 0x000fffffffffffffUL;
        // Now, count the number of leading zeros in the new value.
        int fill_bits = std::countl_zero(temp);
        // Shift a bitmask.
        uint32_t mask = 0x000fffffffffffffUL >> (63 - fill_bits);
        // Now, mask the backfill.
        backfill &= mask;
        // Now, when we perform the subtraction, we can backfill these values in.
        if (bitmanip.floating < 0.0) {
            bitmanip.floating += 1.0;
        } else {
            bitmanip.floating -= 1.0;
        }

        if ((bitmanip.integer & 0x7fffffffffffffUL) != 0) {
            // Backfill. Use xor just to be fancy.
            bitmanip.integer ^= backfill;
        }

        // Now, scale and recenter the value.
        return bitmanip.floating;
    }

  private:
    std::uniform_int_distribution<uint64_t> mag_dist_;
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