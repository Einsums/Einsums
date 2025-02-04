#include <Einsums/Config.hpp>

#include <random>

namespace einsums {

EINSUMS_EXPORT extern std::default_random_engine random_engine;

/**
 * @brief Set the seed of the random number generator.
 *
 * @param seed The new seed for the random number generator.
 */
EINSUMS_EXPORT void seed_random(std::default_random_engine::result_type seed);

} // namespace einsums