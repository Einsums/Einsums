#include <random>
#include <Einsums/Config.hpp>

namespace einsums {

EINSUMS_EXPORT extern std::default_random_engine random_engine;

EINSUMS_EXPORT void seed_random(std::default_random_engine::result_type seed);

}