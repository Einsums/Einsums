#include <random>
#include <chrono>
#include <Einsums/Utilities/Random.hpp>

namespace einsums {
std::default_random_engine random_engine;


void seed_random(std::default_random_engine::result_type seed) {
    random_engine.seed(seed);
}

}
