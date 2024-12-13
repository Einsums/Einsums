#include <random>
#include <chrono>
#include <Einsums/Utilities/Random.hpp>

std::default_random_engine einsums::random_engine(std::chrono::system_clock::now().time_since_epoch().count());