#include <random>
#include <Einsums/Runtime/InitializeFinalize.hpp>
#include <Einsums/Testing.hpp>
#include <Einsums/Utilities/Random.hpp>

namespace einsums {
int initialize_testing() {
    int ret = initialize();
    
    einsums::random_engine = std::default_random_engine(Catch::rngSeed());
    return ret;
}

} // namespace einsums