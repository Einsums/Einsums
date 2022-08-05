#include "einsums/Blas.hpp"
#include "einsums/Timer.hpp"
#include "einsums/_Common.hpp"

namespace einsums {

auto initialize() -> int {
    timer::initialize();
    blas::initialize();

    return 0;
}

void finalize(bool timerReport) {
    blas::finalize();

    if (timerReport)
        timer::report();

    timer::finalize();
}

} // namespace einsums