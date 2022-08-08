#undef EINSUMS_USE_CATCH2
#undef EINSUMS_CONTINUOUSLY_TEST_EINSUM
#undef EINSUMS_TEST_NANS

#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/Timer.hpp"
#include "einsums/Utilities.hpp"

#include <catch2/catch.hpp>

TEST_CASE("timer") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

#pragma omp parallel for
    for (int i = 0; i < 1000; i++) {
        timer::push("A: test timer");
        timer::pop();
    }

    auto A = create_random_tensor("A", 100, 100);
    auto B = create_random_tensor("B", 100, 100);

    // println("pre omp_get_max_active_levels {}", omp_get_max_active_levels());
#pragma omp parallel for
    for (int _i = 0; _i < 1000; _i++) {
        timer::push("B: test timer");
        timer::push("B: test timer 2");

        auto C = create_tensor("C", 100, 100);
        zero(C);

        einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);

        timer::pop();
        timer::pop();
    }

    // println("omp_get_supported_active_levels {}", omp_get_supported_active_levels());
    // println("post omp_get_max_active_levels {}", omp_get_max_active_levels());
}