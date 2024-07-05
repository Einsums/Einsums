#include "einsums/BlockTensor.hpp"
#include "einsums/DeviceTensor.hpp"
#include "einsums/ElementOperations.hpp"

#include <catch2/catch_all.hpp>

#define CLOSE 1e-6

TEST_CASE("gpu element reductions", "[gpu]") {
    using namespace einsums;

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = 3 * i + j + 1;
        }
    }

    CHECK(element_operations::max(A) == 9);
    CHECK(element_operations::min(A) == 1);
    CHECK(element_operations::sum(A) == 45);
}

TEST_CASE("gpu element broadcast", "[gpu]") {
    using namespace einsums;

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3), B("B", einsums::detail::DEV_ONLY, 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = (rand() & 4) ? 3 * i + j + 1 : -(3 * i + j + 1);
            B(i, j) = 3 * i + j + 1;
        }
    }

    auto A_abs = element_operations::new_tensor::abs(A);
    auto A_inv = element_operations::new_tensor::invert(A);
    auto A_exp = element_operations::new_tensor::exp(A);
    auto A_scale = element_operations::new_tensor::scale(2.0, A);

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            CHECK_THAT((double) A_abs(i, j), Catch::Matchers::WithinRel((double) B(i, j), CLOSE));
            CHECK_THAT((double) A_inv(i, j), Catch::Matchers::WithinRel(1 / A(i, j), CLOSE));
            CHECK_THAT((double) A_exp(i, j), Catch::Matchers::WithinRel(std::exp(A(i, j)), CLOSE));
            CHECK_THAT((double) A_scale(i, j), Catch::Matchers::WithinRel(2 * A(i, j), CLOSE));
        }
    }
}

TEST_CASE("gpu block-wise element reductions", "[gpu]") {
    using namespace einsums;

    BlockDeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[0](i, j) = 3 * i + j + 1;
            A[1](i, j) = 3 * i + j + 10;
        }
    }

    CHECK(element_operations::max(A) == 18);
    CHECK(element_operations::min(A) == 1);
    CHECK(element_operations::sum(A) == 171);
}

TEST_CASE("gpu block-wise element broadcast", "[gpu]") {
    using namespace einsums;

    BlockDeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3}, B{"B", einsums::detail::DEV_ONLY, 3, 3};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[0](i, j) = (rand() & 4) ? 3 * i + j + 1 : -(3 * i + j + 1);
            B[0](i, j) = 3 * i + j + 1;
            A[1](i, j) = (rand() & 4) ? 3 * i + j + 10 : -(3 * i + j + 10);
            B[1](i, j) = 3 * i + j + 10;
        }
    }

    auto A_abs = element_operations::new_tensor::abs(A);
    auto A_inv = element_operations::new_tensor::invert(A);
    auto A_exp = element_operations::new_tensor::exp(A);
    auto A_scale = element_operations::new_tensor::scale(2.0, A);

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            CHECK_THAT((double) A_abs[0](i, j), Catch::Matchers::WithinRel((double) B[0](i, j), CLOSE));
            CHECK_THAT((double) A_inv[0](i, j), Catch::Matchers::WithinRel(1 / A[0](i, j), CLOSE));
            CHECK_THAT((double) A_exp[0](i, j), Catch::Matchers::WithinRel(std::exp(A[0](i, j)), CLOSE));
            CHECK_THAT((double) A_scale[0](i, j), Catch::Matchers::WithinRel(2 * A[0](i, j), CLOSE));
            CHECK_THAT((double) A_abs[1](i, j), Catch::Matchers::WithinRel((double) B[1](i, j), CLOSE));
            CHECK_THAT((double) A_inv[1](i, j), Catch::Matchers::WithinRel(1 / A[1](i, j), CLOSE));
            CHECK_THAT((double) A_exp[1](i, j), Catch::Matchers::WithinRel(std::exp(A[1](i, j)), CLOSE));
            CHECK_THAT((double) A_scale[1](i, j), Catch::Matchers::WithinRel(2 * A[1](i, j), CLOSE));
        }
    }
}
