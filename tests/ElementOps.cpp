#include "einsums/BlockTensor.hpp"
#include "einsums/ElementOperations.hpp"
#include "einsums/Tensor.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("element reductions") {
    using namespace einsums;

    Tensor<double, 2> A("A", 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = 3 * i + j + 1;
        }
    }

    CHECK(element_operations::max(A) == 9);
    CHECK(element_operations::min(A) == 1);
    CHECK(element_operations::sum(A) == 45);
}

TEST_CASE("element broadcast") {
    using namespace einsums;

    Tensor<double, 2> A("A", 3, 3), B("B", 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A(i, j) = (rand() & 4) ? 3 * i + j + 1 : -(3 * i + j + 1);
            B(i, j) = 3 * i + j + 1;
        }
    }

    auto A_abs   = element_operations::new_tensor::abs(A);
    auto A_inv   = element_operations::new_tensor::invert(A);
    auto A_exp   = element_operations::new_tensor::exp(A);
    auto A_scale = element_operations::new_tensor::scale(2.0, A);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(A_abs(i, j) == B(i, j));
            CHECK(A_inv(i, j) == 1 / A(i, j));
            CHECK(A_exp(i, j) == std::exp(A(i, j)));
            CHECK(A_scale(i, j) == 2 * A(i, j));
        }
    }
}

TEST_CASE("block-wise element reductions") {
    using namespace einsums;

    BlockTensor<double, 2> A("A", 3, 3);

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

TEST_CASE("block-wise element broadcast") {
    using namespace einsums;

    BlockTensor<double, 2> A("A", 3, 3), B("B", 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[0](i, j) = (rand() & 4) ? 3 * i + j + 1 : -(3 * i + j + 1);
            B[0](i, j) = 3 * i + j + 1;
            A[1](i, j) = (rand() & 4) ? 3 * i + j + 10 : -(3 * i + j + 10);
            B[1](i, j) = 3 * i + j + 10;
        }
    }

    auto A_abs   = element_operations::new_tensor::abs(A);
    auto A_inv   = element_operations::new_tensor::invert(A);
    auto A_exp   = element_operations::new_tensor::exp(A);
    auto A_scale = element_operations::new_tensor::scale(2.0, A);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(A_abs[0](i, j) == B[0](i, j));
            CHECK(A_inv[0](i, j) == 1 / A[0](i, j));
            CHECK(A_exp[0](i, j) == std::exp(A[0](i, j)));
            CHECK(A_scale[0](i, j) == 2 * A[0](i, j));
            CHECK(A_abs[1](i, j) == B[1](i, j));
            CHECK(A_inv[1](i, j) == 1 / A[1](i, j));
            CHECK(A_exp[1](i, j) == std::exp(A[1](i, j)));
            CHECK(A_scale[1](i, j) == 2 * A[1](i, j));
        }
    }
}