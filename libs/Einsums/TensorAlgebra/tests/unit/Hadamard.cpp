//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Hadamard", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- i,i * j*j") {
        auto A  = create_random_tensor<TestType>("A", _i, _i);
        auto B  = create_random_tensor<TestType>("B", _j, _j);
        auto C  = create_tensor<TestType>("C", _i, _j);
        auto C0 = create_tensor<TestType>("C0", _i, _j);
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, i0) * B(j0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, i}, A, Indices{j, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.0001});
            }
        }
    }

    SECTION("i,j <- i,i,j * j,j,i") {
        auto A  = create_random_tensor<TestType>("A", _i, _i, _j);
        auto B  = create_random_tensor<TestType>("B", _j, _j, _i);
        auto C  = create_tensor<TestType>("C", _i, _j);
        auto C0 = create_tensor<TestType>("C0", _i, _j);
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, i0, j0) * B(j0, j0, i0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, i, j}, A, Indices{j, j, i}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.0001});
            }
        }
    }

    SECTION("i,j <- i,j,i * j,i,j") {
        auto A  = create_random_tensor<TestType>("A", _i, _j, _i);
        auto B  = create_random_tensor<TestType>("B", _j, _i, _j);
        auto C  = create_tensor<TestType>("C", _i, _j);
        auto C0 = create_tensor<TestType>("C0", _i, _j);
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.0001});
            }
        }
    }

    SECTION("i,j,i <- i,j,i * j,i,j") {
        auto A  = create_random_tensor<TestType>("A", _i, _j, _i);
        auto B  = create_random_tensor<TestType>("B", _j, _i, _j);
        auto C  = create_tensor<TestType>("C", _i, _j, _i);
        auto C0 = create_tensor<TestType>("C0", _i, _j, _i);
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, j0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, j, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0, i0), C0(i0, j0, i0), RemoveComplexT<TestType>{0.0001});
            }
        }
    }

    SECTION("i,i,i <- i,j,i * j,i,j") {
        auto A  = create_random_tensor<TestType>("A", _i, _j, _i);
        auto B  = create_random_tensor<TestType>("B", _j, _i, _j);
        auto C  = create_tensor<TestType>("C", _i, _i, _i);
        auto C0 = create_tensor<TestType>("C0", _i, _i, _i);
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                C0(i0, i0, i0) += A(i0, j0, i0) * B(j0, i0, j0);
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, i, i}, &C, Indices{i, j, i}, A, Indices{j, i, j}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            CheckWithinRel(C(i0, i0, i0), C0(i0, i0, i0), RemoveComplexT<TestType>{0.0001});
        }
    }

    SECTION("i,i <- i,j,k * j,i,k") {
        auto A  = create_random_tensor<TestType>("A", _i, _j, _k);
        auto B  = create_random_tensor<TestType>("B", _j, _i, _k);
        auto C  = create_tensor<TestType>("C", _i, _i);
        auto C0 = create_tensor<TestType>("C0", _i, _i);
        C0.zero();
        C.zero();

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, i0) += A(i0, j0, k0) * B(j0, i0, k0);
                }
            }
        }

        REQUIRE_NOTHROW(einsum(Indices{i, i}, &C, Indices{i, j, k}, A, Indices{j, i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _i; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.0001});
            }
        }
    }
}
