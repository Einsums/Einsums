//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Transpose C", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    size_t _i = 3, _j = 4, _k = 5;

    SECTION("i,j <- j,k * k,i === true, false, false") {
        Tensor A  = create_random_tensor<TestType>("A", _j, _k);
        Tensor B  = create_random_tensor<TestType>("B", _k, _i);
        Tensor C  = create_tensor<TestType>("C", _i, _j);
        Tensor C0 = create_tensor<TestType>("C0", _i, _j);
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j, k}, A, Indices{k, i}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(j0, k0) * B(k0, i0);
                }
            }
        }

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * k,i === true, true, false") {
        Tensor A  = create_random_tensor<TestType>("A", _k, _j);
        Tensor B  = create_random_tensor<TestType>("B", _k, _i);
        Tensor C  = create_tensor<TestType>("C", _i, _j);
        Tensor C0 = create_tensor<TestType>("C0", _i, _j);
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{k, j}, A, Indices{k, i}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(k0, j0) * B(k0, i0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- j,k * i,k === true, false, true") {
        Tensor A  = create_random_tensor<TestType>("A", _j, _k);
        Tensor B  = create_random_tensor<TestType>("B", _i, _k);
        Tensor C  = create_tensor<TestType>("C", _i, _j);
        Tensor C0 = create_tensor<TestType>("C0", _i, _j);
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j, k}, A, Indices{i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(j0, k0) * B(i0, k0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }

    SECTION("i,j <- k,j * i,k === true, true, true") {
        Tensor A  = create_random_tensor<TestType>("A", _k, _j);
        Tensor B  = create_random_tensor<TestType>("B", _i, _k);
        Tensor C  = create_tensor<TestType>("C", _i, _j);
        Tensor C0 = create_tensor<TestType>("C0", _i, _j);
        C0.zero();

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{k, j}, A, Indices{i, k}, B));

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                for (size_t k0 = 0; k0 < _k; k0++) {
                    C0(i0, j0) += A(k0, j0) * B(i0, k0);
                }
            }
        }

        // println(C0);
        // println(C);

        for (size_t i0 = 0; i0 < _i; i0++) {
            for (size_t j0 = 0; j0 < _j; j0++) {
                CheckWithinRel(C(i0, j0), C0(i0, j0), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(i0, j0), Catch::Matchers::WithinAbs(C0(i0, j0), 0.001));
            }
        }
    }
}
