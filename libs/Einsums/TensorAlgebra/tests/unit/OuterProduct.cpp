//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("outer product", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    size_t _x{100}, _y{100};

    SECTION("1 * 1 -> 2") {
        Tensor A = create_random_tensor<TestType>("A", _x);
        Tensor B = create_random_tensor<TestType>("B", _y);
        Tensor C = create_zero_tensor<TestType>("C", _x, _y);
        zero(C);

        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                CheckWithinRel(C(x, y), A(x) * B(y), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                CheckWithinRel(C(x, y), A(y) * B(x), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(x, y), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{j}, A, Indices{i}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                CheckWithinRel(C(y, x), A(y) * B(x), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(y) * B(x), 0.001));
            }
        }

        // C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{j, i}, &C, Indices{i}, A, Indices{j}, B));

        for (int x = 0; x < _x; x++) {
            for (int y = 0; y < _y; y++) {
                CheckWithinRel(C(y, x), A(x) * B(y), RemoveComplexT<TestType>{0.001});
                // REQUIRE_THAT(C(y, x), Catch::Matchers::WithinAbs(A(x) * B(y), 0.001));
            }
        }
    }

    SECTION("2 * 1 -> 3") {
        Tensor A = create_random_tensor<TestType>("A", 3, 3);
        Tensor B = create_random_tensor<TestType>("B", 3);
        Tensor C = create_tensor<TestType>("C", 3, 3, 3);

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    CheckWithinRel(C(x, y, z), A(x, y) * B(z), RemoveComplexT<TestType>{0.001});
                    // REQUIRE_THAT(C(x, y, z), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{i, j}, A, Indices{k}, B));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    CheckWithinRel(C(z, x, y), A(x, y) * B(z), RemoveComplexT<TestType>{0.001});
                    // REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{k, i, j}, &C, Indices{k}, B, Indices{i, j}, A));

        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                for (int z = 0; z < 3; z++) {
                    CheckWithinRel(C(z, x, y), A(x, y) * B(z), RemoveComplexT<TestType>{0.001});
                    // REQUIRE_THAT(C(z, x, y), Catch::Matchers::WithinAbs(A(x, y) * B(z), 0.001));
                }
            }
        }
    }

    SECTION("2 * 2 -> 4") {
        Tensor A = create_random_tensor<TestType>("A", 3, 3);
        Tensor B = create_random_tensor<TestType>("B", 3, 3);
        Tensor C = create_tensor<TestType>("C", 3, 3, 3, 3);
        ;

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{i, j}, A, Indices{k, l}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        CheckWithinRel(C(w, x, y, z), A(w, x) * B(y, z), RemoveComplexT<TestType>{0.001});
                        // REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(w, x) * B(y, z), 0.001));
                    }
                }
            }
        }

        C.set_all(0.0);
        REQUIRE_NOTHROW(einsum(Indices{i, j, k, l}, &C, Indices{k, l}, A, Indices{i, j}, B));

        for (int w = 0; w < 3; w++) {
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    for (int z = 0; z < 3; z++) {
                        CheckWithinRel(C(w, x, y, z), A(y, z) * B(w, x), RemoveComplexT<TestType>{0.001});
                        // REQUIRE_THAT(C(w, x, y, z), Catch::Matchers::WithinAbs(A(y, z) * B(w, x), 0.001));
                    }
                }
            }
        }
    }
}
