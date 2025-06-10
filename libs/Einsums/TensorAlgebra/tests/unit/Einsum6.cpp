//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/Detail/Utilities.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("F12 - V term", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{1}, ncabs{4}, nobs{2};
    int nall{nobs + ncabs};

    auto F = create_incremented_tensor("F", nall, nall, nall, nall);
    auto G = create_incremented_tensor("G", nall, nall, nall, nall);

    TensorView F_ooco{F, Dim{nocc, nocc, ncabs, nocc}, Offset{0, 0, nobs, 0}};
    TensorView F_oooc{F, Dim{nocc, nocc, nocc, ncabs}, Offset{0, 0, 0, nobs}};
    TensorView F_oopq{F, Dim{nocc, nocc, nobs, nobs}, Offset{0, 0, 0, 0}};
    TensorView G_ooco{G, Dim{nocc, nocc, ncabs, nocc}, Offset{0, 0, nobs, 0}};
    TensorView G_oooc{G, Dim{nocc, nocc, nocc, ncabs}, Offset{0, 0, 0, nobs}};
    TensorView G_oopq{G, Dim{nocc, nocc, nobs, nobs}, Offset{0, 0, 0, 0}};

    Tensor ijkl_1 = create_zero_tensor<TestType>("Einsum Temp 1", nocc, nocc, nocc, nocc);
    Tensor ijkl_2 = create_zero_tensor<TestType>("Einsum Temp 2", nocc, nocc, nocc, nocc);
    Tensor ijkl_3 = create_zero_tensor<TestType>("Einsum Temp 3", nocc, nocc, nocc, nocc);

    Tensor result  = create_zero_tensor<TestType>("Result", nocc, nocc, nocc, nocc);
    Tensor result2 = create_zero_tensor<TestType>("Result2", nocc, nocc, nocc, nocc);

    einsum(Indices{i, j, k, l}, &ijkl_1, Indices{i, j, p, n}, G_ooco, Indices{k, l, p, n}, F_ooco, &alg_choice);
    REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);
    einsum(Indices{i, j, k, l}, &ijkl_2, Indices{i, j, m, q}, G_oooc, Indices{k, l, m, q}, F_oooc, &alg_choice);
    REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);
    einsum(Indices{i, j, k, l}, &ijkl_3, Indices{i, j, p, q}, G_oopq, Indices{k, l, p, q}, F_oopq, &alg_choice);
    REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    for (size_t _p = 0; _p < ncabs; _p++) {
                        for (size_t _n = 0; _n < nocc; _n++) {
                            result(_i, _j, _k, _l) += G(_i, _j, nobs + _p, _n) * F(_k, _l, nobs + _p, _n);
                            result2(_i, _j, _k, _l) += G_ooco(_i, _j, _p, _n) * F_ooco(_k, _l, _p, _n);
                        }
                    }
                }
            }
        }
    }

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(result2(_i, _j, _k, _l), CheckWithinRel(result(_i, _j, _k, _l), 0.001));
                    // REQUIRE_THAT(result2(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }

    for (size_t _i = 0; _i < nocc; _i++) {
        for (size_t _j = 0; _j < nocc; _j++) {
            for (size_t _k = 0; _k < nocc; _k++) {
                for (size_t _l = 0; _l < nocc; _l++) {
                    REQUIRE_THAT(ijkl_1(_i, _j, _k, _l), CheckWithinRel(result(_i, _j, _k, _l), 0.001));
                    // REQUIRE_THAT(ijkl_1(_i, _j, _k, _l), Catch::Matchers::WithinAbs(result(_i, _j, _k, _l), 0.001));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("B_tilde", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    tensor_algebra::detail::AlgorithmChoice alg_choice;

    // int nocc{5}, ncabs{116}, nobs{41};
    int nocc{5}, ncabs{10}, nobs{10};
    assert(nobs > nocc); // sanity check
    int nall{nobs + ncabs}, nvir{nobs - nocc};

    Tensor CD   = create_zero_tensor<TestType>("CD", nocc, nocc, nvir, nvir);
    Tensor CD0  = create_zero_tensor<TestType>("CD0", nocc, nocc, nvir, nvir);
    auto   C    = create_random_tensor<TestType>("C", nocc, nocc, nvir, nvir);
    auto   D    = create_random_tensor<TestType>("D", nocc, nocc, nvir, nvir);
    auto   D_ij = D(2, 2, All, All);

    einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, C, Indices{a, b}, D_ij, &alg_choice);
    REQUIRE(alg_choice == tensor_algebra::detail::GENERIC);

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    CD0(_k, _l, _a, _b) = C(_k, _l, _a, _b) * D(2, 2, _a, _b);
                }
            }
        }
    }

    for (int _k = 0; _k < nocc; _k++) {
        for (int _l = 0; _l < nocc; _l++) {
            for (int _a = 0; _a < nvir; _a++) {
                for (int _b = 0; _b < nvir; _b++) {
                    REQUIRE_THAT(CD(_k, _l, _a, _b), CheckWithinRel(CD0(_k, _l, _a, _b), 0.0001));
                    // REQUIRE_THAT(CD(_k, _l, _a, _b), Catch::Matchers::WithinAbs(CD0(_k, _l, _a, _b), 0.000001));
                }
            }
        }
    }
}
