//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateIncrementedTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("permute2", "[tensor]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    SECTION("Rank 2 - axpy") {
        Tensor A{"A", 3, 3};
        Tensor C{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        permute(Indices{i, j}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE(C(i, j) == A(i, j));
            }
        }

        TensorView A_view{A, Dim<2>{2, 2}, Offset<2>{1, 1}};
        TensorView C_view{C, Dim<2>{2, 2}, Offset<2>{1, 1}};

        permute(Indices{j, i}, &C_view, Indices{i, j}, A_view);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                printf("%d, %d, %d, %e, %e, %e", i, j, ij, A(i, j), C(i, j), C(j, i));
                if (i == 0 || j == 0)
                    REQUIRE(C(i, j) == A(i, j));
                else
                    REQUIRE(C(j, i) == A(i, j));
            }
        }
    }

    SECTION("Rank 2 - axpy (2)") {
        Tensor A = create_incremented_tensor("A", 3, 3);
        Tensor C0{"C", 3, 3};
        Tensor C1{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = ij + A(i, j);
            }
        }

        permute(1.0, Indices{i, j}, &C0, 1.0, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT(C0(i, j), Catch::Matchers::WithinRel(C1(i, j), 0.00001));
            }
        }

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = 2.0 * ij + 0.5 * A(i, j);
            }
        }

        permute(2.0, Indices{i, j}, &C0, 0.5, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT(C0(i, j), Catch::Matchers::WithinRel(C1(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 2") {
        Tensor A{"A", 3, 3};
        Tensor C{"C", 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        permute(Indices{j, i}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE_THAT(C(j, i), Catch::Matchers::WithinRel(A(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 3") {
        Tensor A{"A", 3, 3, 3};
        Tensor B{"B", 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        permute(Indices{k, j, i}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(B(k, j, i), Catch::Matchers::WithinRel(A(i, j, k), 0.00001));
                }
            }
        }

        permute(Indices{i, k, j}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(B(i, k, j), Catch::Matchers::WithinRel(A(i, j, k), 0.00001));
                }
            }
        }

        permute(Indices{j, k, i}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(B(j, k, i), Catch::Matchers::WithinRel(A(i, j, k), 0.00001));
                }
            }
        }

        permute(Indices{i, j, k}, &B, Indices{k, j, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT(B(i, j, k), Catch::Matchers::WithinRel(A(k, j, i), 0.00001));
                }
            }
        }
    }

    SECTION("Rank 4") {
        Tensor A{"A", 3, 3, 3, 3};
        Tensor B{"B", 3, 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++, ij++) {
                        A(i, j, k, l) = ij;
                    }
                }
            }
        }

        permute(0.0, Indices{i, l, k, j}, &B, 0.5, Indices{k, j, l, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        REQUIRE_THAT(B(i, l, k, j), Catch::Matchers::WithinRel(0.5 * A(k, j, l, i), 0.00001));
                    }
                }
            }
        }
    }

    // SECTION("Rank 5") {
    //     Tensor<float, 5> A{"A", 3, 3, 3, 3, 3};
    //     Tensor<float, 5> B{"B", 3, 3, 3, 3, 3};

    //     for (short i = 0, ij = 1; i < 3; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             for (int k = 0; k < 3; k++) {
    //                 for (int l = 0; l < 3; l++) {
    //                     for (int m = 0; m < 3; m++, ij++) {
    //                         A(i, j, k, l, m) = ij;
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     permute(Indices{i, k, l, m, j}, &B, Indices{j, k, l, m, i}, A);
    //     for (int i = 0; i < 3; i++) {
    //         for (int j = 0; j < 3; j++) {
    //             for (int k = 0; k < 3; k++) {
    //                 for (int l = 0; l < 3; l++) {
    //                     for (int m = 0; m < 3; m++) {
    //                         REQUIRE(B(i, k, l, m, j) == A(j, k, l, m, i));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }

    SECTION("Rank 2 - Different Sizes") {
        Tensor A{"A", 3, 9};
        Tensor B{"B", 9, 3};

        for (int i = 0, ij = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++, ij++) {
                A(i, j) = ij;
            }
        }

        permute(Indices{j, i}, &B, Indices{i, j}, A);
        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                REQUIRE_THAT(B(j, i), Catch::Matchers::WithinRel(A(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 3 - Different Sizes") {
        Tensor A{"A", 2, 3, 4};
        Tensor B{"B", 3, 4, 2};

        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        permute(Indices{j, k, i}, &B, Indices{i, j, k}, A);
        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    REQUIRE_THAT(B(j, k, i), Catch::Matchers::WithinRel(A(i, j, k), 0.00001));
                }
            }
        }
    }
}
