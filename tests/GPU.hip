#include "einsums/_Common.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/GPULinearAlgebra.hpp"

#include <catch2/catch_all.hpp>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

#include "catch2/catch_test_macros.hpp"
#include "einsums.hpp"
#include "src/backends/linear_algebra/hipblas/hipblas.hpp"

// Waste time for timing.
__global__ void time_waster(unsigned int max) {
    volatile float value;
    for (unsigned int i = 0; i < max; i++) {
        value = __sinf(i);
    }
}

#define KERNELS 20
#define CYCLES  0xffffU

TEST_CASE("GPU Timer", "[gpu][timer]") {

        einsums::timer::GPUTimer *timer = new einsums::timer::GPUTimer("GPU Timer");
        for (int i = 0; i < KERNELS; i++) {
            time_waster<<<dim3(256), dim3(2)>>>(CYCLES);
        }
        delete timer;
}

TEST_CASE("gpu-sort2") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    SECTION("Rank 2 - axpy") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        sort(Indices{i, j}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE(C(i, j) == A(i, j));
            }
        }

        DeviceTensorView A_view{A, Dim<2>{2, 2}, Offset<2>{1, 1}};
        DeviceTensorView C_view{C, Dim<2>{2, 2}, Offset<2>{1, 1}};

        sort(Indices{j, i}, &C_view, Indices{i, j}, A_view);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                if (i == 0 || j == 0)
                    REQUIRE(C(i, j) == A(i, j));
                else
                    REQUIRE(C(j, i) == A(i, j));
            }
        }
    }

    SECTION("Rank 2 - axpy (2)") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C0{"C", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C1{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = ij + A(i, j);
            }
        }

        sort(1.0, Indices{i, j}, &C0, 1.0, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT((double) C0(i, j), Catch::Matchers::WithinRel((double) C1(i, j), 0.00001));
            }
        }

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                C0(i, j) = ij;
                C1(i, j) = 2.0 * ij + 0.5 * A(i, j);
            }
        }

        sort(2.0, Indices{i, j}, &C0, 0.5, Indices{i, j}, A);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                REQUIRE_THAT((double) C0(i, j), Catch::Matchers::WithinRel((double) C1(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 2") {
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 3};
        DeviceTensor<double, 2> C{"C", einsums::detail::DEV_ONLY, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                A(i, j) = ij;
            }
        }

        sort(Indices{j, i}, &C, Indices{i, j}, A);

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++, ij++) {
                REQUIRE_THAT((double) C(j, i), Catch::Matchers::WithinRel((double) A(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 3") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 3, 3, 3};
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        sort(Indices{k, j, i}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double) B(k, j, i), Catch::Matchers::WithinRel((double) A(i, j, k), 0.00001));
                }
            }
        }

        sort(Indices{i, k, j}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double) B(i, k, j), Catch::Matchers::WithinRel((double) A(i, j, k), 0.00001));
                }
            }
        }

        sort(Indices{j, k, i}, &B, Indices{i, j, k}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double) B(j, k, i), Catch::Matchers::WithinRel((double) A(i, j, k), 0.00001));
                }
            }
        }

        sort(Indices{i, j, k}, &B, Indices{k, j, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    REQUIRE_THAT((double) B(i, j, k), Catch::Matchers::WithinRel((double) A(k, j, i), 0.00001));
                }
            }
        }
    }

    SECTION("Rank 4") {
        DeviceTensor<double, 4> A{"A", einsums::detail::DEV_ONLY, 3, 3, 3, 3};
        DeviceTensor<double, 4> B{"B", einsums::detail::DEV_ONLY, 3, 3, 3, 3};

        for (int i = 0, ij = 1; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++, ij++) {
                        A(i, j, k, l) = ij;
                    }
                }
            }
        }

        sort(0.0, Indices{i, l, k, j}, &B, 0.5, Indices{k, j, l, i}, A);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        REQUIRE_THAT((double) B(i, l, k, j), Catch::Matchers::WithinRel(0.5 * A(k, j, l, i), 0.00001));
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

    //     sort(Indices{i, k, l, m, j}, &B, Indices{j, k, l, m, i}, A);
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
        DeviceTensor<double, 2> A{"A", einsums::detail::DEV_ONLY, 3, 9};
        DeviceTensor<double, 2> B{"B", einsums::detail::DEV_ONLY, 9, 3};

        for (int i = 0, ij = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++, ij++) {
                A(i, j) = ij;
            }
        }

        sort(Indices{j, i}, &B, Indices{i, j}, A);
        for (int i = 0; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                REQUIRE_THAT((double) B(j, i), Catch::Matchers::WithinRel((double) A(i, j), 0.00001));
            }
        }
    }

    SECTION("Rank 3 - Different Sizes") {
        DeviceTensor<double, 3> A{"A", einsums::detail::DEV_ONLY, 2, 3, 4};
        DeviceTensor<double, 3> B{"B", einsums::detail::DEV_ONLY, 3, 4, 2};

        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    A(i, j, k) = ij;
                }
            }
        }

        sort(Indices{j, k, i}, &B, Indices{i, j, k}, A);
        for (int i = 0, ij = 1; i < A.dim(0); i++) {
            for (int j = 0; j < A.dim(1); j++) {
                for (int k = 0; k < A.dim(2); k++, ij++) {
                    REQUIRE_THAT((double) B(j, k, i), Catch::Matchers::WithinRel((double) A(i, j, k), 0.00001));
                }
            }
        }
    }
}