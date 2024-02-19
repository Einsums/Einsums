#include "einsums/_Common.hpp"

#include "einsums/DeviceTensor.hpp"

#include <catch2/catch_all.hpp>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

#include "catch2/catch_test_macros.hpp"

TEST_CASE("Constructing Views", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    einsums::gpu::initialize();

    // Create tensors for testing.
    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    A = 0.0;
    B = 1.0;
    C = 2.0;

    // Test a constructor. They all call the same method, so we should only need to test one.
    DeviceTensorView<double, 2> A1(A, Dim<2>{3, 3});
    DeviceTensorView<double, 2> B1(B, Dim<2>{3, 3});
    DeviceTensorView<double, 2> C1(C, Dim<2>{3, 3});

    DeviceTensorView<double, 1> A2(A, Dim<1>{9});
    DeviceTensorView<double, 1> B2(B, Dim<1>{9});
    DeviceTensorView<double, 1> C2(C, Dim<1>{9});

    // Test various factory methods from DeviceTensor.
    auto A4 = A.to_rank_1_view();
    auto B4 = B.to_rank_1_view();
    auto C4 = C.to_rank_1_view();

    auto A5 = A(AllT{}, AllT{});
    auto B5 = B(AllT{}, AllT{});
    auto C5 = C(AllT{}, AllT{});

    auto A6 = A(Range{0, 2}, Range{0, 2});
    auto B6 = B(Range{0, 2}, Range{0, 2});
    auto C6 = C(Range{0, 2}, Range{0, 2});

    // Test the opposite direction.
    DeviceTensor<double, 2> A7(A6);
    DeviceTensor<double, 2> B7(B6);
    DeviceTensor<double, 2> C7(C6);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(A1(i, j) == 0.0);
            CHECK(B1(i, j) == 1.0);
            CHECK(C1(i, j) == 2.0);
            CHECK(A5(i, j) == 0.0);
            CHECK(B5(i, j) == 1.0);
            CHECK(C5(i, j) == 2.0);

            if (i != 2 && j != 2) {
                CHECK(A6(i, j) == 0.0);
                CHECK(B6(i, j) == 1.0);
                CHECK(C6(i, j) == 2.0);
                CHECK(A7(i, j) == 0.0);
                CHECK(B7(i, j) == 1.0);
                CHECK(C7(i, j) == 2.0);
            }
        }
    }

    for (int i = 0; i < 9; i++) {
        CHECK(A2(i) == 0.0);
        CHECK(B2(i) == 1.0);
        CHECK(C2(i) == 2.0);
        CHECK(A4(i) == 0.0);
        CHECK(B4(i) == 1.0);
        CHECK(C4(i) == 2.0);
    }
}

TEST_CASE("GPU View Set all", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    gpu::initialize();

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    A.set_all(1.0);
    B.set_all(2.0);
    C.set_all(3.0);

    DeviceTensorView<double, 2> A1(A, Dim<2>{3, 3});
    DeviceTensorView<double, 2> B1(B, Dim<2>{3, 3});
    DeviceTensorView<double, 2> C1(C, Dim<2>{3, 3});

    A1 = 4.0;
    B1 = 5.0;
    C1 = 6.0;

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK(A(i, j).get() == 4.0);
            CHECK((B(i, j).get() == 5.0));
            CHECK((C(i, j).get() == 6.0));
            CHECK((A1(i, j) == 4.0));
            CHECK((B1(i, j) == 5.0));
            CHECK((C1(i, j) == 6.0));
        }
    }

    A = 1.0;
    B = 2.0;
    C = 3.0;

    A1 = A;
    B1 = B;
    C1 = C;

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((A(i, j).get() == 1.0));
            CHECK((B(i, j).get() == 2.0));
            CHECK((C(i, j).get() == 3.0));
            CHECK((A1(i, j) == 1.0));
            CHECK((B1(i, j) == 2.0));
            CHECK((C1(i, j) == 3.0));
        }
    }
}