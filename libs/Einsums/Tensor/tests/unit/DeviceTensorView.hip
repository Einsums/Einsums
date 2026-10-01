#include <Einsums/Tensor/DeviceTensor.hpp>

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

#include <Einsums/Testing.hpp>

TEST_CASE("Constructing Views", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    // Create tensors for testing.
    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    hip_catch(hipDeviceSynchronize());

    A = 0.0;
    B = 1.0;
    C = 2.0;

    hip_catch(hipDeviceSynchronize());

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

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((double)A1(i, j) == 0.0);
            CHECK((double)B1(i, j) == 1.0);
            CHECK((double)C1(i, j) == 2.0);
            CHECK((double)A5(i, j) == 0.0);
            CHECK((double)B5(i, j) == 1.0);
            CHECK((double)C5(i, j) == 2.0);

            if (i != 2 && j != 2) {
                CHECK((double)A6(i, j) == 0.0);
                CHECK((double)B6(i, j) == 1.0);
                CHECK((double)C6(i, j) == 2.0);
                CHECK((double)A7(i, j) == 0.0);
                CHECK((double)B7(i, j) == 1.0);
                CHECK((double)C7(i, j) == 2.0);
            }
        }
    }

    for (int i = 0; i < 9; i++) {
        CHECK((double)A2(i) == 0.0);
        CHECK((double)B2(i) == 1.0);
        CHECK((double)C2(i) == 2.0);
        CHECK((double)A4(i) == 0.0);
        CHECK((double)B4(i) == 1.0);
        CHECK((double)C4(i) == 2.0);
    }
}

TEST_CASE("GPU View Set all", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    hip_catch(hipDeviceSynchronize());

    A.set_all(1.0);
    B.set_all(2.0);
    C.set_all(3.0);

    hip_catch(hipDeviceSynchronize());

    DeviceTensorView<double, 2> A1(A, Dim<2>{3, 3});
    DeviceTensorView<double, 2> B1(B, Dim<2>{3, 3});
    DeviceTensorView<double, 2> C1(C, Dim<2>{3, 3});

    hip_catch(hipDeviceSynchronize());

    A1 = 4.0;
    B1 = 5.0;
    C1 = 6.0;

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((double)A(i, j) == 4.0);
            CHECK((double)B(i, j) == 5.0);
            CHECK((double)C(i, j) == 6.0);
            CHECK((double)A1(i, j) == 4.0);
            CHECK((double)B1(i, j) == 5.0);
            CHECK((double)C1(i, j) == 6.0);
        }
    }

    A = 1.0;
    B = 2.0;
    C = 3.0;

    hip_catch(hipDeviceSynchronize());

    A1 = A;
    B1 = B;
    C1 = C;

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((double)A(i, j) == 1.0);
            CHECK((double)B(i, j) == 2.0);
            CHECK((double)C(i, j) == 3.0);
            CHECK((double)A1(i, j) == 1.0);
            CHECK((double)B1(i, j) == 2.0);
            CHECK((double)C1(i, j) == 3.0);
        }
    }
}