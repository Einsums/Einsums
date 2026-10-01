#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

#include <Einsums/Testing.hpp>

TEST_CASE("GPU Tensor Construction", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    // Test creation and access.
    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 1, 1);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 1, 1);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 1, 1);

    A.resize(einsums::Dim<2>{3, 3});
    B.resize(einsums::Dim<2>{3, 3});
    C.resize(einsums::Dim<2>{3, 3});

    CHECK((A.dim(0) == 3 && A.dim(1) == 3));
    CHECK((B.dim(0) == 3 && B.dim(1) == 3));
    CHECK((C.dim(0) == 3 && C.dim(1) == 3));

    DeviceTensor<double, 2> A_dim(einsums::Dim<2>{3, 3}, einsums::detail::DEV_ONLY);
    DeviceTensor<double, 2> B_dim(einsums::Dim<2>{3, 3}, einsums::detail::MAPPED);
    DeviceTensor<double, 2> C_dim(einsums::Dim<2>{3, 3}, einsums::detail::PINNED);

    CHECK((A.dim(0) == 3 && A.dim(1) == 3));
    CHECK((B.dim(0) == 3 && B.dim(1) == 3));
    CHECK((C.dim(0) == 3 && C.dim(1) == 3));

    CHECK((A_dim.dim(0) == 3 && A_dim.dim(1) == 3));
    CHECK((B_dim.dim(0) == 3 && B_dim.dim(1) == 3));
    CHECK((C_dim.dim(0) == 3 && C_dim.dim(1) == 3));

    CHECK((A.size() == 9));
    CHECK((B.size() == 9));
    CHECK((C.size() == 9));
    CHECK((A_dim.size() == 9));
    CHECK((B_dim.size() == 9));
    CHECK((C_dim.size() == 9));

    A.zero();
    B.zero();
    C.zero();

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((A(i, j).get() == 0));
            CHECK((B(i, j).get() == 0));
            CHECK((C(i, j).get() == 0));
        }
    }

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    C(2, 2) = 1.0;

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        CHECK((A(i, i).get() == 1.0));
        CHECK((B(i, i).get() == 1.0));
        CHECK((C(i, i).get() == 1.0));
        for (size_t j = 0; j < i; j++) {
            CHECK((A(i, j).get() == 0));
            CHECK((B(i, j).get() == 0));
            CHECK((C(i, j).get() == 0));
            CHECK((A(j, i).get() == 0));
            CHECK((B(j, i).get() == 0));
            CHECK((C(j, i).get() == 0));
        }
    }

    // Test copy
    DeviceTensor<double, 2> A_copy(A);
    DeviceTensor<double, 2> B_copy(B);
    DeviceTensor<double, 2> C_copy(C);

    CHECK((A_copy.dim(0) == 3 && A_copy.dim(1) == 3));
    CHECK((B_copy.dim(0) == 3 && B_copy.dim(1) == 3));
    CHECK((C_copy.dim(0) == 3 && C_copy.dim(1) == 3));

    A.zero();
    B.zero();
    C.zero();

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((A(i, j).get() == 0));
            CHECK((B(i, j).get() == 0));
            CHECK((C(i, j).get() == 0));
        }
    }

    for (size_t i = 0; i < 3; i++) {
        CHECK((A_copy(i, i).get() == 1.0));
        CHECK((B_copy(i, i).get() == 1.0));
        CHECK((C_copy(i, i).get() == 1.0));
        for (size_t j = 0; j < i; j++) {
            CHECK((A_copy(i, j).get() == 0));
            CHECK((B_copy(i, j).get() == 0));
            CHECK((C_copy(i, j).get() == 0));
            CHECK((A_copy(j, i).get() == 0));
            CHECK((B_copy(j, i).get() == 0));
            CHECK((C_copy(j, i).get() == 0));
        }
    }

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 1.0;

    B(0, 0) = 1.0;
    B(1, 1) = 1.0;
    B(2, 2) = 1.0;

    C(0, 0) = 1.0;
    C(1, 1) = 1.0;
    C(2, 2) = 1.0;

    A_copy.zero();
    B_copy.zero();
    C_copy.zero();

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((A_copy(i, j).get() == 0));
            CHECK((B_copy(i, j).get() == 0));
            CHECK((C_copy(i, j).get() == 0));
        }
    }

    for (size_t i = 0; i < 3; i++) {
        CHECK((A(i, i).get() == 1.0));
        CHECK((B(i, i).get() == 1.0));
        CHECK((C(i, i).get() == 1.0));
        for (size_t j = 0; j < i; j++) {
            CHECK((A(i, j).get() == 0));
            CHECK((B(i, j).get() == 0));
            CHECK((C(i, j).get() == 0));
            CHECK((A(j, i).get() == 0));
            CHECK((B(j, i).get() == 0));
            CHECK((C(j, i).get() == 0));
        }
    }

    // Test move.
    DeviceTensor<double, 2> A_new(std::move(A_copy), "A_new", 3, 3);
    DeviceTensor<double, 2> B_new(std::move(B_copy), "B_new", 3, 3);
    DeviceTensor<double, 2> C_new(std::move(C_copy), "C_new", 1, -1);

    REQUIRE(A_copy.gpu_data() == nullptr);
    REQUIRE(B_copy.gpu_data() == nullptr);
    REQUIRE(C_copy.gpu_data() == nullptr);

    hip_catch(hipDeviceSynchronize());
}

TEST_CASE("GPU Set all", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    A.set_all(1.0);
    B.set_all(2.0);
    C.set_all(3.0);

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((A(i, j).get() == 1.0));
            CHECK((B(i, j).get() == 2.0));
            CHECK((C(i, j).get() == 3.0));
        }
    }

    A = 4.0;
    B = 5.0;
    C = 6.0;

    hip_catch(hipDeviceSynchronize());

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            CHECK((A(i, j).get() == 4.0));
            CHECK((B(i, j).get() == 5.0));
            CHECK((C(i, j).get() == 6.0));
        }
    }

    hip_catch(hipDeviceSynchronize());
}

TEST_CASE("Check memory", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    hip_catch(hipDeviceSynchronize());

    hipError_t            status;
    hipPointerAttribute_t attribs;

    status = hipPointerGetAttributes(&attribs, A.gpu_data());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeDevice));

    status = hipPointerGetAttributes(&attribs, A.gpu_dims());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeDevice));

    status = hipPointerGetAttributes(&attribs, A.gpu_strides());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeDevice));

    CHECK((A.data() == nullptr));

    status = hipPointerGetAttributes(&attribs, B.gpu_data());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeHost));

    status = hipPointerGetAttributes(&attribs, B.data());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeHost));

    status = hipPointerGetAttributes(&attribs, B.gpu_dims());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeDevice));

    status = hipPointerGetAttributes(&attribs, B.gpu_strides());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeDevice));

    status = hipPointerGetAttributes(&attribs, C.gpu_data());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeHost));

    status = hipPointerGetAttributes(&attribs, C.data());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeHost));

    status = hipPointerGetAttributes(&attribs, C.gpu_dims());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeDevice));

    status = hipPointerGetAttributes(&attribs, C.gpu_strides());

    CHECK((status != hipErrorInvalidValue));
    CHECK((attribs.type == hipMemoryTypeDevice));
}

TEST_CASE("Test reading and writing vectors", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    std::vector<double> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    A.read(data);
    B.read(data);
    C.read(data);

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A(i, j) == (double)(i * 3 + j + 1)));
            CHECK((B(i, j) == (double)(i * 3 + j + 1)));
            CHECK((C(i, j) == (double)(i * 3 + j + 1)));
        }
    }

    A.zero();

    hip_catch(hipDeviceSynchronize());

    A.write(data);

    for (int i = 0; i < 9; i++) {
        CHECK((data[i] == 0));
    }

    B.write(data);

    for (int i = 0; i < 9; i++) {
        CHECK((data[i] == i + 1));
    }

    C.zero();

    hip_catch(hipDeviceSynchronize());

    C.write(data);

    for (int i = 0; i < 9; i++) {
        CHECK((data[i] == 0));
    }
}

TEST_CASE("Test reading and writing arrays", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 2> A("A", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B("B", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C("C", einsums::detail::PINNED, 3, 3);

    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    A.read(data);
    B.read(data);
    C.read(data);

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A(i, j) == (double)(i * 3 + j + 1)));
            CHECK((B(i, j) == (double)(i * 3 + j + 1)));
            CHECK((C(i, j) == (double)(i * 3 + j + 1)));
        }
    }

    A.zero();

    hip_catch(hipDeviceSynchronize());

    A.write(data);

    for (int i = 0; i < 9; i++) {
        CHECK((data[i] == 0));
    }

    B.write(data);

    for (int i = 0; i < 9; i++) {
        CHECK((data[i] == i + 1));
    }

    C.zero();

    hip_catch(hipDeviceSynchronize());

    C.write(data);

    for (int i = 0; i < 9; i++) {
        CHECK((data[i] == 0));
    }
}

TEST_CASE("Test tensor assignments", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 2> A0("A0", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B0("B0", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C0("C0", einsums::detail::PINNED, 3, 3);
    DeviceTensor<double, 2> A1("A1", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B1("B1", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C1("C1", einsums::detail::PINNED, 3, 3);
    DeviceTensor<double, 2> A2("A2", einsums::detail::DEV_ONLY, 5, 5);
    DeviceTensor<double, 2> B2("B2", einsums::detail::MAPPED, 5, 5);
    DeviceTensor<double, 2> C2("C2", einsums::detail::PINNED, 5, 5);
    DeviceTensor<float, 2>  A3("A3", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<float, 2>  B3("B3", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<float, 2>  C3("C3", einsums::detail::PINNED, 3, 3);
    DeviceTensor<float, 2>  A4("A4", einsums::detail::DEV_ONLY, 5, 5);
    DeviceTensor<float, 2>  B4("B4", einsums::detail::MAPPED, 5, 5);
    DeviceTensor<float, 2>  C4("C4", einsums::detail::PINNED, 5, 5);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A1(i, j) = (double)(3 * i + j + 1);
            B1(i, j) = 10.0 * (3 * i + j + 1);
            C1(i, j) = 20.0 * (3 * i + j + 1);
            A3(i, j) = 60.0f * (3 * i + j + 1);
            B3(i, j) = 70.0f * (3 * i + j + 1);
            C3(i, j) = 80.0f * (3 * i + j + 1);
        }
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            A2(i, j) = 30.0 * (5 * i + j + 1);
            B2(i, j) = 40.0 * (5 * i + j + 1);
            C2(i, j) = 50.0 * (5 * i + j + 1);
            A4(i, j) = 90.0f * (5 * i + j + 1);
            B4(i, j) = 100.0f * (5 * i + j + 1);
            C4(i, j) = 110.0f * (5 * i + j + 1);
        }
    }

    hip_catch(hipDeviceSynchronize());

    // Test the various copies of the same size, but with different locations.
    A0 = A1;
    B0 = B1;
    C0 = C1;

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A0(i, j) == A1(i, j)));
            CHECK((B0(i, j) == B1(i, j)));
            CHECK((C0(i, j) == C1(i, j)));
        }
    }

    A0 = B1;
    B0 = C1;
    C0 = A1;

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A0(i, j) == B1(i, j)));
            CHECK((B0(i, j) == C1(i, j)));
            CHECK((C0(i, j) == A1(i, j)));
        }
    }

    A0 = C1;
    B0 = A1;
    C0 = B1;

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A0(i, j) == C1(i, j)));
            CHECK((B0(i, j) == A1(i, j)));
            CHECK((C0(i, j) == B1(i, j)));
        }
    }

    // Test the copies of different size.
    A0 = A2;
    B0 = B2;
    C0 = C2;

    hip_catch(hipDeviceSynchronize());

    CHECK((A0.dim(0) == 5 && A0.dim(1) == 5));
    CHECK((B0.dim(0) == 5 && B0.dim(1) == 5));
    CHECK((C0.dim(0) == 5 && C0.dim(1) == 5));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK((A0(i, j) == A2(i, j)));
            CHECK((B0(i, j) == B2(i, j)));
            CHECK((C0(i, j) == C2(i, j)));
        }
    }

    // reset.
    A0 = A1;
    B0 = B1;
    C0 = C1;

    hip_catch(hipDeviceSynchronize());

    A0 = B2;
    B0 = C2;
    C0 = A2;

    hip_catch(hipDeviceSynchronize());

    CHECK((A0.dim(0) == 5 && A0.dim(1) == 5));
    CHECK((B0.dim(0) == 5 && B0.dim(1) == 5));
    CHECK((C0.dim(0) == 5 && C0.dim(1) == 5));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK((A0(i, j) == B2(i, j)));
            CHECK((B0(i, j) == C2(i, j)));
            CHECK((C0(i, j) == A2(i, j)));
        }
    }

    // reset.
    A0 = A1;
    B0 = B1;
    C0 = C1;

    hip_catch(hipDeviceSynchronize());

    A0 = C2;
    B0 = A2;
    C0 = B2;

    hip_catch(hipDeviceSynchronize());

    CHECK((A0.dim(0) == 5 && A0.dim(1) == 5));
    CHECK((B0.dim(0) == 5 && B0.dim(1) == 5));
    CHECK((C0.dim(0) == 5 && C0.dim(1) == 5));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK((A0(i, j) == C2(i, j)));
            CHECK((B0(i, j) == A2(i, j)));
            CHECK((C0(i, j) == B2(i, j)));
        }
    }

    // reset.
    A0 = A1;
    B0 = B1;
    C0 = C1;

    hip_catch(hipDeviceSynchronize());

    // Test the converting copies of the same size.
    A0 = A3;
    B0 = B3;
    C0 = C3;

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A0(i, j) == A3(i, j)));
            CHECK((B0(i, j) == B3(i, j)));
            CHECK((C0(i, j) == C3(i, j)));
        }
    }

    A0 = B3;
    B0 = C3;
    C0 = A3;

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A0(i, j) == B3(i, j)));
            CHECK((B0(i, j) == C3(i, j)));
            CHECK((C0(i, j) == A3(i, j)));
        }
    }

    A0 = C3;
    B0 = A3;
    C0 = B3;

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((A0(i, j) == C3(i, j)));
            CHECK((B0(i, j) == A3(i, j)));
            CHECK((C0(i, j) == B3(i, j)));
        }
    }

    // Test the converting copies of different sizes.
    A0 = A4;
    B0 = B4;
    C0 = C4;

    hip_catch(hipDeviceSynchronize());

    CHECK((A0.dim(0) == 5 && A0.dim(1) == 5));
    CHECK((B0.dim(0) == 5 && B0.dim(1) == 5));
    CHECK((C0.dim(0) == 5 && C0.dim(1) == 5));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK((A0(i, j) == A4(i, j)));
            CHECK((B0(i, j) == B4(i, j)));
            CHECK((C0(i, j) == C4(i, j)));
        }
    }

    // reset.
    A0 = A1;
    B0 = B1;
    C0 = C1;

    hip_catch(hipDeviceSynchronize());

    A0 = B4;
    B0 = C4;
    C0 = A4;

    hip_catch(hipDeviceSynchronize());

    CHECK((A0.dim(0) == 5 && A0.dim(1) == 5));
    CHECK((B0.dim(0) == 5 && B0.dim(1) == 5));
    CHECK((C0.dim(0) == 5 && C0.dim(1) == 5));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK((A0(i, j) == B4(i, j)));
            CHECK((B0(i, j) == C4(i, j)));
            CHECK((C0(i, j) == A4(i, j)));
        }
    }

    // reset.
    A0 = A1;
    B0 = B1;
    C0 = C1;

    hip_catch(hipDeviceSynchronize());

    A0 = C4;
    B0 = A4;
    C0 = B4;

    hip_catch(hipDeviceSynchronize());

    CHECK((A0.dim(0) == 5 && A0.dim(1) == 5));
    CHECK((B0.dim(0) == 5 && B0.dim(1) == 5));
    CHECK((C0.dim(0) == 5 && C0.dim(1) == 5));

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            CHECK((A0(i, j) == C4(i, j)));
            CHECK((B0(i, j) == A4(i, j)));
            CHECK((C0(i, j) == B4(i, j)));
        }
    }
}

TEST_CASE("Test tensor arithmetic assignments", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 2> A0("A0", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B0("B0", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C0("C0", einsums::detail::PINNED, 3, 3);
    DeviceTensor<double, 2> A1("A1", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B1("B1", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C1("C1", einsums::detail::PINNED, 3, 3);
    DeviceTensor<double, 2> A0_orig("A0", einsums::detail::DEV_ONLY, 3, 3);
    DeviceTensor<double, 2> B0_orig("B0", einsums::detail::MAPPED, 3, 3);
    DeviceTensor<double, 2> C0_orig("C0", einsums::detail::PINNED, 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A0(i, j)      = (double)(3 * i + j + 1);
            B0(i, j)      = 10.0 * (3 * i + j + 1);
            C0(i, j)      = 20.0 * (3 * i + j + 1);
            A0_orig(i, j) = (double)(3 * i + j + 1);
            B0_orig(i, j) = 10.0 * (3 * i + j + 1);
            C0_orig(i, j) = 20.0 * (3 * i + j + 1);
            A1(i, j)      = 60.0f * (3 * i + j + 1);
            B1(i, j)      = 70.0f * (3 * i + j + 1);
            C1(i, j)      = 80.0f * (3 * i + j + 1);
        }
    }

    hip_catch(hipDeviceSynchronize());

    SECTION("Multiplication") {
        A0 *= A1;
        B0 *= B1;
        C0 *= C1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A1(i, j) * A0_orig(i, j)));
                CHECK((B0(i, j) == B1(i, j) * B0_orig(i, j)));
                CHECK((C0(i, j) == C1(i, j) * C0_orig(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 *= B1;
        B0 *= C1;
        C0 *= A1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == B1(i, j) * A0_orig(i, j)));
                CHECK((B0(i, j) == C1(i, j) * B0_orig(i, j)));
                CHECK((C0(i, j) == A1(i, j) * C0_orig(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 *= C1;
        B0 *= A1;
        C0 *= B1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == C1(i, j) * A0_orig(i, j)));
                CHECK((B0(i, j) == A1(i, j) * B0_orig(i, j)));
                CHECK((C0(i, j) == B1(i, j) * C0_orig(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());
    }

    SECTION("Addition") {
        A0 += A1;
        B0 += B1;
        C0 += C1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A1(i, j) + A0_orig(i, j)));
                CHECK((B0(i, j) == B1(i, j) + B0_orig(i, j)));
                CHECK((C0(i, j) == C1(i, j) + C0_orig(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 += B1;
        B0 += C1;
        C0 += A1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == B1(i, j) + A0_orig(i, j)));
                CHECK((B0(i, j) == C1(i, j) + B0_orig(i, j)));
                CHECK((C0(i, j) == A1(i, j) + C0_orig(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 += C1;
        B0 += A1;
        C0 += B1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == C1(i, j) + A0_orig(i, j)));
                CHECK((B0(i, j) == A1(i, j) + B0_orig(i, j)));
                CHECK((C0(i, j) == B1(i, j) + C0_orig(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());
    }

    SECTION("Division") {
        A0 /= A1;
        B0 /= B1;
        C0 /= C1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) / A1(i, j)));
                CHECK((B0(i, j) == B0_orig(i, j) / B1(i, j)));
                CHECK((C0(i, j) == C0_orig(i, j) / C1(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 /= B1;
        B0 /= C1;
        C0 /= A1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) / B1(i, j)));
                CHECK((B0(i, j) == B0_orig(i, j) / C1(i, j)));
                CHECK((C0(i, j) == C0_orig(i, j) / A1(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 /= C1;
        B0 /= A1;
        C0 /= B1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) / C1(i, j)));
                CHECK((B0(i, j) == B0_orig(i, j) / A1(i, j)));
                CHECK((C0(i, j) == C0_orig(i, j) / B1(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());
    }

    SECTION("Subtraction") {
        A0 -= A1;
        B0 -= B1;
        C0 -= C1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) - A1(i, j)));
                CHECK((B0(i, j) == B0_orig(i, j) - B1(i, j)));
                CHECK((C0(i, j) == C0_orig(i, j) - C1(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 -= B1;
        B0 -= C1;
        C0 -= A1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) - B1(i, j)));
                CHECK((B0(i, j) == B0_orig(i, j) - C1(i, j)));
                CHECK((C0(i, j) == C0_orig(i, j) - A1(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 -= C1;
        B0 -= A1;
        C0 -= B1;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) - C1(i, j)));
                CHECK((B0(i, j) == B0_orig(i, j) - A1(i, j)));
                CHECK((C0(i, j) == C0_orig(i, j) - B1(i, j)));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());
    }

    SECTION("Scalar operations") {
        A0 *= 2.0;
        B0 *= 3.0;
        C0 *= 4.0;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) * 2.0));
                CHECK((B0(i, j) == B0_orig(i, j) * 3.0));
                CHECK((C0(i, j) == C0_orig(i, j) * 4.0));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 += 2.0;
        B0 += 3.0;
        C0 += 4.0;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) + 2.0));
                CHECK((B0(i, j) == B0_orig(i, j) + 3.0));
                CHECK((C0(i, j) == C0_orig(i, j) + 4.0));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 /= 2.0;
        B0 /= 3.0;
        C0 /= 4.0;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) / 2.0));
                CHECK((B0(i, j) == B0_orig(i, j) / 3.0));
                CHECK((C0(i, j) == C0_orig(i, j) / 4.0));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());

        A0 -= 2.0;
        B0 -= 3.0;
        C0 -= 4.0;

        hip_catch(hipDeviceSynchronize());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                CHECK((A0(i, j) == A0_orig(i, j) - 2.0));
                CHECK((B0(i, j) == B0_orig(i, j) - 3.0));
                CHECK((C0(i, j) == C0_orig(i, j) - 4.0));
            }
        }

        A0 = A0_orig;
        B0 = B0_orig;
        C0 = C0_orig;

        hip_catch(hipDeviceSynchronize());
    }
}

TEST_CASE("Test device-host interfacing", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    Tensor<double, 2> host_tensor("Host", 3, 3);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            host_tensor(i, j) = 3 * i + j + 1;
        }
    }

    DeviceTensor<double, 2> A(host_tensor, einsums::detail::DEV_ONLY);

    hip_catch(hipDeviceSynchronize());

    host_tensor *= 10;

    DeviceTensor<double, 2> B(host_tensor, einsums::detail::PINNED);

    hip_catch(hipDeviceSynchronize());

    host_tensor *= 2;

    DeviceTensor<double, 2> C(host_tensor, einsums::detail::MAPPED);

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK((double) A(i, j) == 3 * i + j + 1);
            CHECK((double) B(i, j) == 10 * (3 * i + j + 1));
            CHECK((double) C(i, j) == 20 * (3 * i + j + 1));
        }
    }

    auto A_host = (Tensor<double, 2>)A;
    auto B_host = (Tensor<double, 2>)B;
    auto C_host = (Tensor<double, 2>)C;

    hip_catch(hipDeviceSynchronize());

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            CHECK(A_host(i, j) == 3 * i + j + 1);
            CHECK(B_host(i, j) == 10 * (3 * i + j + 1));
            CHECK(C_host(i, j) == 20 * (3 * i + j + 1));
        }
    }
}

TEST_CASE("Zero-rank GPU tensors", "[gpu][tensor]") {
    using namespace einsums;
    using namespace einsums::detail;
    using namespace einsums::gpu;
    // using namespace einsums::gpu::detail;

    DeviceTensor<double, 0> A("A", DEV_ONLY);
    DeviceTensor<double, 0> B("B", PINNED);
    DeviceTensor<double, 0> C("C", MAPPED);

    hip_catch(hipDeviceSynchronize());

    DeviceTensor<double, 0> A_copy(A);
    DeviceTensor<double, 0> B_copy(B);
    DeviceTensor<double, 0> C_copy(C);

    hip_catch(hipDeviceSynchronize());

    hipError_t            status;
    hipPointerAttribute_t attribs;

    REQUIRE_NOTHROW(hip_catch(hipPointerGetAttributes(&attribs, A.gpu_data())));

    CHECK((attribs.type == hipMemoryTypeDevice));

    CHECK((A.data() == nullptr));

    REQUIRE_NOTHROW(hip_catch(hipPointerGetAttributes(&attribs, B.gpu_data())));

    CHECK((attribs.type == hipMemoryTypeHost));

    REQUIRE_NOTHROW(hip_catch(hipPointerGetAttributes(&attribs, B.data())));

    CHECK((attribs.type == hipMemoryTypeHost));

    REQUIRE_NOTHROW(hip_catch(hipPointerGetAttributes(&attribs, C.gpu_data())));

    CHECK((attribs.type == hipMemoryTypeHost));

    REQUIRE_NOTHROW(hip_catch(hipPointerGetAttributes(&attribs, C.data())));

    CHECK((attribs.type == hipMemoryTypeHost));

    A = 1.0;
    B = 2.0;
    C = 3.0;

    hip_catch(hipDeviceSynchronize());

    CHECK(((double)A == 1.0));
    CHECK(((double)B == 2.0));
    CHECK(((double)C == 3.0));

    A = B;
    B = C;

    hip_catch(hipDeviceSynchronize());

    CHECK(((double)A == 2.0));
    CHECK(((double)B == 3.0));

    A *= 2.0;
    B *= 2.0;
    C *= 2.0;

    hip_catch(hipDeviceSynchronize());

    CHECK(((double)A == 4.0));
    CHECK(((double)B == 6.0));
    CHECK(((double)C == 6.0));

    A += 2.0;
    B += 2.0;
    C += 2.0;

    hip_catch(hipDeviceSynchronize());

    CHECK(((double)A == 6.0));
    CHECK(((double)B == 8.0));
    CHECK(((double)C == 8.0));

    A /= 2.0;
    B /= 2.0;
    C /= 2.0;

    hip_catch(hipDeviceSynchronize());

    CHECK(((double)A == 3.0));
    CHECK(((double)B == 4.0));
    CHECK(((double)C == 4.0));

    A -= 2.0;
    B -= 2.0;
    C -= 2.0;

    hip_catch(hipDeviceSynchronize());

    CHECK(((double)A == 1.0));
    CHECK(((double)B == 2.0));
    CHECK(((double)C == 2.0));

    Tensor<double, 0> D("D");

    D = 10.0;

    hip_catch(hipDeviceSynchronize());

    DeviceTensor<double, 0> E(D, DEV_ONLY);
    DeviceTensor<double, 0> F(D, MAPPED);
    DeviceTensor<double, 0> G(D, PINNED);

    hip_catch(hipDeviceSynchronize());

    CHECK(((double)E == 10.0));
    CHECK(((double)F == 10.0));
    CHECK(((double)G == 10.0));

    CHECK(((double)(Tensor<double, 0>)E == 10.0));
    CHECK(((double)(Tensor<double, 0>)F == 10.0));
    CHECK(((double)(Tensor<double, 0>)G == 10.0));
}
