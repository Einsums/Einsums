
#include "einsums/DeviceTensor.hpp"
#include "einsums/GPULinearAlgebra.hpp"
#include "src/backends/linear_algebra/hipblas/hipblas.hpp"

#include <catch2/catch_all.hpp>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

#include "catch2/catch_test_macros.hpp"
#include "einsums.hpp"

TEMPLATE_TEST_CASE("GPU Tensor GEMMs", "[gpu][linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::detail;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    gpu::initialize();
    einsums::backend::linear_algebra::hipblas::initialize();

    DeviceTensor<TestType, 2> A("A", DEV_ONLY, 3, 3), B("B", DEV_ONLY, 3, 3), C("C", DEV_ONLY, 3, 3);

    // Input data.
    std::vector<TestType> A_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, B_data{11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};
    // Host vector for comparisons.
    std::vector<TestType> comp_data;
    TestType *C_data = new TestType[9];

    A.read(A_data);
    B.read(B_data);

    einsums::linear_algebra::gpu::gemm<false, false>((TestType) 1.0, A, B, (TestType) 0.0, &C);

    // Issue here
    comp_data = {330.0, 396.0, 462.0, 726.0, 891.0, 1056.0, 1122.0, 1386.0, 1650.0};
    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    einsums::linear_algebra::gpu::gemm<true, false>((TestType) 1.0, A, B, (TestType) 0.0, &C);

    comp_data = {726.0, 858.0, 990.0, 858.0, 1023.0, 1188.0, 990.0, 1188.0, 1386.0};
    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    einsums::linear_algebra::gpu::gemm<false, true>((TestType) 1.0, A, B, (TestType) 0.0, &C);

    comp_data = {154.0, 352.0, 550.0, 352.0, 847.0, 1342.0, 550.0, 1342.0, 2134.0};
    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    einsums::linear_algebra::gpu::gemm<true, true>((TestType) 1.0, A, B, (TestType) 0.0, &C);

    comp_data = {330.0, 726.0, 1122.0, 396.0, 891.0, 1386.0, 462.0, 1056.0, 1650.0};
    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    delete[] C_data;

    einsums::backend::linear_algebra::hipblas::finalize();

}

TEMPLATE_TEST_CASE("GPU dot products", "[gpu][linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::detail;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    gpu::initialize();
    einsums::backend::linear_algebra::hipblas::initialize();

    DeviceTensor<TestType, 2> A("A", DEV_ONLY, 3, 3), B("B", DEV_ONLY, 3, 3);
    DeviceTensor<TestType, 0> C("C", DEV_ONLY);

    // Input data.
    std::vector<TestType> A_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, B_data{11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};

    A.read(A_data);
    B.read(B_data);

    einsums::linear_algebra::gpu::dot((TestType) 0.0, C, (TestType) 1.0, A, B);

    REQUIRE((TestType) C == (TestType) 3135.0);

    einsums::backend::linear_algebra::hipblas::finalize();
}

TEMPLATE_TEST_CASE("GPU outer products", "[gpu][linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::detail;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    gpu::initialize();
    einsums::backend::linear_algebra::hipblas::initialize();

    DeviceTensor<TestType, 1> A("A", DEV_ONLY, 3), B("B", DEV_ONLY, 3);
    DeviceTensor<TestType, 2> C("C", DEV_ONLY, 3, 3);

    // Input data.
    std::vector<TestType> A_data{1.0, 2.0, 3.0}, B_data{11.0, 22.0, 33.0};

    A.read(A_data);
    B.read(B_data);
    C.zero();

    einsums::linear_algebra::gpu::ger((TestType) 1.0, A, B, &C);

    std::vector<TestType> comp_data{11.0, 22.0, 33.0, 22.0, 44.0, 66.0, 33.0, 66.0, 99.0};

    for(int i = 0; i < 9; i++) {
        CHECK(C(i / 3, i % 3).get() == comp_data[i]);
    }

    einsums::backend::linear_algebra::hipblas::finalize();
}

TEMPLATE_TEST_CASE("GPU Matrix vector products", "[gpu][linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::detail;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    gpu::initialize();
    einsums::backend::linear_algebra::hipblas::initialize();

    DeviceTensor<TestType, 2> A("A", DEV_ONLY, 3, 3);
    DeviceTensor<TestType, 1> B("B", DEV_ONLY, 3);
    DeviceTensor<TestType, 1> C("C", DEV_ONLY, 3);

    // Input data.
    std::vector<TestType> A_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, B_data{11.0, 22.0, 33.0};

    A.read(A_data);
    B.read(B_data);
    C.zero();


    einsums::linear_algebra::gpu::gemv<false>((TestType) 1.0, A, B, (TestType) 0.0, &C);

    std::vector<TestType> comp_data{154.0, 352.0, 550.0};

    for(int i = 0; i < 3; i++) {
        CHECK(C(i).get() == comp_data[i]);
    }

    einsums::linear_algebra::gpu::gemv<true>((TestType) 1.0, A, B, (TestType) 0.0, &C);

    comp_data = {330.0,  396.0, 462.0};

    for(int i = 0; i < 3; i++) {
        CHECK(C(i).get() == comp_data[i]);
    }

    einsums::backend::linear_algebra::hipblas::finalize();
}

TEMPLATE_TEST_CASE("GPU scale", "[gpu][linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::detail;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    gpu::initialize();
    einsums::backend::linear_algebra::hipblas::initialize();

    DeviceTensor<TestType, 2> A("A", DEV_ONLY, 3, 3);

    std::vector<TestType> A_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    A.read(A_data);

    einsums::linear_algebra::gpu::scale((TestType) 10.0, &A);

    std::vector<TestType> comp_data{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0};

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            CHECK(A(i, j).get() == comp_data[i * 3 + j]);
        }
    }
}

TEMPLATE_TEST_CASE("Test Symmetric Multiplication", "[gpu][linear-algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::detail;
    using namespace einsums::gpu;
    using namespace einsums::gpu::detail;

    gpu::initialize();
    einsums::backend::linear_algebra::hipblas::initialize();

    DeviceTensor<double, 2> A("A", DEV_ONLY, 3, 3), B("B", DEV_ONLY, 3, 3), C("C", DEV_ONLY, 3, 3);

    // Input data.
    std::vector<double> A_data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, B_data{11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0, 99.0};
    // Host vector for comparisons.
    std::vector<double> comp_data{121968.0, 150282.0, 178596.0, 145926.0, 179685.0, 213444.0, 169884.0, 209088.0, 248292.0};
    double *C_data = new double[9];

    A.read(A_data);
    B.read(B_data);

    einsums::linear_algebra::gpu::symm_gemm<false, false>(A, B, &C);

    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    einsums::linear_algebra::gpu::symm_gemm<true, false>(A, B, &C);

    comp_data = {121968.0, 145926.0, 169884.0, 150282.0, 179685.0, 209088.0, 178596.0, 213444.0, 248292.0};

    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    einsums::linear_algebra::gpu::symm_gemm<false, true>(A, B, &C);

    comp_data = {27588.0, 66792.0, 105996.0, 62436.0, 150645.0, 238854.0, 97284.0, 234498.0, 371712.0};

    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    einsums::linear_algebra::gpu::symm_gemm<true, true>(A, B, &C);
    
    comp_data = {27588.0, 62436.0, 97284.0, 66792.0, 150645.0, 234498.0, 105996.0, 238854.0, 371712.0};

    C.write(C_data);

    for(int i = 0; i < 9; i++) {
        CHECK(C_data[i] == comp_data[i]);
    }

    delete[] C_data;

    einsums::backend::linear_algebra::hipblas::finalize();
}
