#include "einsums/_Common.hpp"

#include "einsums/DeviceTensor.hpp"
#include "einsums/Tensor.hpp"

#include <catch2/catch_all.hpp>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>
#include <type_traits>

#include "catch2/catch_test_macros.hpp"

TEST_CASE("HostDevReference Tests", "[gpu][utilities]") {
    using namespace einsums;
    using namespace einsums::gpu;

    double *host_ptr, *dev_ptr;

    hip_catch(hipMalloc((void **)&dev_ptr, sizeof(double)));

    host_ptr = new double{3.14};

    hip_catch(hipHostRegister((void *)host_ptr, sizeof(double), hipHostRegisterDefault));

    HostDevReference<double> host_ref(host_ptr, true), dev_ref(dev_ptr, false);

    CHECK(host_ref.get() == 3.14);
    host_ref = 0.0;
    CHECK((double)host_ref == 0.0);

    dev_ref = 2.7;
    CHECK(dev_ref.get() == 2.7);

    host_ref = dev_ref;
    CHECK(host_ref.get() == 2.7);

    CHECK(&host_ref == host_ptr);

    hip_catch(hipFree((void *)dev_ptr));
    hip_catch(hipHostUnregister((void *)host_ptr));

    delete host_ptr;
}

TEST_CASE("Index Combinations", "[gpu][utilities]") {
    using namespace einsums;
    using namespace einsums::gpu;

    Tensor<double, 2>     tensor("Tensor", 3, 3);
    std::array<size_t, 2> index;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            tensor(i, j) = i * 3 + j;

            index = {(size_t)i, (size_t)j};

            size_t sentinel = einsums::detail::combination_to_index(index, tensor.dims(), tensor.strides());

            CHECK(sentinel == i * 3 + j);
        }
    }

    for (int i = 0; i < 9; i++) {
        einsums::detail::index_to_combination(i, tensor.dims(), index);
        CHECK(tensor(index[0], index[1]) == i);
    }
}