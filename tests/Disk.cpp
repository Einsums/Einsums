#include "einsums/LinearAlgebra.hpp"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Timer.hpp"
#include "einsums/Utilities.hpp"
#include "range/v3/view/cartesian_product.hpp"

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <filesystem>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

TEST_CASE("disktensor-creation", "[disktensor]") {
    using namespace einsums;

    SECTION("double") {
        DiskTensor A(state::data, "/A0", 3, 3);
        DiskTensor B(state::data, "/B0", 3, 3);
        DiskTensor C(state::data, "/C0", 3, 3);

        REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
        REQUIRE((B.dim(0) == 3 && B.dim(1) == 3));
        REQUIRE((C.dim(0) == 3 && C.dim(1) == 3));
    }

    SECTION("float") {
        auto A = create_disk_tensor<float>(state::data, "/A1", 3, 3);
        REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    }

    // Complex datatypes currently not supported.  It should be able to handle this through defining a HDF5 compound datatype.
    // SECTION("complex<double>") {
    //     auto A = create_disk_tensor<std::complex<double>>(state::data, "/A2", 3, 3);
    //     REQUIRE((A.dim(0) == 3 && A.dim(1) == 3));
    // }
}

#if 0
TEST_CASE("Write/Read", "[disktensor]") {
    using namespace einsums;

    DiskTensor<2> A(state::data, "/A1", 3, 3);

    // Data must exist on disk before it can be read in.
    Tensor<2> Ad = create_random_tensor("A", 3, 3);
    // println(Ad);
    A._write(Ad);

    auto suba = A(0, All);
    // println(suba);
    REQUIRE(suba(0) == Ad(0, 0));
    REQUIRE(suba(1) == Ad(0, 1));
    REQUIRE(suba(2) == Ad(0, 2));

    auto subb = A(1, All);
    // println(subb);
    REQUIRE(subb(0) == Ad(1, 0));
    REQUIRE(subb(1) == Ad(1, 1));
    REQUIRE(subb(2) == Ad(1, 2));

    auto subc = A(All, 1);
    // println(subc);
    REQUIRE(subc(0) == Ad(0, 1));
    REQUIRE(subc(1) == Ad(1, 1));
    REQUIRE(subc(2) == Ad(2, 1));

    auto subd = A(Range{0, 2}, All);
    // println(subd);
    REQUIRE((subd.dim(0) == 2 && subd.dim(1) == 3));
    REQUIRE(subd(0, 0) == Ad(0, 0));
    REQUIRE(subd(0, 1) == Ad(0, 1));
    REQUIRE(subd(0, 2) == Ad(0, 2));
    REQUIRE(subd(1, 0) == Ad(1, 0));
    REQUIRE(subd(1, 1) == Ad(1, 1));
    REQUIRE(subd(1, 2) == Ad(1, 2));

    auto tempe = A(All, Range{1, 3});
    auto &sube = tempe.get();
    REQUIRE((sube.dim(0) == 3 && sube.dim(1) == 2));
    REQUIRE(sube(0, 0) == Ad(0, 1));
    REQUIRE(sube(0, 1) == Ad(0, 2));
    REQUIRE(sube(1, 0) == Ad(1, 1));
    REQUIRE(sube(1, 1) == Ad(1, 2));
    REQUIRE(sube(2, 0) == Ad(2, 1));
    REQUIRE(sube(2, 1) == Ad(2, 2));
}

TEST_CASE("DiskView 3x3", "[disktensor]") {
    using namespace einsums;

    DiskTensor<2> A(state::data, "/A2", 3, 3);

    // Data must exist on disk before it can be read in.
    Tensor<2> Ad = create_random_tensor("A", 3, 3);
    // println(Ad);
    A._write(Ad);

    {
        // Obtaining a DiskView does not automatically allocate memory.
        auto suba = A(2, All);

        Tensor<1> tempa{"tempa", 3};
        tempa(0) = 1.0;
        tempa(1) = 2.0;
        tempa(2) = 3.0;

        // Perform a write setting the data on disk to tempa
        suba = tempa;

        // Perform a read
        auto &tempb = suba.get();

        // Test
        REQUIRE(tempb(0) == 1.0);
        REQUIRE(tempb(1) == 2.0);
        REQUIRE(tempb(2) == 3.0);
    }

    {
        auto suba = A(2, Range{1, 3});

        Tensor<1> tempa{"tempb", 2};
        tempa(0) = 4.0;
        tempa(1) = 5.0;

        suba = tempa;

        // Perform a read
        auto &tempb = suba.get();

        REQUIRE(tempb(0) == 4.0);
        REQUIRE(tempb(1) == 5.0);
    }

    {
        auto suba = A(Range{1, 3}, Range{0, 2});

        Tensor<2> tempa{"tempa", 2, 2};
        tempa(0, 0) = 10.0;
        tempa(0, 1) = 11.0;
        tempa(1, 0) = 12.0;
        tempa(1, 1) = 13.0;

        suba = tempa;

        auto &tempb = suba.get();

        REQUIRE(tempb(0, 0) == 10.0);
        REQUIRE(tempb(0, 1) == 11.0);
        REQUIRE(tempb(1, 0) == 12.0);
        REQUIRE(tempb(1, 1) == 13.0);
    }
}
#endif

TEST_CASE("DiskView 7x7x7x7", "[disktensor]") {
    using namespace einsums;

    SECTION("Write [7,7] data to [:,2,4,:]") {
        DiskTensor g(state::data, "g0", 7, 7, 7, 7);
        Tensor     data   = create_random_tensor("data", 7, 7);
        g(All, 2, 4, All) = data;
    }

    SECTION("Write [7,2,7] data to [:,4-5,2,:]") {
        DiskTensor g(state::data, "g1", 7, 7, 7, 7);
        Tensor     data2            = create_random_tensor("data", 7, 2, 7);
        g(All, Range{4, 6}, 2, All) = data2;
    }

    SECTION("Write/Read [7,7] data to/from [2,2,:,:]") {
        DiskTensor g(state::data, "g2", 3, 3, 3, 3);
        Tensor     data3 = create_random_tensor("data", 3, 3);
        double     value = 0.0;

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {

                for (size_t k = 0; k < 3; k++) {
                    for (size_t l = 0; l < 3; l++) {
                        data3(k, l) = value;
                        value += 1.0;
                    }
                }

                auto  diskView = g(i, j, All, All);
                auto &tensor   = diskView.get();

                tensor = data3;

                diskView.put();
            }
        }
    }
}
