//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <Einsums/Testing.hpp>

TEMPLATE_TEST_CASE("Khatri-Rao", "[tensor_algebra]", float, double, std::complex<float>, std::complex<double>) {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    int const _I{8}, _M{4}, _r{16};

    SECTION("einsum") {

        auto KR  = create_tensor<TestType>("KR", _I, _M, _r);
        auto KR0 = create_tensor<TestType>("KR0", _I, _M, _r);

        auto T = create_random_tensor<TestType>("T", _I, _r);
        auto U = create_random_tensor<TestType>("U", _M, _r);

        einsum(Indices{I, M, r}, &KR, Indices{I, r}, T, Indices{M, r}, U);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    REQUIRE_THAT(KR(x, y, z), CheckWithinRel(KR0(x, y, z), 0.0001));
                    // REQUIRE_THAT(KR(x, y, z), Catch::Matchers::WithinAbs(KR0(x, y, z), 0.000001));
                }
            }
        }
    }

    SECTION("special function") {
        auto KR0 = create_tensor<TestType>("KR0", _I, _M, _r);

        auto T = create_random_tensor<TestType>("T", _I, _r);
        auto U = create_random_tensor<TestType>("U", _M, _r);

        auto KR = khatri_rao(Indices{I, r}, T, Indices{M, r}, U);
        // println(result);

        for (int x = 0; x < _I; x++) {
            for (int y = 0; y < _M; y++) {
                for (int z = 0; z < _r; z++) {
                    KR0(x, y, z) = T(x, z) * U(y, z);
                }
            }
        }

        auto KR0_view = TensorView{KR0, Dim{_I * _M, _r}};

        for (int x = 0; x < _I * _M; x++) {
            for (int z = 0; z < _r; z++) {
                REQUIRE_THAT(KR(x, z), CheckWithinRel(KR0_view(x, z), 0.0001));
                // REQUIRE_THAT(KR(x, z), Catch::Matchers::WithinAbs(KR0_view(x, z), 0.000001));
            }
        }
    }
}
