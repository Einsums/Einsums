#include <catch2/catch_all.hpp>

#include "einsums.hpp"

using namespace einsums;

TEST_CASE("Runtime Tensor Assignment") {
    RuntimeTensor<double> A{"A", std::vector<size_t>{10, 10}};
    A = create_random_tensor("A", 10, 10);
    RuntimeTensor<double> C{std::vector<size_t>{10, 10}};
    C                       = create_random_tensor("C", 10, 10);
    RuntimeTensor<double> D = create_random_tensor("D", 20, 20);
    RuntimeTensor<double> E;
    RuntimeTensor<double> F = C;
    E                       = A;
    auto D_view             = D(Range{0, 10}, Range{0, 10});

    REQUIRE(A.rank() == 2);

    Tensor<double, 4> B_base = create_random_tensor("B", 10, 10, 10, 10);

    RuntimeTensor<double> B = (RuntimeTensor<double>)B_base(Range{0, 5}, Range{1, 6}, Range{2, 7}, Range{3, 8});

    REQUIRE(B.rank() == 4);

    REQUIRE(A.data() != nullptr);
    REQUIRE(B.data() != nullptr);
    REQUIRE(C.data() != nullptr);
    REQUIRE(D.data() != nullptr);
    REQUIRE((E.data() != nullptr && E.data() != A.data()));
    REQUIRE((F.data() != nullptr && F.data() != C.data()));

    REQUIRE(A.data(std::array<ptrdiff_t, 2>{-1, 1}) != nullptr);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 5; k++) {
                for (int l = 0; l < 5; l++) {
                    REQUIRE(B(std::vector<ptrdiff_t>{i, j, k, l}) == B_base(i, j + 1, k + 2, l + 3));
                }
            }
        }
    }

    A = C;

    const auto &C_const = *&C;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(A(i, j) == std::get<double>(C_const(i, j)));
        }
    }

    A                   = D_view;
    const auto &D_const = *&D;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            REQUIRE(A(std::array<int, 2>{i, j}) == D_const(std::array<int, 2>{i, j}));
        }
    }

    D_view.zero();

    D_view = 1.0;

    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 20; j++) {
            if (i < 10 && j < 10) {
                REQUIRE(D(i, j) == 1.0);
            }
        }
    }
}