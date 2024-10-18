#include <catch2/catch_all.hpp>

#include "einsums.hpp"

using namespace einsums;

TEST_CASE("Runtime Tensor Assignment") {
    RuntimeTensor<double> A = create_random_tensor("A", 10, 10);
    RuntimeTensor<double> C = create_random_tensor("C", 10, 10);
    RuntimeTensor<double> D = create_random_tensor("D", 20, 20);
    auto D_view = D(Range{0, 10}, Range{0, 10});

    REQUIRE(A.rank() == 2);

    Tensor<double, 4> B_base = create_random_tensor("B", 10, 10, 10, 10);

    RuntimeTensor<double> B = (RuntimeTensor<double>)B_base(Range{0, 5}, Range{1, 6}, Range{2, 7}, Range{3, 8});

    REQUIRE(B.rank() == 4);

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

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            REQUIRE(A(i, j) == C(i, j));
        }
    }

    A = D_view;

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            REQUIRE(A(i, j) == D(i, j));
        }
    }

    D_view.zero();

    D_view = 1.0;
}