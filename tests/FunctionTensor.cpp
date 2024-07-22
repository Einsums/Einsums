#include "einsums.hpp"

#include "catch2/catch_all.hpp"

double prod(const std::array<int, 2> &vals) {
    return (vals[0] + 1) * (vals[1] + 1);
}

TEST_CASE("Function Tensor") {
    auto A = einsums::FuncPointerTensor<double, 2>("A", prod, 10, 10);

    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            CHECK_THAT(A(i, j), Catch::Matchers::WithinAbs((i + 1) * (j + 1), 1e-7));
        }
    }
}