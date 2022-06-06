#include "einsums/Decomposition.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/LinearAlgebra.hpp"

#include <catch2/catch.hpp>

TEST_CASE("CP 1") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::decomposition;

    Tensor<3, double> I_rand = create_random_tensor("I_rand", 3, 4, 2);

    println(I_rand);

    auto factors = parafac(I_rand, 2, 10, 1.0e-6);
}