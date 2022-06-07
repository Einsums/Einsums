#include "einsums/Decomposition.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"
#include "einsums/LinearAlgebra.hpp"

#include <catch2/catch.hpp>

TEST_CASE("CP 1") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::decomposition;

    Tensor<3, double> test1("test 1", 3, 3, 3);
    test1.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>
                            {0.94706517, 0.3959549, 0.14122476, 0.83665482, 0.27340639, 0.29811429,
                             0.1823041, 0.66556282, 0.73178046, 0.72504222, 0.58360409, 0.68301135,
                             0.8316929, 0.66955444, 0.25182224, 0.24108674, 0.09582611, 0.93056666,
                             0.60919366, 0.97363788, 0.24531965, 0.23757898, 0.43426057, 0.64763913,
                             0.61224901, 0.86068415, 0.12051599};

    auto factors = parafac(test1, 2, 10, 1.0e-6);

    CHECK_THAT(factors[0].vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                            1.52094859, 0.25583807, 1.74374003, 0.79559132, 1.498266, -0.50882886}));

    CHECK_THAT(factors[1].vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                            -0.66754524, -0.26756225, -0.5229646, 0.41945054, -0.54818362, -0.92958005}));
    
    CHECK_THAT(factors[2].vector_data(), Catch::Matchers::Equals(std::vector<double, einsums::AlignedAllocator<double, 64>>{
                                            -0.66634092, 0.53102556, -0.62631181, 0.57979445, -0.44226493, -0.69426301}));
}