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

    Tensor<3, double> test1_cp = parafac_reconstruct<3, double>(factors);

    double diff = rmsd(test1, test1_cp);

    REQUIRE((rmsd(test1, test1_cp) < 0.17392));
}