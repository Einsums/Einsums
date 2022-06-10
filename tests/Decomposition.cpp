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

    REQUIRE((rmsd(test1, test1_cp) < 0.17392));
}

TEST_CASE("CP 2") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::decomposition;

    Tensor<3, double> test2("test 2", 3, 4, 2);
    
    test2.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>
                            {0.29945093, 0.0090937, 0.99788559, 0.0821231, 0.29625705, 0.80278977,
                             0.15189681, 0.35832086, 0.09648153, 0.39398175, 0.49662056, 0.83101396,
                             0.84288292, 0.48603425, 0.93286471, 0.47101289, 0.32736096, 0.50067919,
                             0.49932342, 0.91922942, 0.44777189, 0.23009644, 0.34874549, 0.19356636};

    auto factors = parafac(test2, 2, 10, 1.0e-6);

    Tensor<3, double> test2_cp = parafac_reconstruct<3, double>(factors);

    REQUIRE((rmsd(test2, test2_cp) <= 0.122492));
}

TEST_CASE("CP 3") {
    using namespace einsums;
    using namespace einsums::tensor_algebra;
    using namespace einsums::decomposition;

    Tensor<4, double> test3("test 3", 3, 2, 3, 2);
    
    test3.vector_data() = std::vector<double, einsums::AlignedAllocator<double, 64>>
                            {0.37001224, 0.77676895, 0.17589323, 0.02762156, 0.21037116, 0.83686174,
                             0.35042434, 0.19117270, 0.58095640, 0.99220655, 0.33536840, 0.15210615,
                             0.95033534, 0.73212124, 0.31346639, 0.83961596, 0.15418801, 0.58927303,
                             0.46744825, 0.44001279, 0.50372353, 0.09696069, 0.96449749, 0.71151666,
                             0.72334792, 0.98646368, 0.13764230, 0.95949904, 0.07774470, 0.18239083,
                             0.82591821, 0.40939436, 0.22088749, 0.90281597, 0.37465773, 0.02541923};

    auto factors = parafac(test3, 2, 10, 1.0e-6);

    Tensor<4, double> test2_cp = parafac_reconstruct<4, double>(factors);

    REQUIRE((rmsd(test3, test2_cp) <= 0.228199));
}