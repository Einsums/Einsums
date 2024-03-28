#pragma once


#include "einsums/_Common.hpp"

//#include "psi4/libmints/matrix.h"

namespace psi {
    class SharedMatrix;
}

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::psi4)

EINSUMS_EXPORT Tensor<double, 2> matrix_to_tensor(psi::SharedMatrix mat);

END_EINSUMS_NAMESPACE_HPP(einsums:psi4)