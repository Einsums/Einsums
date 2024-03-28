#include "einsums/psi4/TensorConvert.hpp"
#include "einsums/_Common.hpp"

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::psi4)

Tensor<double, 2> matrix_to_tensor(psi::SharedMatrix mat) {
    auto out = Tensor<double, 2>(mat->name(), mat->nrow(), mat->ncol());

    
}

END_EINSUMS_NAMESPACE_CPP(einsums::psi4)