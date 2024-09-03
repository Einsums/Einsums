#pragma once

#include "einsums/_Common.hpp"
#include <pybind11/pybind11.h>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::python)

namespace detail {
enum PyPlanDataType { FLOAT, DOUBLE, LONG_DOUBLE, COMPLEX_FLOAT, COMPLEX_DOUBLE, COMPLEX_LONG_DOUBLE };

inline size_t sizeof_plantype(PyPlanDataType type) {
    switch (type) {
    case FLOAT:
        return sizeof(float);
    case DOUBLE:
        return sizeof(double);
    case LONG_DOUBLE:
        return sizeof(long double);
    case COMPLEX_FLOAT:
        return sizeof(std::complex<float>);
    case COMPLEX_DOUBLE:
        return sizeof(std::complex<double>);
    case COMPLEX_LONG_DOUBLE:
        return sizeof(std::complex<long double>);
    }
}

inline PyPlanDataType get_larger_type(PyPlanDataType A_datatype, PyPlanDataType B_datatype) {
    if (sizeof_plantype(A_datatype) > sizeof_plantype(B_datatype)) {
        return A_datatype;
    }
    return B_datatype;
}

enum PyPlanTensorType { SCALAR, BASIC, BLOCK, TILED };

enum PyPlanUnit { CPU, GPU };

} // namespace detail

EINSUMS_EXPORT void export_tensor_algebra(pybind11::module_ &m);

END_EINSUMS_NAMESPACE_HPP(einsums::python)

#include "einsums/python/PyTensorAlgebra.hpp"