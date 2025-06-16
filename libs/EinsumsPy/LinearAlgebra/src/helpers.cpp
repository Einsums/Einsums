#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

int determine_easy_vector(pybind11::buffer const &X, size_t *easy_elems, size_t *incx, size_t *hard_elems,
                           std::vector<size_t> *X_strides, std::vector<size_t> *hard_index_strides) {
    py::buffer_info X_info = X.request(false);

    int    easy_rank   = -1;
    size_t total_elems = 1;

    *easy_elems = 1;
    *hard_elems = 1;

    *incx = X_info.strides[X_info.ndim - 1] / X_info.itemsize;

    X_strides->resize(X_info.ndim);
    hard_index_strides->resize(X_info.ndim);

    for (int i = X_info.ndim - 1; i >= 0; i--) {
        (*hard_index_strides)[i] = total_elems;
        (*X_strides)[i]       = X_info.strides[i] / X_info.itemsize;
        total_elems *= X_info.shape[i];

        if (total_elems * *incx == (*X_strides)[i]) {
            *easy_elems *= X_info.shape[i];
            easy_rank = i;
        } else {
            *hard_elems *= X_info.shape[i];
        }
    }

    if (easy_rank == 0) {
        *hard_elems = 0;
    }

    return easy_rank;
}

void recalc_index_strides(std::vector<size_t> *index_strides, int new_rank) {
    index_strides->resize(new_rank);

    size_t div_val = index_strides->at(index_strides->size() - 1);

    for(int i = 0; i < index_strides->size(); i++) {
        index_strides->at(i) /= div_val;
    }
}

} // namespace detail
} // namespace python
} // namespace einsums