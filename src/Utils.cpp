#include "einsums.hpp"

using namespace einsums;

size_t einsums::tensor_algebra::detail::dims_to_strides(const std::vector<size_t> &dims, std::vector<size_t> &out) {
    size_t stride = 1;

    out.resize(dims.size());

    for (int i = dims.size() - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= dims[i];
    }

    return stride;
}