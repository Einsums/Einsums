#include <Einsums/TensorBase/IndexUtilities.hpp>

EINSUMS_EXPORT size_t einsums::dims_to_strides(std::vector<size_t> const &dims, std::vector<size_t> &out) {
    size_t stride = 1;

    out.resize(dims.size());

    for (int i = dims.size() - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= dims[i];
    }

    return stride;
}