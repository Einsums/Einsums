#include "einsums/python/PyGPUView.hpp"
#include "einsums/python/PyTensorAlgebra.hpp"

using namespace einsums;
using namespace einsums::python;

std::vector<size_t> einsums::tensor_algebra::detail::get_dim_ranges_for_many(const PyGPUView &C, const std::vector<int> &C_perm, const PyGPUView &A,
                                            const std::vector<int> &A_perm, const PyGPUView &B, const std::vector<int> &B_perm,
                                            int unique_indices) {
    std::vector<size_t> out(unique_indices);
    for (int i = 0; i < unique_indices; i++) {
        out[i] = 0;
    }

    for (int i = 0; i < C_perm.size(); i++) {
        if (out[C_perm[i]] == 0) {
            out[C_perm[i]] = C.dim(i);
        }
    }

    for (int i = 0; i < A_perm.size(); i++) {
        if (out[A_perm[i]] == 0) {
            out[A_perm[i]] = A.dim(i);
        }
    }

    for (int i = 0; i < B_perm.size(); i++) {
        if (out[B_perm[i]] == 0) {
            out[B_perm[i]] = B.dim(i);
        }
    }

    return out;
}