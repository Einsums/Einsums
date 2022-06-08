#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

#include <cmath>
#include <functional>

using namespace einsums;
using namespace einsums::tensor_algebra;
using namespace einsums::tensor_algebra::index;

namespace einsums::decomposition {

template <size_t TRank>
auto validate_cp_rank(const Dim<TRank> shape, const std::string &rounding = "round") -> size_t {
    using rounding_func_t = double (*)(double);
    rounding_func_t rounding_func;

    if (rounding == "ceil") {
        rounding_func = ::ceil;
    } else if (rounding == "floor") {
        rounding_func = ::floor;
    } else if (rounding == "round") {
        rounding_func = ::round;
    } else {
        throw std::runtime_error(fmt::format("Rounding should of round, floow, or ceil, but got {}", rounding));
    }

    double prod = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<double>());
    double sum = std::accumulate(shape.begin(), shape.end(), 0, std::plus<double>());

    return static_cast<int>(rounding_func(prod / sum));
}

/**
 * Computes the RMSD between two tensors of arbitrary dimension
 */
template <template <size_t, typename> typename TTensor, size_t TRank, typename TType = double>
auto rmsd(const TTensor<TRank, TType> &tensor1, const TTensor<TRank, TType> &tensor2) -> TType {
    TType diff = 0.0;
    auto target_dims = get_dim_ranges<TRank>(tensor1);

    size_t nelem = 0;
    for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
        TType target1 = std::apply(tensor1, target_combination);
        TType target2 = std::apply(tensor2, target_combination);
        diff += (target1 - target2)*(target1 - target2);
        nelem += 1;
    }

    return std::sqrt(diff / nelem);
}

/**
 * Reconstructs a tensor given a CANDECOMP/PARAFAC decomposition
 * 
 *   factors = The decomposed CANDECOMP matrices (dimension: [dim[i], rank])
 */
template <size_t TRank, typename TType = double>
auto parafac_reconstruct(const std::vector<Tensor<2, TType>>& factors) -> Tensor<TRank, TType> {

    size_t rank = 0;
    size_t tensor_size;
    Dim<TRank> dims;
    std::vector<size_t> offsets_per_dim;

    size_t i = 0;
    for (const auto& factor : factors) {
        dims[i] = factor.dim(0);
        if (!rank) rank = factor.dim(1);
        i++;
    }

    offsets_per_dim.resize(TRank);
    size_t offset = 1;
    for (size_t n = 0; n < TRank; n++) {
        size_t m = TRank - n - 1;
        offsets_per_dim[m] = offset;
        offset *= dims[m];
    }
    tensor_size = offset;

    Tensor<TRank, TType> new_tensor(dims);
    new_tensor.zero();

    auto indices = get_dim_ranges<TRank>(new_tensor);

    for (auto idx_combo : std::apply(ranges::views::cartesian_product, indices)) {
        TType &target = std::apply(new_tensor, idx_combo);
        for (size_t r = 0; r < rank; r++) {
            double temp = 1.0;
            for_sequence<TRank>([&](auto n) {
                temp *= factors[n](std::get<n>(idx_combo), r);
            });
            target += temp;
        }
    }

    return new_tensor;

}

template <template <size_t, typename> typename TTensor, size_t TRank, typename TType = double>
auto initialize_cp(const TTensor<TRank, TType> &tensor, size_t rank) -> std::vector<Tensor<2, TType>> {

    std::vector<Tensor<2, TType>> factors;

    // Perform compile-time looping.
    for_sequence<TRank>([&](auto i) {
        auto [U, S, _] = linear_algebra::svd_a(tensor_algebra::unfold<i>(tensor));

        // println(tensor_algebra::unfold<i>(tensor));
        // println(S);

        if (tensor.dim(i) < rank) {
            println_warn("dimension {} size {} is less than the requested decomposition rank {}", i, tensor.dim(i), rank);
            // TODO: Need to padd U up to rank
        }

        // Need to save the factors
        factors.emplace_back(Tensor{U(All{}, Range{0, rank})});
    });

    return factors;
}

/**
 * CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights; factor[0], ..., factors[-1] |].
 */
template <template <size_t, typename> typename TTensor, size_t TRank, typename TType = double>
auto parafac(const TTensor<TRank, TType> &tensor, size_t rank, int n_iter_max = 100, double tolerance = 1.e-8) -> std::vector<Tensor<2, TType>> {

    // Perform SVD guess for parafac decomposition procedure
    auto factors = initialize_cp(tensor, rank);

    // Get set of unfolded matrices
    std::vector<Tensor<2, TType>> unfolded_matrices;
    for_sequence<TRank>([&](auto i) {
        unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor));
    });

    int iter = 0;
    while (iter < n_iter_max) {
        for_sequence<TRank>([&](auto n_ind) {
            // Form V and Khatri-Rao product intermediates
            Tensor<2, TType> V;
            std::vector<Tensor<2, TType>> KRs;
            bool first = true;

            for_sequence<TRank>([&](auto m_ind) {
                if (m_ind != n_ind) {
                    Tensor<2, TType> A_tA{"V", rank, rank};
                    // A_tA = A^T[j] @ A[j]
                    einsum(0.0, Indices{r, s}, &A_tA, 
                           1.0, Indices{I, r}, factors[m_ind], Indices{I, s}, factors[m_ind]);

                    if (first) {
                        V = A_tA;
                        KRs.push_back(factors[m_ind]);
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<2, TType> Vcopy = V;
                        einsum(0.0, Indices{r, s}, &V,
                               1.0, Indices{r, s}, Vcopy, Indices{r, s}, A_tA);
                        
                        // Perform a Khatri-Rao contraction
                        // TODO: Implement an actual Khatri-Rao procedure to replace this "hacky" workaround
                        Tensor<2, TType>& oldKR = KRs.back();

                        size_t running_dim = oldKR.dim(0);
                        size_t appended_dim = tensor.dim(m_ind);
                        Tensor<3, TType> KRtemp{"KRtemp", running_dim, appended_dim, rank};

                        einsum(0.0, Indices{I, M, r}, &KRtemp,
                               1.0, Indices{I, r}, oldKR, Indices{M, r}, factors[m_ind]);

                        // TensorView<2, TType> KRview{KRtemp, Dim<2>{running_dim * appended_dim, rank}};
                        // KR = KRview;

                        Tensor<2, TType> newKR{"New KR", running_dim * appended_dim, rank};

                        for (size_t I = 0; I < running_dim; I++) {
                            for (size_t M = 0; M < appended_dim; M++) {
                                for (size_t r = 0; r < rank; r++) {
                                    newKR(I * appended_dim + M, r) = KRtemp(I, M, r);
                                }
                            }
                        }
                        
                        KRs.push_back(newKR);
                    }
                }
            });

            // Update factors[n_ind]

            size_t ndim = tensor.dim(n_ind);

            // Step 1: Matrix Multiplication
            einsum(0.0, Indices{I, r}, &factors[n_ind],
                   1.0, Indices{I, K}, unfolded_matrices[n_ind], Indices{K, r}, KRs.back());

            // Step 2: Linear Solve (instead of inversion, for numerical stability, column-major ordering)
            linear_algebra::gesv(&V, &factors[n_ind]);
            
        });
        iter += 1;
    }

    // Return **non-normalized** factors
    return factors;

}

} // namespace einsums::decomposition