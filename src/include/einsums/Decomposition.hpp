#pragma once

#include "einsums/LinearAlgebra.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/TensorAlgebra.hpp"

#include <cmath>
#include <functional>

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
    Dim<TRank> dims;

    size_t i = 0;
    for (const auto& factor : factors) {
        dims[i] = factor.dim(0);
        if (!rank) rank = factor.dim(1);
        i++;
    }

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

template <size_t TRank, typename TType = double>
auto initialize_cp(std::vector<Tensor<2, TType>> &folds, size_t rank) -> std::vector<Tensor<2, TType>> {

    std::vector<Tensor<2, TType>> factors;

    // Perform compile-time looping.
    for_sequence<TRank>([&](auto i) {
        auto [U, S, _] = linear_algebra::svd_a(folds[i]);

        // println(tensor_algebra::unfold<i>(tensor));
        // println(S);

        if (folds[i].dim(0) < rank) {
            println_warn("dimension {} size {} is less than the requested decomposition rank {}", i, folds[i].dim(0), rank);
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

    using namespace einsums::tensor_algebra;
    using namespace einsums::tensor_algebra::index;

    // Compute set of unfolded matrices
    std::vector<Tensor<2, TType>> unfolded_matrices;
    for_sequence<TRank>([&](auto i) {
        unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor));
    });

    // Perform SVD guess for parafac decomposition procedure
    std::vector<Tensor<2, TType>> factors = initialize_cp<TRank, TType>(unfolded_matrices, rank);

    // Keep track of previous factors (for tracking convergence)
    std::vector<Tensor<2, TType>> prev_factors;
    for_sequence<TRank>([&](auto i) {
        prev_factors.push_back(Tensor<2, TType>{"prev factor element", tensor.dim(i), rank});
        prev_factors[i].zero();
    });

    int iter = 0;
    bool converged = false;
    while (iter < n_iter_max) {
        for_sequence<TRank>([&](auto n_ind) {
            // Update prev factors
            prev_factors[n_ind] = factors[n_ind];

            // Form V and Khatri-Rao product intermediates
            Tensor<2, TType> V;
            Tensor<2, TType> *KR;
            bool first = true;

            for_sequence<TRank>([&](auto m_ind) {
                if (m_ind != n_ind) {
                    Tensor<2, TType> A_tA{"V", rank, rank};
                    // A_tA = A^T[j] @ A[j]
                    einsum(0.0, Indices{r, s}, &A_tA, 
                           1.0, Indices{I, r}, factors[m_ind], Indices{I, s}, factors[m_ind]);

                    if (first) {
                        V = A_tA;
                        Tensor<2, TType>* KRcopy = new Tensor<2, TType>("KR product", tensor.dim(m_ind), rank);
                        *KRcopy = factors[m_ind];
                        KR = KRcopy;
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<2, TType> Vcopy = V;
                        einsum(0.0, Indices{r, s}, &V,
                               1.0, Indices{r, s}, Vcopy, Indices{r, s}, A_tA);
                        
                        // Perform a Khatri-Rao contraction
                        // TODO: Implement an actual Khatri-Rao procedure to replace this "hacky" workaround

                        size_t running_dim = KR->dim(0);
                        size_t appended_dim = tensor.dim(m_ind);
                        
                        Tensor<2, TType> *newKR = new Tensor<2, TType>("KR product", running_dim * appended_dim, rank);
                        newKR->zero();

                        // einsum(0.0, Indices{I, M, r}, newKR,
                        //       1.0, Indices{I, r}, *KR, Indices{M, r}, factors[m_ind]);

                        for (size_t I = 0; I < running_dim; I++) {
                            for (size_t M = 0; M < appended_dim; M++) {
                                for (size_t R = 0; R < rank; R++) {
                                    (*newKR)(I * appended_dim + M, R) += (*KR)(I, R) * factors[m_ind](M, R);
                                }
                            }
                        }

                        delete KR;
                        KR = newKR;
                    }
                }
            });

            // Update factors[n_ind]
            size_t ndim = tensor.dim(n_ind);

            // Step 1: Matrix Multiplication
            einsum(0.0, Indices{I, r}, &factors[n_ind],
                   1.0, Indices{I, K}, unfolded_matrices[n_ind], Indices{K, r}, *KR);
            delete KR;

            // Step 2: Linear Solve (instead of inversion, for numerical stability, column-major ordering)
            linear_algebra::gesv(&V, &factors[n_ind]);
            
        });

        // Check for convergence
        double rmsd_max = 0.0;
        for_sequence<TRank>([&](auto n) {
            rmsd_max = std::max(rmsd_max, rmsd(factors[n], prev_factors[n]));
        });

        if (rmsd_max < tolerance) {
            converged = true;
            break;
        }

        iter += 1;
    }
    if (!converged) {
        println_warn("CP decomposition failed to converge in {} iterations", n_iter_max);
    }

    // Return **non-normalized** factors
    return factors;

}

} // namespace einsums::decomposition