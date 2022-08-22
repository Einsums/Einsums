#pragma once

#include "einsums/Tensor.hpp"

#include <algorithm>

namespace einsums::element_operations {

namespace detail {
template <typename vector, typename Functor>
void omp_loop(vector &data, Functor functor) {
    // TODO: This only works for Tensors not their views because we assume data is a std::vector
#pragma omp parallel
    {
        auto tid = omp_get_thread_num();
        auto chunksize = data.size() / omp_get_num_threads();
        auto begin = data.begin() + chunksize * tid;
        auto end = (tid == omp_get_num_threads() - 1) ? data.end() : begin + chunksize;

#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#pragma omp simd
#endif
        for (auto i = begin; i < end; i++) {
            *i = functor(*i);
        }
    }
}
} // namespace detail

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto sum(const TensorType<T, Rank> &tensor) -> T {
    // TODO: This currently only works with Tensor's not TensorViews. And it needs to be OpenMP'd
    T result = std::accumulate(tensor.vector_data().begin(), tensor.vector_data().end(), T{0});
    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto max(const TensorType<T, Rank> &tensor) -> T {
    // TODO: This currently only works with Tensor's not TensorViews. And it needs to be OpenMP'd
    auto result = std::max_element(tensor.vector_data().begin(), tensor.vector_data().end());
    return *result;
}

namespace new_tensor {

using einsums::element_operations::max; // nolint
using einsums::element_operations::sum; // nolint

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto abs(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    auto result = create_tensor_like(tensor);
    result = tensor;

    detail::omp_loop(result.vector_data(), [](T &value) { return std::abs(value); });

    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto invert(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    auto result = create_tensor_like(tensor);
    result = tensor;
    auto &data = result.vector_data();

    // TODO: This only works for Tensor's not their views.
    detail::omp_loop(data, [&](T &value) { return T{1} / value; });

    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t Rank>
auto exp(const TensorType<T, Rank> &tensor) -> Tensor<T, Rank> {
    auto result = create_tensor_like(tensor);
    result = tensor;
    auto &data = result.vector_data();

    detail::omp_loop(data, [&](T &value) { return std::exp(value); });

    return result;
}

} // namespace new_tensor
} // namespace einsums::element_operations