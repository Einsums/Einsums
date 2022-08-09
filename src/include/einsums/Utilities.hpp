#pragma once

#include "einsums/Blas.hpp"
#include "einsums/LinearAlgebra.hpp"
#include "einsums/OpenMP.h"

namespace einsums {

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
auto sum_square(const AType<ADataType, ARank> &a, remove_complex_t<ADataType> *scale, remove_complex_t<ADataType> *sumsq) ->
    typename std::enable_if_t<is_incore_rank_tensor_v<AType<ADataType, ARank>, 1, ADataType>> {
    int n = a.dim(0);
    int incx = a.stride(0);
    blas::lassq(n, a.data(), incx, scale, sumsq);
}

template <typename T = double, typename... MultiIndex>
auto create_incremented_tensor(const std::string &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    Tensor<T, sizeof...(MultiIndex)> A(name, std::forward<MultiIndex>(index)...);

    T counter{0.0};
    auto target_dims = get_dim_ranges<sizeof...(MultiIndex)>(A);
    auto view = std::apply(ranges::views::cartesian_product, target_dims);

    for (auto it = view.begin(); it != view.end(); it++) {
        std::apply(A, *it) = counter;
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            counter += T{1.0, 1.0};
        } else {
            counter += T{1.0};
        }
    }

    return A;
}

template <typename T = double, bool Normalize = false, typename... MultiIndex>
auto create_random_tensor(const std::string &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    Tensor<T, sizeof...(MultiIndex)> A(name, std::forward<MultiIndex>(index)...);

    double lower_bound = -1.0;
    double upper_bound = 1.0;

    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    {
        static std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();

        // std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now() - beginning;

        // re.seed(d.count());
    }

    if constexpr (std::is_same_v<T, std::complex<float>>) {
#pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin = A.vector_data().begin() + chunksize * tid;
            auto end = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() {
                return T{static_cast<float>(unif(re)), static_cast<float>(unif(re))};
            });
        }
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
#pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin = A.vector_data().begin() + chunksize * tid;
            auto end = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() {
                return T{static_cast<double>(unif(re)), static_cast<double>(unif(re))};
            });
        }
    } else {
#pragma omp parallel
        {
            auto tid = omp_get_thread_num();
            auto chunksize = A.vector_data().size() / omp_get_num_threads();
            auto begin = A.vector_data().begin() + chunksize * tid;
            auto end = (tid == omp_get_num_threads() - 1) ? A.vector_data().end() : begin + chunksize;
            std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() { return static_cast<T>(unif(re)); });
        }
    }

    if constexpr (Normalize == true && sizeof...(MultiIndex) == 2) {
        for (int col = 0; col < A.dim(-1); col++) {
            remove_complex_t<T> scale{1}, sumsq{0};

            auto column = A(All, col);
            // auto collapsed = TensorView{A, Dim<2>{-1, A.dim(-1)}};
            // auto column = collapsed(All, col);
            sum_square(column, &scale, &sumsq);
            T value = scale * sqrt(sumsq);
            column /= value;
        }
    }

    return A;
}

namespace detail {

template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank, typename Tuple, std::size_t... I>
void set_to(TensorType<DataType, Rank> &tensor, DataType value, Tuple const &tuple, std::index_sequence<I...>) {
    tensor(std::get<I>(tuple)...) = value;
}

} // namespace detail

template <typename T = double, typename... MultiIndex>
auto create_identity_tensor(const std::string &name, MultiIndex... index) -> Tensor<T, sizeof...(MultiIndex)> {
    static_assert(sizeof...(MultiIndex) >= 1, "Rank parameter doesn't make sense.");

    Tensor<T, sizeof...(MultiIndex)> A{name, std::forward<MultiIndex>(index)...};
    A.zero();

    for (size_t dim = 0; dim < std::get<0>(std::forward_as_tuple(index...)); dim++) {
        detail::set_to(A, T{1.0}, create_tuple<sizeof...(MultiIndex)>(dim), std::make_index_sequence<sizeof...(MultiIndex)>());
    }

    return A;
}

template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
auto create_tensor_like(const TensorType<DataType, Rank> &tensor) -> Tensor<DataType, Rank> {
    return Tensor<DataType, Rank>{tensor.dims()};
}

template <template <typename, size_t> typename TensorType, typename DataType, size_t Rank>
auto create_tensor_like(const std::string name, const TensorType<DataType, Rank> &tensor) -> Tensor<DataType, Rank> {
    auto result = Tensor<DataType, Rank>{tensor.dims()};
    result.set_name(name);
    return result;
}

template <typename T>
auto arange(T start, T stop, T step = T{1}) -> Tensor<T, 1> {
    assert(stop >= start);

    // Determine the number of elements that will be produced
    int nelem = static_cast<int>((stop - start) / step);

    auto result = create_tensor<T>("arange created tensor", nelem);
    zero(result);

    int index{0};
    for (T value = start; value < stop; value += step) {
        result(index++) = value;
    }

    return result;
}

template <typename T>
auto arange(T stop) -> Tensor<T, 1> {
    return arange(T{0}, stop);
}

struct DisableOMPNestedScope {
    DisableOMPNestedScope() {
        _old_nested = omp_get_nested();
        omp_set_nested(0);
    }

    ~DisableOMPNestedScope() { omp_set_nested(_old_nested); }

  private:
    int _old_nested;
};

} // namespace einsums