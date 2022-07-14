#pragma once

#include "einsums/LinearAlgebra.hpp"

namespace einsums {

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

    double lower_bound = 0.0;
    double upper_bound = 1.0;

    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    {
        static std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now();

        // std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now() - beginning;

        // re.seed(d.count());
    }

    if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() {
            return T{static_cast<float>(unif(re)), static_cast<float>(unif(re))};
        });
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() {
            return T{static_cast<double>(unif(re)), static_cast<double>(unif(re))};
        });
    } else {
        std::generate(A.vector_data().begin(), A.vector_data().end(), [&]() { return static_cast<T>(unif(re)); });
    }

    if constexpr (Normalize == true && sizeof...(MultiIndex) == 2) {
        for (int col = 0; col < A.dim(-1); col++) {
            complex_type_t<T> scale{1}, sumsq{0};

            auto column = A(All, col);
            // auto collapsed = TensorView{A, Dim<2>{-1, A.dim(-1)}};
            // auto column = collapsed(All, col);
            linear_algebra::sum_square(column, &scale, &sumsq);
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
} // namespace einsums