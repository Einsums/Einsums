#pragma once

#include <h5cpp/core>
#include <type_traits>

namespace h5::impl {

// 1. Object -> H5T_xxx
//    Conversion from Tensor to the underlying double.
template <typename T, size_t Rank>
struct decay<::einsums::Tensor<Rank, T>> {
    using type = T;
};

template <typename T, size_t Rank>
struct decay<::einsums::TensorView<Rank, T>> {
    using type = T;
};

template <typename T, size_t Rank>
auto data(const ::einsums::Tensor<Rank, T> &ref) -> const T * {
    return ref.data();
}

template <typename T, size_t Rank>
auto data(const ::einsums::TensorView<Rank, T> &ref) -> const T * {
    return ref.data();
}

template <typename T, size_t Rank>
auto data(::einsums::Tensor<Rank, T> &ref) -> T * {
    return ref.data();
}

// Determine rank and dimensions
template <typename T, size_t Rank>
struct rank<::einsums::Tensor<Rank, T>> : public std::integral_constant<size_t, Rank> {};

template <typename T, size_t Rank>
struct rank<::einsums::TensorView<Rank, T>> : public std::integral_constant<size_t, Rank> {};

template <typename T, size_t Rank>
inline auto size(const ::einsums::Tensor<Rank, T> &ref) -> std::array<size_t, Rank> {
    return ref.dims();
}

template <typename T, size_t Rank>
inline auto size(const ::einsums::TensorView<Rank, T> &ref) -> std::array<size_t, Rank> {
    return ref.dims();
}

// Constructors
//  Not sure if this can be generalized.
//  Only allow Tensor to be read in and not TensorView
template <typename T>
struct get<::einsums::Tensor<1, T>> {
    static inline auto ctor(std::array<size_t, 1> dims) -> ::einsums::Tensor<1, T> {
        return ::einsums::Tensor<1, T>("hdf5 auto created", dims[0]);
    }
};
template <typename T>
struct get<::einsums::Tensor<2, T>> {
    static inline auto ctor(std::array<size_t, 2> dims) -> ::einsums::Tensor<2, T> {
        return ::einsums::Tensor<2, T>("hdf5 auto created", dims[0], dims[1]);
    }
};
template <typename T>
struct get<::einsums::Tensor<3, T>> {
    static inline auto ctor(std::array<size_t, 3> dims) -> ::einsums::Tensor<3, T> {
        return ::einsums::Tensor<3, T>("hdf5 auto created", dims[0], dims[1], dims[2]);
    }
};
template <typename T>
struct get<::einsums::Tensor<4, T>> {
    static inline auto ctor(std::array<size_t, 4> dims) -> ::einsums::Tensor<4, T> {
        return ::einsums::Tensor<4, T>("hdf5 auto created", dims[0], dims[1], dims[2], dims[3]);
    }
};

} // namespace h5::impl

#include <h5cpp/io>

namespace h5 {

inline bool exists(hid_t hid, const std::string &name) { return H5Lexists(hid, name.c_str(), H5P_DEFAULT) > 0 ? true : false; }

} // namespace h5