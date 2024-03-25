//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_SymmIndex.hpp"

#include "einsums/OpenMP.h"
#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/State.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/utility/ComplexTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"
#include "range/v3/range_fwd.hpp"
#include "range/v3/view/cartesian_product.hpp"
#include "range/v3/view/iota.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <complex>
#include <concepts>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums {

/**
 * @brief Represents a general tensor
 *
 * @tparam T data type of the underlying tensor data
 * @tparam Rank the rank of the tensor
 */
template <typename T, size_t Rank, typename... IndexSpec>
struct SymmTensor final : public detail::TensorBase<T, Rank> {

    using datatype = T;
    using Vector   = std::vector<T, AlignedAllocator<T, 64>>;

    /**
     * @brief Construct a new Tensor object. Default constructor.
     */
    SymmTensor() = default;

    /**
     * @brief Construct a new Tensor object. Default copy constructor
     */
    SymmTensor(const SymmTensor &) = default;

    /**
     * @brief Destroy the Tensor object.
     */
    ~SymmTensor() = default;

    /**
     * @brief Construct a new Tensor object with the given name and dimensions.
     *
     * Constructs a new Tensor object using the information provided in \p name and \p dims .
     *
     * @code
     * auto A = Tensor("A", 3, 3);
     * @endcode
     *
     * The newly constructed Tensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling Tensor.zero() or zero(Tensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param dims The dimensions of each rank of the tensor.
     */
    template <typename... Dims>
    explicit SymmTensor(std::string name, Dims... dims)
        : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...}, _index_spec() {
        static_assert(Rank == sizeof...(dims), "Declared Rank does not match provided dims");

        struct Stride {
            size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        _true_dims = _index_spec.find_true_dims(_dims);

        // Row-major order of dimensions
        std::transform(_true_dims.rbegin(), _true_dims.rend(), _strides.rbegin(), Stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _true_dims[0];

        // Resize the data structure
        // TODO: Is this setting all the elements to zero?
        _data.resize(size);
    }

    // Once this is called "otherTensor" is no longer a valid tensor.
    /**
     * @brief Construct a new Tensor object. Moving \p existingTensor data to the new tensor.
     *
     * This constructor is useful for reshaping a tensor. It does not modify the underlying
     * tensor data. It only creates new mapping arrays for how the data is viewed.
     *
     * @code
     * auto A = Tensor("A", 27); // Creates a rank-1 tensor of 27 elements
     * auto B = Tensor(std::move(A), "B", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     * // At this point A is no longer valid.
     * @endcode
     *
     * Supports using -1 for one of the ranks to automatically compute the dimensional of it.
     *
     * @code
     * auto A = Tensor("A", 27);
     * auto B = Tensor(std::move(A), "B", 3, -1, 3); // Automatically determines that -1 should be 3.
     * @endcode
     *
     * @tparam OtherRank The rank of \p existingTensor can be different than the rank of the new tensor
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param existingTensor The existing tensor that holds the tensor data.
     * @param name The name of the new tensor
     * @param dims The dimensionality of each rank of the new tensor.
     */
    template <size_t OtherRank, typename... Dims>
    explicit SymmTensor(SymmTensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims)
        : _data(std::move(existingTensor._data)), _name{std::move(name)}, _dims{static_cast<size_t>(dims)...},
          _index_spec(std::move(existingTensor._index_spec)) {
        static_assert(Rank == sizeof...(dims), "Declared rank does not match provided dims");

        struct Stride {
            size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we
        // compute this dimensionality of this "0" index for them.
        int nfound{0};
        int location{-1};
        for (auto [i, dim] : enumerate(_dims)) {
            if (dim == -1) {
                nfound++;
                location = i;
            }
        }

        if (nfound > 1) {
            throw std::runtime_error("More than one -1 was provided.");
        }

        if (nfound == 1) {
            size_t size{1};
            for (auto [i, dim] : enumerate(_dims)) {
                if (i != location)
                    size *= dim;
            }
            if (size > existingTensor.size()) {
                throw std::runtime_error("Size of new tensor is larger than the parent tensor.");
            }
            _dims[location] = existingTensor.size() / size;
        }

        _true_dims = _index_spec.find_true_dims(_dims);

        // Row-major order of dimensions
        std::transform(_true_dims.rbegin(), _true_dims.rend(), _strides.rbegin(), Stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _true_dims[0];

        // Check size
        if (_data.size() != size) {
            throw std::runtime_error("Provided dims to not match size of parent tensor");
        }
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit SymmTensor(Dim<Rank> dims)
        : _dims{std::move(dims)}, _index_spec() {
        struct Stride {
            size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        _true_dims = _index_spec.find_true_dims(_dims);

        // Row-major order of dimensions
        std::transform(_true_dims.rbegin(), _true_dims.rend(), _strides.rbegin(), Stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _true_dims[0];

        // Resize the data structure
        // TODO: Is this setting all the elements to zero?
        _data.resize(size);
    }

    /**
     * @brief Construct a new Tensor object from a TensorView.
     *
     * Data is explicitly copied from the view to the new tensor.
     *
     * @param other The tensor view to copy.
     */
    // Tensor(const TensorView<T, Rank> &other) : _name{other._name}, _dims{other._dims} {
    //     struct Stride {
    //         size_t value{1};
    //         Stride() = default;
    //         auto operator()(size_t dim) -> size_t {
    //             auto old_value = value;
    //             value *= dim;
    //             return old_value;
    //         }
    //     };

    //     // Row-major order of dimensions
    //     std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
    //     size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    //     // Resize the data structure
    //     _data.resize(size);

    //     // TODO: Attempt to thread this.
    //     auto target_dims = get_dim_ranges<Rank>(*this);
    //     for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
    //         T &target_value = std::apply(*this, target_combination);
    //         T  value        = std::apply(other, target_combination);
    //         target_value    = value;
    //     }
    // }

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero() {
        // #pragma omp parallel
        //         {
        //             auto tid       = omp_get_thread_num();
        //             auto chunksize = _data.size() / omp_get_num_threads();
        //             auto begin     = _data.begin() + chunksize * tid;
        //             auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
        //             memset(&(*begin), 0, end - begin);
        //         }
        memset(_data.data(), 0, sizeof(T) * _data.size());
    }

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value) {
        // #pragma omp parallel
        //         {
        //             auto tid       = omp_get_thread_num();
        //             auto chunksize = _data.size() / omp_get_num_threads();
        //             auto begin     = _data.begin() + chunksize * tid;
        //             auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;
        //             std::fill(begin, end, value);
        //         }
        std::fill(_data.begin(), _data.end(), value);
    }

    /**
     * @brief Returns a pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return T* A pointer to the data.
     */
    auto data() -> T * { return _data.data(); }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    auto data() const -> const T * { return _data.data(); }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     * @code
     * auto A = Tensor("A", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     *
     * double* A_pointer = A.data(1, 2, 3); // Returns the pointer to element (1, 2, 3) in A.
     * @endcode
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> T * {
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
        assert(sizeof...(MultiIndex) <= _dims.size());

        auto index_list = std::vector{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        size_t ordinal = _index_spec(_strides, &index_list);
        return &_data[ordinal];
#endif
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) const -> const T & {

        assert(sizeof...(MultiIndex) == _dims.size());

        auto index_list = std::vector{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        size_t ordinal = _index_spec(_strides, &index_list);
        return _data[ordinal];
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> T & {

        assert(sizeof...(MultiIndex) == _dims.size());

        auto index_list = std::vector{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        size_t ordinal = _index_spec(_strides, &index_list);
        return _data[ordinal];
    }

    // WARNING: Chances are this function will not work if you mix All{}, Range{} and explicit indexes.
    // template <typename... MultiIndex>
    //     requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
    // auto operator()(MultiIndex... index) -> TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
    //     // Construct a TensorView using the indices provided as the starting point for the view.
    //     // e.g.:
    //     //    Tensor T{"Big Tensor", 7, 7, 7, 7};
    //     //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
    //     // println("Here");
    //     const auto &indices = std::forward_as_tuple(index...);

    //     Offset<Rank>                                                                         offsets;
    //     Stride<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> strides{};
    //     Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

    //     int counter{0};
    //     for_sequence<sizeof...(MultiIndex)>([&](auto i) {
    //         // println("looking at {}", i);
    //         if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
    //             auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
    //             if (tmp < 0)
    //                 tmp = _dims[i] + tmp;
    //             offsets[i] = tmp;
    //         } else if constexpr (std::is_same_v<AllT, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
    //             strides[counter] = _strides[i];
    //             dims[counter]    = _dims[i];
    //             counter++;

    //         } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
    //             auto range       = std::get<i>(indices);
    //             offsets[counter] = range[0];
    //             if (range[1] < 0) {
    //                 auto temp = _dims[i] + range[1];
    //                 range[1]  = temp;
    //             }
    //             dims[counter]    = range[1] - range[0];
    //             strides[counter] = _strides[i];
    //             counter++;
    //         }
    //     });

    //     return TensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>{*this, std::move(dims),
    //     offsets,
    //                                                                                                        strides};
    // }

    // template <typename... MultiIndex>
    //     requires NumOfType<Range, Rank, MultiIndex...>
    // auto operator()(MultiIndex... index) const -> TensorView<T, Rank> {
    //     Dim<Rank>    dims{};
    //     Offset<Rank> offset{};
    //     Stride<Rank> stride = _strides;

    //     auto ranges = get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

    //     for (int r = 0; r < Rank; r++) {
    //         auto range = ranges[r];
    //         offset[r]  = range[0];
    //         if (range[1] < 0) {
    //             auto temp = _dims[r] + range[1];
    //             range[1]  = temp;
    //         }
    //         dims[r] = range[1] - range[0];
    //     }

    //     return TensorView<T, Rank>{*this, std::move(dims), std::move(offset), std::move(stride)};
    // }

    auto operator=(const SymmTensor<T, Rank, IndexSpec...> &other) -> SymmTensor<T, Rank, IndexSpec...> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0 || (dim(i) != other.dim(i)))
                realloc = true;
        }

        if (realloc) {
            struct Stride {
                size_t value{1};
                Stride() = default;
                auto operator()(size_t dim) -> size_t {
                    auto old_value = value;
                    value *= dim;
                    return old_value;
                }
            };

            _dims = other._dims;

            _true_dims = _index_spec.find_true_dims(_dims);

            // Row-major order of dimensions
            std::transform(_true_dims.rbegin(), _true_dims.rend(), _strides.rbegin(), Stride());
            size_t size = _strides.size() == 0 ? 0 : _strides[0] * _true_dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        std::copy(other._data.begin(), other._data.end(), _data.begin());

        return *this;
    }

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const SymmTensor<TOther, Rank, IndexSpec...> &other) -> SymmTensor<T, Rank, IndexSpec...> & {
        bool realloc{false};
        for (int i = 0; i < Rank; i++) {
            if (dim(i) == 0)
                realloc = true;
            else if (dim(i) != other.dim(i)) {
                std::string str = fmt::format("Tensor::operator= dimensions do not match (this){} (other){}", dim(i), other.dim(i));
                if constexpr (Rank != 1)
                    throw std::runtime_error(str);
                else
                    realloc = true;
            }
        }

        if (realloc) {
            struct Stride {
                size_t value{1};
                Stride() = default;
                auto operator()(size_t dim) -> size_t {
                    auto old_value = value;
                    value *= dim;
                    return old_value;
                }
            };

            _dims = other._dims;

            _true_dims = _index_spec.find_true_dims(_dims);

            // Row-major order of dimensions
            std::transform(_true_dims.rbegin(), _true_dims.rend(), _strides.rbegin(), Stride());
            size_t size = _strides.size() == 0 ? 0 : _strides[0] * _true_dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        for (size_t i = 0; i < size(); i++) {
            _data[i] = (T)other._data[i];
        }

        return *this;
    }

    // template <typename TOther>
    // auto operator=(const TensorView<TOther, Rank> &other) -> SymmTensor<T, Rank> & {
    //     auto target_dims = get_dim_ranges<Rank>(*this);
    //     for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
    //         T &target_value = std::apply(*this, target_combination);
    //         T  value        = std::apply(other, target_combination);
    //         target_value    = value;
    //     }

    //     return *this;
    // }

    auto operator=(const T &fill_value) -> SymmTensor & {
        set_all(fill_value);
        return *this;
    }

#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &b) -> SymmTensor<T, Rank, IndexSpec...> & {                                                                  \
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto begin     = _data.begin() + chunksize * tid;                                                                              \
            auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;                                         \
            EINSUMS_OMP_SIMD for (auto i = begin; i < end; i++) {                                                                          \
                (*i) OP b;                                                                                                                 \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    auto operator OP(const SymmTensor<T, Rank, IndexSpec...> &b) -> SymmTensor<T, Rank, IndexSpec...> & {                                  \
        if (size() != b.size()) {                                                                                                          \
            throw std::runtime_error(fmt::format("operator" EINSUMS_STRINGIFY(OP) " : tensors differ in size : {} {}", size(), b.size())); \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto abegin    = _data.begin() + chunksize * tid;                                                                              \
            auto bbegin    = b._data.begin() + chunksize * tid;                                                                            \
            auto aend      = (tid == omp_get_num_threads() - 1) ? _data.end() : abegin + chunksize;                                        \
            auto j         = bbegin;                                                                                                       \
            EINSUMS_OMP_SIMD for (auto i = abegin; i < aend; i++) {                                                                        \
                (*i) OP(*j++);                                                                                                             \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=)
    OPERATOR(/=)
    OPERATOR(+=)
    OPERATOR(-=)

#undef OPERATOR

    [[nodiscard]] auto dim(int d) const -> size_t {
        // Add support for negative indices.
        if (d < 0)
            d += Rank;
        return _dims[d];
    }
    auto dims() const -> Dim<Rank> { return _dims; }

    ALIAS_TEMPLATE_FUNCTION(shape, dims);

    auto vector_data() const -> const Vector & { return _data; }
    auto vector_data() -> Vector & { return _data; }

    [[nodiscard]] auto name() const -> const std::string & { return _name; }
    void               set_name(const std::string &name) { _name = name; }

    [[nodiscard]] auto stride(int d) const noexcept -> size_t {
        if (d < 0)
            d += sizeof...(IndexSpec);
        return _strides[d];
    }

    auto strides() const noexcept -> const auto & { return _strides; }

    // auto to_rank_1_view() const -> TensorView<T, 1> {
    //     size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
    //     Dim<1> dim{size};

    //     return TensorView<T, 1>{*this, dim};
    // }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const {
        return std::accumulate(std::begin(_true_dims), std::begin(_true_dims) + Rank, 1, std::multiplies<>{});
    }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }

  private:
    std::string                         _name{"(Unnamed)"};
    Dim<Rank>                           _dims;
    Dim<sizeof...(IndexSpec)>           _true_dims;
    Stride<sizeof...(IndexSpec)>        _strides;
    Vector                              _data;
    symm_index::IndexList<IndexSpec...> _index_spec;

    template <typename T_, size_t OtherRank, typename... OtherIndexSpec>
    friend struct SymmTensor;
};

} // namespace einsums