#pragma once

#include "einsums/_Common.hpp"
#include "einsums/Tensor.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums)

template<typename T, size_t Rank>
struct BlockTensor : public detail::TensorBase<T, Rank> {
  private:
    std::string  _name{"(Unnamed)"};
    size_t _dim; // Only allowing square tensors.

    std::vector<Tensor<T, Rank>> _blocks;
    std::vector<Range> _ranges;

    template <typename T_, size_t OtherRank>
    friend struct BlockTensor;

  public:
        using datatype = T;
    using Vector = std::vector<T, AlignedAllocator<T, 64>>;

    /**
     * @brief Construct a new Tensor object. Default constructor.
     */
    BlockTensor() = default;

    /**
     * @brief Construct a new Tensor object. Default copy constructor
     */
    BlockTensor(const BlockTensor &) = default;

    /**
     * @brief Destroy the Tensor object.
     */
    ~BlockTensor() = default;

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
    explicit BlockTensor(std::string name, size_t dim) : _name{std::move(name)}, _dim{dim}, _blocks{}, _ranges{} {
    }

    /**
     * @brief Construct a new Tensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit BlockTensor(size_t dim) : _dim{dim} {
    }

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

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
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

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
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

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return _data[ordinal];
    }

    auto operator=(const Tensor<T, Rank> &other) -> Tensor<T, Rank> & {
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

            // Row-major order of dimensions
            std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
            size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        std::copy(other._data.begin(), other._data.end(), _data.begin());

        return *this;
    }

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const Tensor<TOther, Rank> &other) -> Tensor<T, Rank> & {
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

            // Row-major order of dimensions
            std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
            size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

            // Resize the data structure
            _data.resize(size);
        }

        auto target_dims = get_dim_ranges<Rank>(*this);
        for (auto target_combination : std::apply(ranges::views::cartesian_product, target_dims)) {
            T &target_value = std::apply(*this, target_combination);
            T  value        = std::apply(other, target_combination);
            target_value    = value;
        }

        return *this;
    }

#define OPERATOR(OP)                                                                                                                       \
    auto operator OP(const T &b) -> Tensor<T, Rank> & {                                                                                    \
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
    auto operator OP(const Tensor<T, Rank> &b) -> Tensor<T, Rank> & {                                                                      \
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
            d += Rank;
        return _strides[d];
    }

    auto strides() const noexcept -> const auto & { return _strides; }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }
};

END_EINSUMS_NAMESPACE_HPP(einsums)