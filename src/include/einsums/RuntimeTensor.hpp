#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/TensorBases.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <variant>
#include <vector>

namespace einsums {

namespace detail {
class RuntimeTensorNoType {
  public:
    RuntimeTensorNoType()                            = default;
    RuntimeTensorNoType(const RuntimeTensorNoType &) = default;

    virtual ~RuntimeTensorNoType() = default;
};

class RuntimeTensorViewNoType : public virtual RuntimeTensorNoType {
  public:
    RuntimeTensorViewNoType()                                = default;
    RuntimeTensorViewNoType(const RuntimeTensorViewNoType &) = default;

    virtual ~RuntimeTensorViewNoType() = default;
};
} // namespace detail

// forward declaration.
template <typename T>
class RuntimeTensorView;

/**
 * @class RuntimeTensor
 *
 * @brief Represents a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
class RuntimeTensor : public virtual tensor_props::TensorBase,
                      public virtual tensor_props::TypedTensorBase<T>,
                      public virtual tensor_props::BasicTensorBase,
                      public virtual detail::RuntimeTensorNoType {
  public:
    using Vector = VectorData<T>;

  protected:
    Vector              _data;
    std::string         _name{"(unnamed)"};
    std::vector<size_t> _dims, _strides;
    size_t              _rank{0};

    template <typename TOther>
    friend class RuntimeTensorView;

    template <typename TOther>
    friend class RuntimeTensor;

  public:
    RuntimeTensor() = default;

    RuntimeTensor(const RuntimeTensor<T> &copy) = default;

    RuntimeTensor(std::string name, const std::vector<size_t> &dims) : _rank{dims.size()}, _name{name}, _dims{dims} {
        size_t size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = size;
            size *= _dims[i];
        }

        _data.resize(size);
    }

    explicit RuntimeTensor(const std::vector<size_t> &dims) : _rank{dims.size()}, _dims{dims} {
        size_t size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = size;
            size *= _dims[i];
        }

        _data.resize(size);
    }

    template <size_t Rank>
    RuntimeTensor(const Tensor<T, Rank> &copy) : _rank{Rank}, _dims(Rank), _strides(Rank), _name{copy.name()} {
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = copy.dim(i);
            _strides[i] = copy.stride(i);
        }

        _data.resize(copy.size());

        std::memcpy(_data.data(), copy.data(), copy.size() * sizeof(T));
    }

    template <size_t Rank>
    RuntimeTensor(const TensorView<T, Rank> &copy) : _rank{Rank}, _dims(Rank), _strides(Rank) {
        size_t size = 1;

        for (int i = Rank - 1; i >= 0; i--) {
            _strides[i] = size;
            _dims[i]    = (size_t)copy.dim(i);
            size *= _dims[i];
        }

        _data.resize(size);

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < this->size(); sentinel++) {
            size_t hold = sentinel, ord = 0;
            for (int i = 0; i < Rank; i++) {
                ord += copy.stride(i) * (hold / _strides[i]);
                hold %= _strides[i];
            }
            _data[sentinel] = copy.data()[ord];
        }
    }

    virtual ~RuntimeTensor() = default;

    virtual void zero() { std::memset(_data.data(), 0, _data.size() * sizeof(T)); }

    virtual void set_all(T val) { std::fill(_data.begin(), _data.end(), val); }

    T *data() { return _data.data(); }

    const T *data() const { return _data.data(); }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T *data(const Storage &index) {
        return &(_data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T *data(const Storage &index) const {
        return &(_data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T &operator()(const Storage &index) {
        if (index.size() < rank()) {
            throw EINSUMSEXCEPTION("Too few indices passed to subscript tensor!");
        }
        return _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T &operator()(const Storage &index) const {
        if (index.size() < rank()) {
            throw EINSUMSEXCEPTION("Too few indices passed to subscript tensor!");
        }
        return _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename... Args>
    T *data(Args... args) THROWS(einsums::EinsumsException, std::out_of_range) {
        if (sizeof...(Args) > rank()) {
            throw EINSUMSEXCEPTION("Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename... Args>
    const T *data(Args... args) const THROWS(einsums::EinsumsException, std::out_of_range) {
        if (sizeof...(Args) > rank()) {
            throw EINSUMSEXCEPTION("Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    /**
     * TODO: std::variant can't handle references. We might be able to make our own variant that can.
     * This new variant may also be able to replace HostDevReference.
     */
    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    T &operator()(Args... args) THROWS(einsums::EinsumsException) {
        if (sizeof...(Args) < rank()) {
            throw EINSUMSEXCEPTION("Not yet implemented: can not handle fewer integral indices than rank in (non-const) runtime tensor.");
        } else if (sizeof...(Args) > rank()) {
            throw EINSUMSEXCEPTION("Too many indices passed to subscript operator!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    std::variant<T, RuntimeTensorView<T>> operator()(Args... args) const THROWS(einsums::EinsumsException) {
        if (sizeof...(Args) > rank()) {
            throw EINSUMSEXCEPTION("Too many indices passed to subscript operator!");
        }

        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};

        if (sizeof...(Args) < rank()) {
            std::vector<Range> slices(sizeof...(Args));

            for (int i = 0; i < sizeof...(Args); i++) {
                slices[i] = Range{-1, index[i]};
            }
            return std::variant<T, RuntimeTensorView<T>>((*this)(slices));
        } else {
            return std::variant<T, RuntimeTensorView<T>>(
                _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index)));
        }
    }

    template <typename... Args>
        requires((!std::is_integral_v<Args>) || ...)
    RuntimeTensorView<T> operator()(Args... args) const THROWS(einsums::EinsumsException) {
        if (sizeof...(Args) > rank()) {
            throw EINSUMSEXCEPTION("Too many indices passed to subscript operator!");
        }

        std::tuple<Args...> arg_tuple = std::make_tuple(args...);
        std::vector<Range>  slices(sizeof...(Args));

        for_sequence<sizeof...(Args)>([&](auto n) {
            using Arg = std::tuple_element_t<n, std::tuple<Args...>>;
            if constexpr (std::is_same_v<Arg, AllT>) {
                slices[n] = Range{0, this->dim(n)};
            } else if constexpr (std::is_same_v<Arg, Range>) {
                slices[n] = std::get<n>(arg_tuple);
            } else if constexpr (std::is_integral_v<Arg>) {
                auto index = std::get<n>(arg_tuple);

                if (index < 0) {
                    index += this->dim(n);
                }

                slices[n] = Range{-1, index};
            }
        });

        return (*this)(slices);
    }

    /*
     * Special cases:
     *    Range{a, a + 1}: Keep the axis in the view. It will have dimension 1 and only have the a'th element. a can not be negative.
     *    Range{-1, a}: Remove the axis from the view. It will still affect the offset. a can not be negative.
     */
    RuntimeTensorView<T> operator()(const std::vector<Range> &slices) {
        if (slices.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        throw EINSUMSEXCEPTION("Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    RuntimeTensorView<T> operator()(const std::vector<Range> &slices) const {
        if (slices.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        throw EINSUMSEXCEPTION("Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    template <size_t Rank>
    RuntimeTensor<T> &operator=(const Tensor<T, Rank> &other) {
        if (_rank != Rank) {
            _rank = Rank;
            _dims.resize(Rank);
            _strides.resize(Rank);
        }
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }
        std::memcpy(_data.data(), other.data(), other.size() * sizeof(T));

        return *this;
    }

    template <typename TOther, size_t Rank>
    RuntimeTensor<T> &operator=(const Tensor<TOther, Rank> &other) {
        if (_rank != Rank) {
            _rank = Rank;
            _dims.resize(Rank);
            _strides.resize(Rank);
        }
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < _data.size(); i++) {
            _data[i] = (T)other.data()[i];
        }

        return *this;
    }

    template <typename TOther, size_t Rank>
    RuntimeTensor<T> &operator=(const TensorView<TOther, Rank> &other) {
        if (_rank != Rank) {
            _rank = Rank;
            _dims.resize(Rank);
            _strides.resize(Rank);
        }
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            std::array<size_t, Rank> index;

            einsums::tensor_algebra::detail::sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = (T)std::apply(other, index);
        }

        return *this;
    }

    virtual RuntimeTensor<T> &operator=(const RuntimeTensor<T> &other) {
        if (_rank != other.rank()) {
            _rank = other.rank();
            _dims.resize(other.rank());
            _strides.resize(other.rank());
        }
        for (int i = 0; i < other.rank(); i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            _data[sentinel] = other.data()[sentinel];
        }

        return *this;
    }

    virtual RuntimeTensor<T> &operator=(const RuntimeTensorView<T> &other) {
        if (_dims != other.dims() || _rank != other.rank()) {
            if (_rank != other.rank()) {
                _rank = other.rank();
                _dims.resize(other.rank());
                _strides.resize(other.rank());
            }
            _data.resize(other.size());
            for (int i = 0; i < other.rank(); i++) {
                _dims[i] = other.dim(i);
            }
            size_t stride = 1;
            for (int i = _rank - 1; i >= 0; i--) {
                _strides[i] = stride;
                stride *= _dims[i];
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            std::vector<size_t> index;

            tensor_algebra::detail::sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = other(index);
        }

        return *this;
    }

    template <typename TOther>
    RuntimeTensor<T> &operator=(const RuntimeTensor<TOther> &other) {
        if (_rank != other.rank()) {
            _rank = other.rank();
            _dims.resize(other.rank());
            _strides.resize(other.rank());
        }
        for (int i = 0; i < other.rank(); i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            _data[sentinel] = other.data()[sentinel];
        }

        return *this;
    }

    template <typename TOther>
    RuntimeTensor<T> &operator=(const RuntimeTensorView<TOther> &other) {
        if (_rank != other.rank()) {
            _rank = other.rank();
            _dims.resize(other.rank());
            _strides.resize(other.rank());
        }
        for (int i = 0; i < other.rank(); i++) {
            _dims[i]    = other.dim(i);
            _strides[i] = other.stride(i);
        }
        if (_data.size() != other.size()) {
            _data.resize(other.size());
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _data.size(); sentinel++) {
            std::vector<size_t> index;

            tensor_algebra::detail::sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = other(index);
        }

        return *this;
    }

    virtual RuntimeTensor<T> &operator=(T value) {
        set_all(value);
        return *this;
    }

#define OPERATOR(OP, NAME)                                                                                                                 \
    template <typename TOther>                                                                                                             \
    auto operator OP(const TOther &b)->RuntimeTensor<T> & {                                                                                \
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto begin     = _data.begin() + chunksize * tid;                                                                              \
            auto end       = (tid == omp_get_num_threads() - 1) ? _data.end() : begin + chunksize;                                         \
            EINSUMS_OMP_SIMD for (auto i = begin; i < end; i++) {                                                                          \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    (*i) OP(T)(RemoveComplexT<T>) b;                                                                                       \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    (*i) OP(T) b.real();                                                                                                   \
                } else {                                                                                                                   \
                    (*i) OP(T) b;                                                                                                          \
                }                                                                                                                          \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
    template <typename TOther>                                                                                                             \
    auto operator OP(const RuntimeTensor<TOther> &b)->RuntimeTensor<T> & {                                                                 \
        if (size() != b.size()) {                                                                                                          \
            throw EINSUMSEXCEPTION(fmt::format("tensors differ in size : {} {}", size(), b.size()));                                       \
        }                                                                                                                                  \
        T            *this_data = this->data();                                                                                            \
        const TOther *b_data    = b.data();                                                                                                \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                         \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                this->_data[sentinel] OP(T)(RemoveComplexT<T>) b_data[sentinel];                                                           \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                this->_data[sentinel] OP(T) b_data[sentinel].real();                                                                       \
            } else {                                                                                                                       \
                this->_data[sentinel] OP(T) b_data[sentinel];                                                                              \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename TOther>                                                                                                             \
    auto operator OP(const RuntimeTensorView<TOther> &b)->RuntimeTensor<T> & {                                                             \
        if (b.rank() != rank()) {                                                                                                          \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with runtime tensor and view of different ranks!");                            \
        }                                                                                                                                  \
        if (b.dims() != dims()) {                                                                                                          \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with runtime tensor and view of different dimensions!");                       \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                         \
            std::vector<size_t> index(rank());                                                                                             \
            tensor_algebra::detail::sentinel_to_indices(sentinel, this->_strides, index);                                                  \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                (*this)(index) OP(T)(RemoveComplexT<T>) b(index);                                                                          \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                (*this)(index) OP(T) b(index).real();                                                                                      \
            } else {                                                                                                                       \
                (*this)(index) OP(T) b(index);                                                                                             \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=, mult)
    OPERATOR(/=, div)
    OPERATOR(+=, add)
    OPERATOR(-=, sub)

#undef OPERATOR

    virtual auto dim(int d) const THROWS(std::out_of_range) -> size_t {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims.at(d);
    }
    virtual auto dims() const noexcept -> std::vector<size_t> { return _dims; }

    virtual auto vector_data() const -> const Vector & { return _data; }
    virtual auto vector_data() -> Vector & { return _data; }

    virtual auto stride(int d) const THROWS(std::out_of_range) -> size_t {
        if (d < 0) {
            d += _rank;
        }
        return _strides.at(d);
    }

    virtual auto strides() const noexcept -> std::vector<size_t> { return _strides; }

    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        size_t              size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        std::vector<size_t> dim{size};

        return RuntimeTensorView<T>{*this, dim};
    }

    // Returns the linear size of the tensor
    virtual auto size() const -> size_t { return _data.size(); }

    virtual auto full_view_of_underlying() const noexcept -> bool override { return true; }

    virtual const std::string &name() const noexcept override { return _name; };

    virtual void set_name(const std::string &new_name) override { _name = new_name; };

    virtual size_t rank() const noexcept override { return _rank; }
}; // namespace einsums

/**
 * @class RuntimeTensorView
 *
 * @brief Represents a view of a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
class RuntimeTensorView : public virtual tensor_props::TensorViewBase<RuntimeTensor<T>>,
                          public virtual tensor_props::TypedTensorBase<T>,
                          public virtual tensor_props::BasicTensorBase,
                          public virtual detail::RuntimeTensorViewNoType {
  protected:
    T                  *_data;
    std::string         _name{"(unnamed view)"};
    std::vector<size_t> _dims, _strides, _index_strides;
    size_t              _rank{0}, _size{0}, _alloc_size{0};
    bool                _full_view{false};

  public:
    RuntimeTensorView() = default;

    RuntimeTensorView(const RuntimeTensorView<T> &copy) = default;

    RuntimeTensorView(RuntimeTensor<T> &view)
        : _data{view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(const RuntimeTensor<T> &view)
        : _data{(T *)view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(const RuntimeTensor<T> &other, const std::vector<size_t> &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{true}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = (T *)other.data();
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensor<T> &other, const std::vector<size_t> &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{true}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = other.data();
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensorView<T> &other, const std::vector<size_t> &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{other.full_view_of_underlying()}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = other.data();
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(const RuntimeTensorView<T> &other, const std::vector<size_t> &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{other.full_view_of_underlying()}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = (T *)other.data();
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensor<T> &other, const std::vector<size_t> &dims, const std::vector<size_t> &strides,
                      const std::vector<size_t> &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides}, _full_view{other.dims() == dims && other.strides() == strides},
          _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = other.data(offsets);
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(const RuntimeTensor<T> &other, const std::vector<size_t> &dims, const std::vector<size_t> &strides,
                      const std::vector<size_t> &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides}, _full_view{other.dims() == dims && other.strides() == strides},
          _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = (T *)other.data(offsets);
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensorView<T> &other, const std::vector<size_t> &dims, const std::vector<size_t> &strides,
                      const std::vector<size_t> &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides},
          _full_view{other.full_view_of_underlying() && other.dims() == dims && other.strides() == strides}, _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = other.data(offsets);
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(const RuntimeTensorView<T> &other, const std::vector<size_t> &dims, const std::vector<size_t> &strides,
                      const std::vector<size_t> &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides},
          _full_view{other.full_view_of_underlying() && other.dims() == dims && other.strides() == strides}, _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = (T *)other.data(offsets);
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    template <size_t Rank>
    RuntimeTensorView(TensorView<T, Rank> &copy)
        : _data{copy.data()}, _dims{copy.dims()}, _strides{copy.strides()}, _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank - 1; i >= 0; i--) {
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    template <size_t Rank>
    RuntimeTensorView(const TensorView<T, Rank> &copy)
        : _data{copy.data()}, _dims{copy.dims()}, _strides{copy.strides()}, _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank - 1; i >= 0; i--) {
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    template <size_t Rank>
    RuntimeTensorView(Tensor<T, Rank> &copy)
        : _data{copy.data()}, _dims{copy.dims()}, _strides{copy.strides()}, _rank{Rank}, _full_view{true}, _index_strides{copy.strides()},
          _size{copy.size()} {}

    template <size_t Rank>
    RuntimeTensorView(const Tensor<T, Rank> &copy)
        : _data{copy.data()}, _dims{copy.dims()}, _strides{copy.strides()}, _rank{Rank}, _full_view{true}, _index_strides{copy.strides()},
          _size{copy.size()} {}

    virtual ~RuntimeTensorView() = default;

    virtual void zero() {
        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0;

            for (int i = 0; i < _rank; i++) {
                ord += _strides[i] * (hold / _index_strides[i]);
                hold %= _index_strides[i];
            }

            _data[ord] = T{0.0};
        }
    }

    virtual void set_all(T val) {
        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0;

            for (int i = 0; i < _rank; i++) {
                ord += _strides[i] * (hold / _index_strides[i]);
                hold %= _index_strides[i];
            }

            _data[ord] = val;
        }
    }

    T *data() { return _data; }

    const T *data() const { return _data; }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T *data(const Storage &index) {
        return &(_data[einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T *data(const Storage &index) const {
        return &(_data[einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T &operator()(const Storage &index) {
        return _data[einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T &operator()(const Storage &index) const {
        return _data[einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    T *data(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data[index * _strides[0]]);
    }

    const T *data(ptrdiff_t index) const {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data[index * _strides[0]]);
    }

    T &operator()(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return _data[index * _strides[0]];
    }

    const T &operator()(ptrdiff_t index) const {
        if (index < 0) {
            index += _dims[0];
        }
        return _data[index * _strides[0]];
    }

    /*
     * Special cases:
     *    Rank{a, a}: Keep the axis in the view. It will have dimension 1 and only have the a'th element. a can not be negative.
     *    Rank{-1, a}: Remove the axis from the view. It will still affect the offset. a can not be negative.
     */
    RuntimeTensorView<T> operator()(const std::vector<Range> &slices) {
        if (slices.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        throw EINSUMSEXCEPTION("Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    RuntimeTensorView<T> operator()(const std::vector<Range> &slices) const {
        if (slices.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets(_rank), strides;
        dims.reserve(_rank);
        strides.reserve(_rank);

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets[i] = 0;
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets[i] = end;
                } else {
                    if (start < 0) {
                        start += _dims[i];
                    }
                    if (end < 0) {
                        end += _dims[i];
                    }

                    if (start < 0 || end < 0 || start >= _dims[i] || end > _dims[i] || start >= end) {
                        throw EINSUMSEXCEPTION("Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(const Tensor<TOther, Rank> &other) {
        if (_rank != Rank) {
            throw EINSUMSEXCEPTION("Can not assign a tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] != other.dim(i)) {
                throw EINSUMSEXCEPTION("Can not assign a tensor to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t                   hold = sentinel, ord = 0;
            std::array<size_t, Rank> index;

            for (int i = 0; i < Rank; i++) {
                size_t ind = hold / _index_strides[i];
                index[i]   = ind;
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = std::apply(other, index);
        }

        return *this;
    }

    template <typename TOther, size_t Rank>
    RuntimeTensorView<T> &operator=(const TensorView<TOther, Rank> &other) {
        if (_rank != Rank) {
            throw EINSUMSEXCEPTION("Can not assign a tensor view to a runtime view with a different rank!");
        }
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] != other.dim(i)) {
                throw EINSUMSEXCEPTION("Can not assign a tensor view to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t                   hold = sentinel, ord = 0;
            std::array<size_t, Rank> index;

            for (int i = 0; i < Rank; i++) {
                size_t ind = hold / _index_strides[i];
                index[i]   = ind;
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = std::apply(other, index);
        }

        return *this;
    }

    virtual RuntimeTensorView<T> &operator=(const RuntimeTensor<T> &other) {
        if (_rank != other.rank()) {
            throw EINSUMSEXCEPTION("Can not assign a runtime tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                throw EINSUMSEXCEPTION("Can not assign a runtime tensor to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[sentinel];
        }

        return *this;
    }

    virtual RuntimeTensorView<T> &operator=(const RuntimeTensorView<T> &other) {
        if (_rank != other.rank()) {
            throw EINSUMSEXCEPTION("Can not assign a runtime view to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                throw EINSUMSEXCEPTION("Can not assign a runtime view to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                other_ord += ind * other.stride(i);
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[other_ord];
        }

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator=(const RuntimeTensor<TOther> &other) {
        if (_rank != other.rank()) {
            throw EINSUMSEXCEPTION("Can not assign a runtime tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                throw EINSUMSEXCEPTION("Can not assign a runtime tensor to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[sentinel];
        }

        return *this;
    }

    template <typename TOther>
    RuntimeTensorView<T> &operator=(const RuntimeTensorView<TOther> &other) {
        if (_rank != other.rank()) {
            throw EINSUMSEXCEPTION("Can not assign a runtime view to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                throw EINSUMSEXCEPTION("Can not assign a runtime view to a runtime view with different dimensions!");
            }
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {
            size_t hold = sentinel, ord = 0, other_ord = 0;

            for (int i = 0; i < _rank; i++) {
                size_t ind = hold / _index_strides[i];
                ord += ind * _strides[i];
                other_ord += ind * other.stride(i);
                hold %= _index_strides[i];
            }

            _data[ord] = other.data()[other_ord];
        }

        return *this;
    }

    virtual RuntimeTensorView<T> &operator=(T value) {
        set_all(value);
        return *this;
    }

#define OPERATOR(OP, NAME)                                                                                                                 \
    template <typename TOther>                                                                                                             \
    auto operator OP(const TOther &b)->RuntimeTensorView<T> & {                                                                            \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {                                                                          \
            size_t hold = sentinel, ord = 0;                                                                                               \
                                                                                                                                           \
            for (int i = 0; i < _rank; i++) {                                                                                              \
                ord += _strides[i] * (hold / _index_strides[i]);                                                                           \
                hold %= _index_strides[i];                                                                                                 \
            }                                                                                                                              \
                                                                                                                                           \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                this->_data[ord] OP(T)(RemoveComplexT<T>) b;                                                                               \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                this->_data[ord] OP(T) b.real();                                                                                           \
            } else {                                                                                                                       \
                this->_data[ord] OP(T) b;                                                                                                  \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename TOther>                                                                                                             \
    auto operator OP(const RuntimeTensor<TOther> &b)->RuntimeTensorView<T> & {                                                             \
        if (b.rank() != rank()) {                                                                                                          \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with runtime views of different ranks!");                                      \
        }                                                                                                                                  \
        if (b.dims() != dims()) {                                                                                                          \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with runtime views of different dimensions!");                                 \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {                                                                          \
            size_t hold = sentinel, ord = 0, b_ord = 0;                                                                                    \
                                                                                                                                           \
            for (int i = 0; i < _rank; i++) {                                                                                              \
                ord += _strides[i] * (hold / _index_strides[i]);                                                                           \
                b_ord += b.stride(i) * (hold / _index_strides[i]);                                                                         \
                hold %= _index_strides[i];                                                                                                 \
            }                                                                                                                              \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                this->_data[ord] OP(T)(RemoveComplexT<T>) b.data()[b_ord];                                                                 \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                this->_data[ord] OP(T) b.data()[b_ord].real();                                                                             \
            } else {                                                                                                                       \
                this->_data[ord] OP(T) b.data()[b_ord];                                                                                    \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    template <typename TOther>                                                                                                             \
    auto operator OP(const RuntimeTensorView<TOther> &b)->RuntimeTensorView<T> & {                                                         \
        if (b.rank() != rank()) {                                                                                                          \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with runtime views of different ranks!");                                      \
        }                                                                                                                                  \
        if (b.dims() != dims()) {                                                                                                          \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with runtime views of different dimensions!");                                 \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {                                                                          \
            size_t hold = sentinel, ord = 0, b_ord = 0;                                                                                    \
                                                                                                                                           \
            for (int i = 0; i < _rank; i++) {                                                                                              \
                ord += _strides[i] * (hold / _index_strides[i]);                                                                           \
                b_ord += b.stride(i) * (hold / _index_strides[i]);                                                                         \
                hold %= _index_strides[i];                                                                                                 \
            }                                                                                                                              \
                                                                                                                                           \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                this->_data[ord] OP(T)(RemoveComplexT<T>) b.data()[b_ord];                                                                 \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                this->_data[ord] OP(T) b.data()[b_ord].real();                                                                             \
            } else {                                                                                                                       \
                this->_data[ord] OP(T) b.data()[b_ord];                                                                                    \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=, mult)
    OPERATOR(/=, div)
    OPERATOR(+=, add)
    OPERATOR(-=, sub)

#undef OPERATOR

    virtual auto dim(int d) const THROWS(std::out_of_range) -> size_t {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims.at(d);
    }
    virtual auto dims() const noexcept -> std::vector<size_t> { return _dims; }

    virtual auto stride(int d) const THROWS(std::out_of_range) -> size_t {
        if (d < 0) {
            d += _rank;
        }
        return _strides.at(d);
    }

    virtual auto strides() const noexcept -> std::vector<size_t> { return _strides; }

    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        size_t              size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        std::vector<size_t> dim{size};

        return RuntimeTensorView<T>{*this, dim};
    }

    // Returns the linear size of the tensor
    virtual auto size() const noexcept -> size_t { return _size; }

    virtual auto full_view_of_underlying() const noexcept -> bool override { return true; }

    virtual const std::string &name() const override { return _name; };

    virtual void set_name(const std::string &new_name) override { _name = new_name; };

    virtual size_t rank() const noexcept override { return _rank; }
}; // namespace einsums

} // namespace einsums

template <einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void fprintln(std::ostream &os, const AType &A, TensorPrintOptions options) {
    using T     = typename AType::data_type;
    size_t Rank = A.rank();

    fprintln(os, "Name: {}", A.name());
    {
        print::Indent const indent{};

        if constexpr (einsums::CoreTensorConcept<AType>) {
            if constexpr (!einsums::TensorViewConcept<AType>)
                fprintln(os, "Type: In Core Runtime Tensor");
            else
                fprintln(os, "Type: In Core Runtime Tensor View");
#ifdef __HIP__
        } else if constexpr (einsums::DeviceTensorConcept<AType>) {
            if constexpr (!einsums::TensorViewConcept<AType>)
                fprintln(os, "Type: Device Runtime Tensor");
            else
                fprintln(os, "Type: Device Runtime Tensor View");
#endif
        } else if constexpr (einsums::DiskTensorConcept<AType>) {
            fprintln(os, "Type: Runtime Disk Tensor");
        } else {
            fprintln(os, "Type: {}", type_name<AType>());
        }

        fprintln(os, "Data Type: {}", type_name<T>());

        if (Rank > 0) {
            std::ostringstream oss;
            for (size_t i = 0; i < Rank; i++) {
                oss << A.dim(i) << " ";
            }
            fprintln(os, "Dims{{{}}}", oss.str().c_str());
        }

        if constexpr (einsums::BasicTensorConcept<AType>) {
            if (Rank > 0) {
                std::ostringstream oss;
                for (size_t i = 0; i < Rank; i++) {
                    oss << A.stride(i) << " ";
                }
                fprintln(os, "Strides{{{}}}", oss.str());
            }
        }

        if (options.full_output) {
            fprintln(os);

            if (Rank == 0) {
                T value = A;

                std::ostringstream oss;
                oss << "              ";
                if constexpr (std::is_floating_point_v<T>) {
                    if (std::abs(value) < 1.0E-4) {
                        oss << fmt::format("{:14.4e} ", value);
                    } else {
                        oss << fmt::format("{:14.8f} ", value);
                    }
                } else if constexpr (einsums::IsComplexV<T>) {
                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                } else
                    oss << fmt::format("{:14} ", value);

                fprintln(os, "{}", oss.str());
                fprintln(os);
            } else if (Rank > 1) {
#ifndef __HIP__
                if constexpr (einsums::CoreTensorConcept<AType>) {
#else
                if constexpr (einsums::CoreTensorConcept<AType> || einsums::DeviceTensorConcept<AType>) {
#endif
                    auto                target_dims = std::vector<size_t>(A.dims().begin(), A.dims().end() - 1);
                    auto                final_dim   = A.dim(Rank - 1);
                    auto                ndigits     = einsums::ndigits(final_dim);
                    std::vector<size_t> index_strides, index(Rank - 1);
                    index.reserve(Rank);

                    size_t size = einsums::tensor_algebra::detail::dims_to_strides(target_dims, index_strides);

                    for (size_t i = 0; i < size; i++) {
                        std::ostringstream oss;
                        einsums::tensor_algebra::detail::sentinel_to_indices(i, index_strides, index);
                        index.resize(Rank);

                        for (int j = 0; j < final_dim; j++) {
                            if (j % options.width == 0) {
                                std::ostringstream tmp;
                                for (int k = 0; k < Rank - 1; k++) {
                                    tmp << index[k] << ", ";
                                }
                                if (final_dim >= j + options.width)
                                    oss << fmt::format("{:<14}", fmt::format("({} {:{}d}-{:{}d}): ", tmp.str(), j, ndigits,
                                                                             j + options.width - 1, ndigits));
                                else
                                    oss << fmt::format("{:<14}",
                                                       fmt::format("({} {:{}d}-{:{}d}): ", tmp.str(), j, ndigits, final_dim - 1, ndigits));
                            }
                            index[Rank - 1] = j;
                            T value         = std::get<T>(A(index));
                            if (std::abs(value) > 1.0E+10) {
                                if constexpr (std::is_floating_point_v<T>)
                                    oss << "\x1b[0;37;41m" << fmt::format("{:14.8f} ", value) << "\x1b[0m";
                                else if constexpr (einsums::IsComplexV<T>)
                                    oss << "\x1b[0;37;41m(" << fmt::format("{:14.8f} ", value.real()) << " + "
                                        << fmt::format("{:14.8f}i)", value.imag()) << "\x1b[0m";
                                else
                                    oss << "\x1b[0;37;41m" << fmt::format("{:14d} ", value) << "\x1b[0m";
                            } else {
                                if constexpr (std::is_floating_point_v<T>) {
                                    if (std::abs(value) < 1.0E-4) {
                                        oss << fmt::format("{:14.4e} ", value);
                                    } else {
                                        oss << fmt::format("{:14.8f} ", value);
                                    }
                                } else if constexpr (einsums::IsComplexV<T>) {
                                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                                } else
                                    oss << fmt::format("{:14} ", value);
                            }
                            if (j % options.width == options.width - 1 && j != final_dim - 1) {
                                oss << "\n";
                            }
                        }
                        fprintln(os, "{}", oss.str());
                        fprintln(os);
                    }
                } else if (Rank == 1) {
#ifndef __HIP__
                    if constexpr (einsums::CoreTensorConcept<AType>) {
#else
                    if constexpr (einsums::CoreTensorConcept<AType> || einsums::DeviceTensorConcept<AType>) {
#endif
                        for (size_t i = 0; i < A.size(); i++) {
                            std::ostringstream oss;
                            oss << "(" << i << "): ";

                            T value = A(i);
                            if (std::abs(value) > 1.0E+5) {
                                if constexpr (std::is_floating_point_v<T>)
                                    oss << "\x1b[0;37;41m" << fmt::format("{:14.8f} ", value) << "\x1b[0m";
                                else if constexpr (einsums::IsComplexV<T>) {
                                    oss << "\x1b[0;37;41m(" << fmt::format("{:14.8f} ", value.real()) << " + "
                                        << fmt::format("{:14.8f}i)", value.imag()) << "\x1b[0m";
                                } else
                                    oss << "\x1b[0;37;41m" << fmt::format("{:14} ", value) << "\x1b[0m";
                            } else {
                                if constexpr (std::is_floating_point_v<T>)
                                    if (std::abs(value) < 1.0E-4) {
                                        oss << fmt::format("{:14.4e} ", value);
                                    } else {
                                        oss << fmt::format("{:14.8f} ", value);
                                    }
                                else if constexpr (einsums::IsComplexV<T>) {
                                    oss << fmt::format("({:14.8f} ", value.real()) << " + " << fmt::format("{:14.8f}i)", value.imag());
                                } else
                                    oss << fmt::format("{:14} ", value);
                            }

                            fprintln(os, "{}", oss.str());
                        }
                    }
                }
            }
        }
    }
    fprintln(os);
}

template <einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void fprintln(std::FILE *fp, const AType &A, TensorPrintOptions options) {
    std::stringstream stream;

    fprintln(stream, A, options);

    std::fprintf(fp, "%s", stream.str().c_str());

    std::fflush(fp);
}

template <einsums::TensorConcept AType>
    requires requires {
        requires einsums::BasicTensorConcept<AType> || !einsums::AlgebraTensorConcept<AType>;
        requires !einsums::RankTensorConcept<AType>;
    }
void println(const AType &A, TensorPrintOptions options) {
    fprintln(std::cout, A, options);
}