#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>

#include <stdexcept>
#include <variant>
#include <vector>

namespace einsums {

// forward declaration.
template <typename T>
class RuntimeTensorView;

/**
 * @class RuntimeTensor
 *
 * @brief Represents a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
class EINSUMS_EXPORT RuntimeTensor : public virtual tensor_base::TypedTensor<T>,
                                     public virtual tensor_base::AlgebraOptimizedTensor,
                                     public virtual tensor_base::CoreTensor {
  public:
    using Vector = VectorData<T>;

    RuntimeTensor() = default;

    RuntimeTensor(RuntimeTensor<T> const &copy) = default;

    RuntimeTensor(std::string name, std::vector<size_t> const &dims) : _rank{dims.size()}, _name{name}, _dims{dims} {
        size_t size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = size;
            size *= _dims[i];
        }

        _data.resize(size);
    }

    explicit RuntimeTensor(std::vector<size_t> const &dims) : _rank{dims.size()}, _dims{dims} {
        size_t size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = size;
            size *= _dims[i];
        }

        _data.resize(size);
    }

    template <size_t Rank>
    RuntimeTensor(Tensor<T, Rank> const &copy) : _rank{Rank}, _dims(Rank), _strides(Rank), _name{copy.name()} {
        for (int i = 0; i < Rank; i++) {
            _dims[i]    = copy.dim(i);
            _strides[i] = copy.stride(i);
        }

        _data.resize(copy.size());

        std::memcpy(_data.data(), copy.data(), copy.size() * sizeof(T));
    }

    template <size_t Rank>
    RuntimeTensor(TensorView<T, Rank> const &copy) : _rank{Rank}, _dims(Rank), _strides(Rank) {
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

    T const *data() const { return _data.data(); }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T *data(Storage const &index) {
        return &(_data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T *data(Storage const &index) const {
        return &(_data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T &operator()(Storage const &index) {
        if (index.size() < rank()) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Too few indices passed to subscript tensor!");
        }
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T &operator()(Storage const &index) const {
        if (index.size() < rank()) {
            EINSUMS_THROW_EXCEPTION(not_enough_args, "Too few indices passed to subscript tensor!");
        }
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename... Args>
    T *data(Args... args) {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename... Args>
    T const *data(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    /**
     * TODO: std::variant can't handle references. We might be able to make our own variant that can.
     * This new variant may also be able to replace HostDevReference.
     */
    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    T &operator()(Args... args) {
        if (sizeof...(Args) < rank()) {
            EINSUMS_THROW_EXCEPTION(todo_error,
                                    "Not yet implemented: can not handle fewer integral indices than rank in (non-const) runtime tensor.");
        } else if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    std::variant<T, RuntimeTensorView<T>> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }

        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};

        if (sizeof...(Args) < rank()) {
            std::vector<Range> slices(sizeof...(Args));

            for (int i = 0; i < sizeof...(Args); i++) {
                slices[i] = Range{-1, index[i]};
            }
            return std::variant<T, RuntimeTensorView<T>>((*this)(slices));
        } else {
            return std::variant<T, RuntimeTensorView<T>>(_data.at(einsums::indices_to_sentinel_negative_check(_strides, _dims, index)));
        }
    }

    template <typename... Args>
        requires((!std::is_integral_v<Args>) || ...)
    RuntimeTensorView<T> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
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
    RuntimeTensorView<T> operator()(std::vector<Range> const &slices) {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
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
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    RuntimeTensorView<T> const operator()(std::vector<Range> const &slices) const {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
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
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
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
    RuntimeTensor<T> &operator=(Tensor<T, Rank> const &other) {
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
    RuntimeTensor<T> &operator=(Tensor<TOther, Rank> const &other) {
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
    RuntimeTensor<T> &operator=(TensorView<TOther, Rank> const &other) {
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

            einsums::sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = (T)std::apply(other, index);
        }

        return *this;
    }

    virtual RuntimeTensor<T> &operator=(RuntimeTensor<T> const &other) {
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

    virtual RuntimeTensor<T> &operator=(RuntimeTensorView<T> const &other) {
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

            sentinel_to_indices(sentinel, _strides, index);

            _data[sentinel] = other(index);
        }

        return *this;
    }

    template <typename TOther>
    RuntimeTensor<T> &operator=(RuntimeTensor<TOther> const &other) {
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
    RuntimeTensor<T> &operator=(RuntimeTensorView<TOther> const &other) {
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

            sentinel_to_indices(sentinel, _strides, index);

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
            EINSUMS_THROW_EXCEPTION(dimension_error, fmt::format("tensors differ in size : {} {}", size(), b.size()));                     \
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
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not perform " #OP " with runtime tensor and view of different ranks!");      \
        }                                                                                                                                  \
        if (b.dims() != dims()) {                                                                                                          \
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform " #OP " with runtime tensor and view of different dimensions!");     \
        }                                                                                                                                  \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                         \
            std::vector<size_t> index(rank());                                                                                             \
            sentinel_to_indices(sentinel, this->_strides, index);                                                                          \
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

    virtual auto dim(int d) const -> size_t {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims.at(d);
    }
    virtual auto dims() const noexcept -> std::vector<size_t> { return _dims; }

    virtual auto vector_data() const -> Vector const & { return _data; }
    virtual auto vector_data() -> Vector & { return _data; }

    virtual auto stride(int d) const -> size_t {
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

    virtual bool full_view_of_underlying() const noexcept override { return true; }

    virtual std::string const &name() const noexcept override { return _name; };

    virtual void set_name(std::string const &new_name) override { _name = new_name; };

    virtual size_t rank() const noexcept override { return _rank; }

  protected:
    Vector              _data;
    std::string         _name{"(unnamed)"};
    std::vector<size_t> _dims, _strides;
    size_t              _rank{0};

    template <typename TOther>
    friend class RuntimeTensorView;

    template <typename TOther>
    friend class RuntimeTensor;
}; // namespace einsums

/**
 * @class RuntimeTensorView
 *
 * @brief Represents a view of a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
class EINSUMS_EXPORT RuntimeTensorView : public virtual tensor_base::TensorView<RuntimeTensor<T>>,
                                         public virtual tensor_base::TypedTensor<T>,
                                         public virtual tensor_base::AlgebraOptimizedTensor,
                                         public virtual tensor_base::CoreTensor {
  protected:
    T                  *_data;
    std::string         _name{"(unnamed view)"};
    std::vector<size_t> _dims, _strides, _index_strides;
    size_t              _rank{0}, _size{0}, _alloc_size{0};
    bool                _full_view{false};

  public:
    RuntimeTensorView() = default;

    RuntimeTensorView(RuntimeTensorView<T> const &copy) = default;

    RuntimeTensorView(RuntimeTensor<T> &view)
        : _data{view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensor<T> const &view)
        : _data{(T *)view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensor<T> const &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{true}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = (T *)other.data();
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensor<T> &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{true}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = other.data();
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensorView<T> &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{other.full_view_of_underlying()}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = other.data();
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensorView<T> const &other, std::vector<size_t> const &dims)
        : _rank{dims.size()}, _dims{dims}, _full_view{other.full_view_of_underlying()}, _index_strides(dims.size()) {
        _size = 1;
        _strides.resize(_rank);

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i] = _size;
            _size *= _dims[i];
        }

        _data = (T *)other.data();
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensor<T> &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides}, _full_view{other.dims() == dims && other.strides() == strides},
          _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensor<T> const &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides}, _full_view{other.dims() == dims && other.strides() == strides},
          _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = (T *)other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensorView<T> &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides},
          _full_view{other.full_view_of_underlying() && other.dims() == dims && other.strides() == strides}, _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(RuntimeTensorView<T> const &other, std::vector<size_t> const &dims, std::vector<size_t> const &strides,
                      std::vector<size_t> const &offsets)
        : _rank{dims.size()}, _dims{dims}, _strides{strides},
          _full_view{other.full_view_of_underlying() && other.dims() == dims && other.strides() == strides}, _index_strides(dims.size()) {

        _size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>{});

        _data = (T *)other.data(offsets);
        dims_to_strides(_dims, _index_strides);
    }

    template <size_t Rank>
    RuntimeTensorView(TensorView<T, Rank> &copy)
        : _data{copy.data()}, _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    template <size_t Rank>
    RuntimeTensorView(TensorView<T, Rank> const &copy)
        : _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _data = const_cast<T *>(copy.data());

        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    template <size_t Rank>
    RuntimeTensorView(Tensor<T, Rank> &copy)
        : _data{copy.data()}, _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{true}, _index_strides(Rank), _size{copy.size()} {
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = copy.stride(i);
        }
    }

    template <size_t Rank>
    RuntimeTensorView(Tensor<T, Rank> const &copy)
        : _dims(Rank), _strides(Rank), _rank{Rank}, _full_view{true}, _index_strides(Rank), _size{copy.size()} {
        _data = const_cast<T *>(copy.data());
        for (int i = Rank - 1; i >= 0; i--) {
            _dims[i]          = copy.dim(i);
            _strides[i]       = copy.stride(i);
            _index_strides[i] = copy.stride(i);
        }
    }

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

    T const *data() const { return _data; }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T *data(Storage const &index) {
        return &(_data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T *data(Storage const &index) const {
        return &(_data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    T &operator()(Storage const &index) {
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T &operator()(Storage const &index) const {
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    T *data(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data[index * _strides[0]]);
    }

    T const *data(ptrdiff_t index) const {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data[index * _strides[0]]);
    }

    template <typename... Args>
    T *data(Args... args) {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    template <typename... Args>
    T const *data(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to data!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    /**
     * TODO: std::variant can't handle references. We might be able to make our own variant that can.
     * This new variant may also be able to replace HostDevReference.
     */
    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    T &operator()(Args... args) {
        if (sizeof...(Args) < rank()) {
            EINSUMS_THROW_EXCEPTION(todo_error,
                                    "Not yet implemented: can not handle fewer integral indices than rank in (non-const) runtime tensor.");
        } else if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }
        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};
        return _data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    template <typename... Args>
        requires(std::is_integral_v<Args> && ...)
    std::variant<T, RuntimeTensorView<T>> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
        }

        std::array<ptrdiff_t, sizeof...(Args)> index{static_cast<ptrdiff_t>(args)...};

        if (sizeof...(Args) < rank()) {
            std::vector<Range> slices(sizeof...(Args));

            for (int i = 0; i < sizeof...(Args); i++) {
                slices[i] = Range{-1, index[i]};
            }
            return std::variant<T, RuntimeTensorView<T>>((*this)(slices));
        } else {
            return std::variant<T, RuntimeTensorView<T>>(_data[einsums::indices_to_sentinel_negative_check(_strides, _dims, index)]);
        }
    }

    template <typename... Args>
        requires((!std::is_integral_v<Args>) || ...)
    const RuntimeTensorView<T> operator()(Args... args) const {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
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

    template <typename... Args>
        requires((!std::is_integral_v<Args>) || ...)
    RuntimeTensorView<T> operator()(Args... args) {
        if (sizeof...(Args) > rank()) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to subscript operator!");
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

    T &operator()(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return _data[index * _strides[0]];
    }

    T const &operator()(ptrdiff_t index) const {
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
    RuntimeTensorView<T> operator()(std::vector<Range> const &slices) {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
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
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
                    }

                    dims.push_back(end - start);
                    offsets[i] = start;
                    strides.push_back(_strides[i]);
                }
            }
        }

        return RuntimeTensorView<T>(*this, dims, strides, offsets);
    }

    RuntimeTensorView<T> operator()(std::vector<Range> const &slices) const {
        if (slices.size() > _rank) {
            EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices passed to tensor!");
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
                        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Index out of range! Either the start or end is out of range!");
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
    RuntimeTensorView<T> &operator=(Tensor<TOther, Rank> const &other) {
        if (_rank != Rank) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a tensor to a runtime view with different dimensions!");
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
    RuntimeTensorView<T> &operator=(TensorView<TOther, Rank> const &other) {
        if (_rank != Rank) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a tensor view to a runtime view with a different rank!");
        }
        for (int i = 0; i < Rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a tensor view to a runtime view with different dimensions!");
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

    virtual RuntimeTensorView<T> &operator=(RuntimeTensor<T> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime tensor to a runtime view with different dimensions!");
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

    virtual RuntimeTensorView<T> &operator=(RuntimeTensorView<T> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime view to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime view to a runtime view with different dimensions!");
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
    RuntimeTensorView<T> &operator=(RuntimeTensor<TOther> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime tensor to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime tensor to a runtime view with different dimensions!");
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
    RuntimeTensorView<T> &operator=(RuntimeTensorView<TOther> const &other) {
        if (_rank != other.rank()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not assign a runtime view to a runtime view with a different rank!");
        }
        for (int i = 0; i < _rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "Can not assign a runtime view to a runtime view with different dimensions!");
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
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not perform " #OP " with runtime views of different ranks!");                \
        }                                                                                                                                  \
        if (b.dims() != dims()) {                                                                                                          \
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform " #OP " with runtime views of different dimensions!");               \
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
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Can not perform " #OP " with runtime views of different ranks!");                \
        }                                                                                                                                  \
        if (b.dims() != dims()) {                                                                                                          \
            EINSUMS_THROW_EXCEPTION(dimension_error, "Can not perform " #OP " with runtime views of different dimensions!");               \
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

    virtual auto dim(int d) const -> size_t {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims.at(d);
    }
    virtual auto dims() const noexcept -> std::vector<size_t> { return _dims; }

    virtual auto stride(int d) const -> size_t {
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

    virtual bool full_view_of_underlying() const noexcept override { return true; }

    virtual std::string const &name() const override { return _name; };

    virtual void set_name(std::string const &new_name) override { _name = new_name; };

    virtual size_t rank() const noexcept override { return _rank; }
}; // namespace einsums

} // namespace einsums
