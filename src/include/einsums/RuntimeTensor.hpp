#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/TensorBases.hpp"

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
class RuntimeTensor : public virtual tensor_props::TensorBase,
                      public virtual tensor_props::TypedTensorBase<T>,
                      public virtual tensor_props::BasicTensorBase {
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

        _data.resize(size());

        std::memcpy(_data.data(), copy.data(), size() * sizeof(T));
    }

    template <size_t Rank>
    RuntimeTensor(const TensorView<T, Rank> &copy) : _rank{Rank}, _dims(Rank) {
        size_t size = 1;
        _strides.resize(rank());

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
        return _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename Storage>
        requires(!std::is_arithmetic_v<Storage>)
    const T &operator()(const Storage &index) const {
        return _data.at(einsums::tensor_algebra::detail::indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    T *data(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data.at(index));
    }

    const T *data(ptrdiff_t index) const {
        if (index < 0) {
            index += _dims[0];
        }
        return &(_data.at(index));
    }

    T &operator()(ptrdiff_t index) {
        if (index < 0) {
            index += _dims[0];
        }
        return _data.at(index);
    }

    const T &operator()(ptrdiff_t index) const {
        if (index < 0) {
            index += _dims[0];
        }
        return _data.at(index);
    }

    /*
     * Special cases:
     *    Rank{a, a + 1}: Keep the axis in the view. It will have dimension 1 and only have the a'th element. a can not be negative.
     *    Rank{-1, a}: Remove the axis from the view. It will still affect the offset. a can not be negative.
     */
    RuntimeTensorView<T> operator()(const std::vector<Range> &slices) {
        if (slices.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }

        std::vector<size_t> dims, offsets, strides;

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets.push_back(0);
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets.push_back(end);
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
                    offsets.push_back(start);
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

        std::vector<size_t> dims, offsets, strides;

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets.push_back(0);
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets.push_back(end);
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
                    offsets.push_back(start);
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
    virtual auto operator OP(const T &b)->RuntimeTensor<T> & {                                                                             \
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
    virtual auto operator OP(const RuntimeTensor<T> &b)->RuntimeTensor<T> & {                                                              \
        if (size() != b.size()) {                                                                                                          \
            throw EINSUMSEXCEPTION(fmt::format("tensors differ in size : {} {}", size(), b.size()));                                       \
        }                                                                                                                                  \
        T       *this_data = this->data();                                                                                                 \
        const T *b_data    = b.data();                                                                                                     \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                         \
            this_data[sentinel] OP b_data[sentinel];                                                                                       \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
    virtual auto operator OP(const RuntimeTensorView<T> &b)->RuntimeTensor<T> & {                                                          \
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
            this->operator()(index) OP b(index);                                                                                           \
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
        return _dims[d];
    }
    virtual auto dims() const -> std::vector<size_t> { return _dims; }

    virtual auto vector_data() const -> const Vector & { return _data; }
    virtual auto vector_data() -> Vector & { return _data; }

    virtual auto stride(int d) const noexcept -> size_t {
        if (d < 0) {
            d += _rank;
        }
        return _strides[d];
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

    virtual const std::string &name() const override { return _name; };

    virtual void set_name(const std::string &new_name) override { _name = new_name; };

    virtual size_t rank() const override { return _rank; }
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
                          public std::enable_shared_from_this<RuntimeTensorView<T>> {
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

        std::vector<size_t> dims, offsets, strides;

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets.push_back(0);
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets.push_back(end);
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
                    offsets.push_back(start);
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

        std::vector<size_t> dims, offsets, strides;

        for (int i = 0; i < _rank; i++) {
            if (i >= slices.size()) {
                dims.push_back(_dims[i]);
                strides.push_back(_strides[i]);
                offsets.push_back(0);
            } else {
                size_t start = slices[i][0], end = slices[i][1];

                if (start == -1 && end >= 0) {
                    offsets.push_back(end);
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
                    offsets.push_back(start);
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
    virtual auto operator OP(const T &b)->RuntimeTensorView<T> & {                                                                         \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < _size; sentinel++) {                                                                          \
            size_t hold = sentinel, ord = 0;                                                                                               \
                                                                                                                                           \
            for (int i = 0; i < _rank; i++) {                                                                                              \
                ord += _strides[i] * (hold / _index_strides[i]);                                                                           \
                hold %= _index_strides[i];                                                                                                 \
            }                                                                                                                              \
                                                                                                                                           \
            _data[ord] OP b;                                                                                                               \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
                                                                                                                                           \
    virtual auto operator OP(const RuntimeTensor<T> &b)->RuntimeTensorView<T> & {                                                          \
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
            _data[ord] OP b.data()[b_ord];                                                                                                 \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }                                                                                                                                      \
    virtual auto operator OP(const RuntimeTensorView<T> &b)->RuntimeTensorView<T> & {                                                      \
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
            _data[ord] OP b.data()[b_ord];                                                                                                 \
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
        return _dims[d];
    }
    virtual auto dims() const -> std::vector<size_t> { return _dims; }

    virtual auto stride(int d) const noexcept -> size_t {
        if (d < 0) {
            d += _rank;
        }
        return _strides[d];
    }

    virtual auto strides() const noexcept -> std::vector<size_t> { return _strides; }

    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        size_t              size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        std::vector<size_t> dim{size};

        return RuntimeTensorView<T>{*this, dim};
    }

    // Returns the linear size of the tensor
    virtual auto size() const -> size_t { return _size; }

    virtual auto full_view_of_underlying() const noexcept -> bool override { return true; }

    virtual const std::string &name() const override { return _name; };

    virtual void set_name(const std::string &new_name) override { _name = new_name; };

    virtual size_t rank() const override { return _rank; }
};

} // namespace einsums