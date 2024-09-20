#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Tensor.hpp"
#include "einsums/utility/IndexUtils.hpp"
#include "einsums/utility/TensorBases.hpp"

#include <memory>
#include <pybind11/pybind11.h>

namespace einsums {

// forward declaration.
template <typename T>
class RuntimeTensor;

// Container definition.
template <typename T>
using SharedRuntimeTensor = std::shared_ptr<RuntimeTensor<T>>;

// forward declaration.
template <typename T>
class RuntimeTensorView;

// Container definition.
template <typename T>
using SharedRuntimeTensorView = std::shared_ptr<RuntimeTensorView<T>>;

/**
 * @class RuntimeTensor
 *
 * @brief Represents a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
class RuntimeTensor : public virtual tensor_props::TensorBase,
                      virtual tensor_props::TypedTensorBase<T>,
                      virtual tensor_props::BasicTensorBase,
                      std::enable_shared_from_this<RuntimeTensor<T>> {
  public:
    using Vector = VectorData<T>;

  private:
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

    RuntimeTensor(const SharedRuntimeTensor<T> &copy) : RuntimeTensor(*copy) {}

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
    explicit RuntimeTensor(const Tensor<T, Rank> &copy) : _rank{Rank}, _dims{copy.dims()}, _strides{copy.strides()}, _name{copy.name()} {
        _data.resize(size());

        std::memcpy(_data.data(), copy.data(), size() * sizeof(T));
    }

    template <size_t Rank>
    explicit RuntimeTensor(const TensorView<T, Rank> &copy) : _rank{Rank}, _dims{copy.dims()} {
        size_t size = 1;
        _strides.resize(rank());

        for (int i = Rank; i >= 0; i--) {
            _strides[i] = size;
            size *= _dims[i];
        }

        _data.resize(size);
    }

    virtual ~RuntimeTensor() = default;

    virtual void zero() { std::memset(_data.data(), 0, _data.size() * sizeof(T)); }

    virtual void set_all(T val) { std::fill(_data.begin(), _data.end(), val); }

    T *data() override { return _data.data(); }

    const T *data() const override { return _data.data(); }

    template <typename Storage>
    T *data(const Storage &index) {
        return &(_data.at(indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    template <typename Storage>
    const T *data(const Storage &index) const {
        return &(_data.at(indices_to_sentinel_negative_check(_strides, _dims, index)));
    }

    template <typename Storage>
    T &operator()(const Storage &index) {
        return _data.at(indices_to_sentinel_negative_check(_strides, _dims, index));
    }

    template <typename Storage>
    const T &operator()(const Storage &index) const {
        return _data.at(indices_to_sentinel_negative_check(_strides, _dims, index));
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

                    if (start < 0 || end < 0 || start >= _dims[i] || end >= _dims[i] || start > end) {
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

                    if (start < 0 || end < 0 || start >= _dims[i] || end >= _dims[i] || start > end) {
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

  private:
#define COPY_CAST_OP(OP, NAME)                                                                                                             \
    template <typename TOther>                                                                                                             \
    void copy_and_cast_##NAME(const pybind11::buffer_info &buffer_info, bool is_view) {                                                    \
        TOther *buffer_data = (TOther *)buffer_info.ptr;                                                                                   \
        if (is_view) {                                                                                                                     \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                     \
                size_t buffer_sent = 0, hold = sentinel;                                                                                   \
                for (int i = 0; i < _rank; i++) {                                                                                          \
                    buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / _strides[i]);                                 \
                    hold %= _strides[i];                                                                                                   \
                }                                                                                                                          \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    _data[sentinel] OP(T)(RemoveComplexT<T>) buffer_data[buffer_sent];                                                     \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    _data[sentinel] OP(T) buffer_data[buffer_sent].real();                                                                 \
                } else if constexpr (IsComplexV<T> && IsComplexV<TOther>) {                                                                \
                    _data[sentinel].real() OP(RemoveComplexT<T>) buffer_data[buffer_sent].real();                                          \
                    _data[sentinel].imag() OP(RemoveComplexT<T>) buffer_data[buffer_sent].imag();                                          \
                } else {                                                                                                                   \
                    _data[sentinel] OP(T) buffer_data[buffer_sent];                                                                        \
                }                                                                                                                          \
            }                                                                                                                              \
        } else {                                                                                                                           \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                     \
                if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                        \
                    _data[sentinel] OP(T)(RemoveComplexT<T>) buffer_data[sentinel];                                                        \
                } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                               \
                    _data[sentinel] OP(T) buffer_data[sentinel].real();                                                                    \
                } else if constexpr (IsComplexV<T> && IsComplexV<TOther>) {                                                                \
                    _data[sentinel].real() OP(RemoveComplexT<T>) buffer_data[sentinel].real();                                             \
                    _data[sentinel].imag() OP(RemoveComplexT<T>) buffer_data[sentinel].imag();                                             \
                } else {                                                                                                                   \
                    _data[sentinel] OP(T) buffer_data[sentinel];                                                                           \
                }                                                                                                                          \
            }                                                                                                                              \
        }                                                                                                                                  \
    }

    COPY_CAST_OP(=, assign)
    COPY_CAST_OP(+=, add)
    COPY_CAST_OP(-=, sub)
    COPY_CAST_OP(*=, mult)
    COPY_CAST_OP(/=, div)
#undef COPY_CAST_OP

  public:
    virtual RuntimeTensor<T> &operator=(const pybind11::buffer &buffer) {
        auto buffer_info = buffer.request();

        if (rank() != buffer_info.ndim) {
            _rank = buffer_info.ndim;
            _dims.resize(_rank);
            _strides.resize(_rank);
        }

        size_t new_size = 1;
        bool   is_view  = false;
        for (int i = buffer_info.ndim; i >= 0; i--) {
            _dims[i]    = buffer_info.shape[i];
            _strides[i] = new_size;
            new_size *= _dims[i];

            if (_strides[i] != buffer_info.strides[i] / buffer_info.itemsize) {
                is_view = true;
            }
        }

        if (new_size != _data.size()) {
            _data.resize(new_size);
        }

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            T *buffer_data = (T *)buffer_info.ptr;
            if (is_view) {
                EINSUMS_OMP_PARALLEL_FOR
                for (size_t sentinel = 0; sentinel < size(); sentinel++) {
                    size_t buffer_sent = 0, hold = sentinel;
                    for (int i = 0; i < _rank; i++) {
                        buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / _strides[i]);
                        hold %= _strides[i];
                    }
                    _data[sentinel] = buffer_data[buffer_sent];
                }
            } else {
                std::memcpy(_data.data(), buffer_data, sizeof(T) * _data.size());
            }
        } else {
            switch (buffer_info.format[0]) {
            case 'b':
                copy_and_cast_assign<char>(buffer_info, is_view);
                break;
            case 'B':
                copy_and_cast_assign<unsigned char>(buffer_info, is_view);
                break;
            case 'h':
                copy_and_cast_assign<short>(buffer_info, is_view);
                break;
            case 'H':
                copy_and_cast_assign<unsigned short>(buffer_info, is_view);
                break;
            case 'i':
                copy_and_cast_assign<int>(buffer_info, is_view);
                break;
            case 'I':
                copy_and_cast_assign<unsigned int>(buffer_info, is_view);
                break;
            case 'q':
                copy_and_cast_assign<long>(buffer_info, is_view);
                break;
            case 'Q':
                copy_and_cast_assign<unsigned long>(buffer_info, is_view);
                break;
            case 'f':
                copy_and_cast_assign<float>(buffer_info, is_view);
                break;
            case 'd':
                copy_and_cast_assign<double>(buffer_info, is_view);
                break;
            case 'g':
                copy_and_cast_assign<long double>(buffer_info, is_view);
                break;
            case 'Z':
                switch (buffer_info.format[1]) {
                case 'f':
                    copy_and_cast_assign<std::complex<float>>(buffer_info, is_view);
                    break;
                case 'd':
                    copy_and_cast_assign<std::complex<double>>(buffer_info, is_view);
                    break;
                case 'g':
                    copy_and_cast_assign<std::complex<long double>>(buffer_info, is_view);
                    break;
                }
                [[fallthrough]];
            default:
                throw EINSUMSEXCEPTION("Can not convert format descriptor " + buffer_info.format + " to " + type_name<T>() + "!");
            }
        }
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
        EINSUMS_OMP_PARALLEL {                                                                                                             \
            auto tid       = omp_get_thread_num();                                                                                         \
            auto chunksize = _data.size() / omp_get_num_threads();                                                                         \
            auto abegin    = _data.begin() + chunksize * tid;                                                                              \
            auto bbegin    = b._data.begin() + chunksize * tid;                                                                            \
            auto aend      = (tid == omp_get_num_threads() - 1) ? _data.end() : abegin + chunksize;                                        \
            auto j         = bbegin;                                                                                                       \
            EINSUMS_OMP_SIMD for (auto i = abegin; i < aend; i++, j++) {                                                                   \
                (*i) OP(*j);                                                                                                               \
            }                                                                                                                              \
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
            tensor_algebra::detail::sentinel_to_indices(sentinel, _strides, index);                                                        \
            operator()(index) = b(index);                                                                                                  \
        }                                                                                                                                  \
    }                                                                                                                                      \
    virtual RuntimeTensor<T> &operator OP(const pybind11::buffer & buffer) {                                                               \
        auto buffer_info = buffer.request();                                                                                               \
                                                                                                                                           \
        if (rank() != buffer_info.ndim) {                                                                                                  \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different rank!");                                     \
        }                                                                                                                                  \
                                                                                                                                           \
        bool is_view = false;                                                                                                              \
        for (int i = buffer_info.ndim; i >= 0; i--) {                                                                                      \
            if (_dims[i] != buffer_info.shape[i]) {                                                                                        \
                throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different dimensions!");                           \
            }                                                                                                                              \
                                                                                                                                           \
            if (_strides[i] != buffer_info.strides[i] / buffer_info.itemsize) {                                                            \
                is_view = true;                                                                                                            \
            }                                                                                                                              \
        }                                                                                                                                  \
                                                                                                                                           \
        if (buffer_info.item_type_is_equivalent_to<T>()) {                                                                                 \
            T *buffer_data = (T *)buffer_info.ptr;                                                                                         \
            if (is_view) {                                                                                                                 \
                EINSUMS_OMP_PARALLEL_FOR                                                                                                   \
                for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                 \
                    size_t buffer_sent = 0, hold = sentinel;                                                                               \
                    for (int i = 0; i < _rank; i++) {                                                                                      \
                        buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / _strides[i]);                             \
                        hold %= _strides[i];                                                                                               \
                    }                                                                                                                      \
                    _data[sentinel] OP buffer_data[buffer_sent];                                                                           \
                }                                                                                                                          \
            } else {                                                                                                                       \
                EINSUMS_OMP_PARALLEL_FOR                                                                                                   \
                for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                 \
                    _data[sentinel] OP buffer_data[sentinel];                                                                              \
                }                                                                                                                          \
            }                                                                                                                              \
        } else {                                                                                                                           \
            switch (buffer_info.format[0]) {                                                                                               \
            case 'b':                                                                                                                      \
                copy_and_cast_##NAME<char>(buffer_info, is_view);                                                                          \
                break;                                                                                                                     \
            case 'B':                                                                                                                      \
                copy_and_cast_##NAME<unsigned char>(buffer_info, is_view);                                                                 \
                break;                                                                                                                     \
            case 'h':                                                                                                                      \
                copy_and_cast_##NAME<short>(buffer_info, is_view);                                                                         \
                break;                                                                                                                     \
            case 'H':                                                                                                                      \
                copy_and_cast_##NAME<unsigned short>(buffer_info, is_view);                                                                \
                break;                                                                                                                     \
            case 'i':                                                                                                                      \
                copy_and_cast_##NAME<int>(buffer_info, is_view);                                                                           \
                break;                                                                                                                     \
            case 'I':                                                                                                                      \
                copy_and_cast_##NAME<unsigned int>(buffer_info, is_view);                                                                  \
                break;                                                                                                                     \
            case 'q':                                                                                                                      \
                copy_and_cast_##NAME<long>(buffer_info, is_view);                                                                          \
                break;                                                                                                                     \
            case 'Q':                                                                                                                      \
                copy_and_cast_##NAME<unsigned long>(buffer_info, is_view);                                                                 \
                break;                                                                                                                     \
            case 'f':                                                                                                                      \
                copy_and_cast_##NAME<float>(buffer_info, is_view);                                                                         \
                break;                                                                                                                     \
            case 'd':                                                                                                                      \
                copy_and_cast_##NAME<double>(buffer_info, is_view);                                                                        \
                break;                                                                                                                     \
            case 'g':                                                                                                                      \
                copy_and_cast_##NAME<long double>(buffer_info, is_view);                                                                   \
                break;                                                                                                                     \
            case 'Z':                                                                                                                      \
                switch (buffer_info.format[1]) {                                                                                           \
                case 'f':                                                                                                                  \
                    copy_and_cast_##NAME<std::complex<float>>(buffer_info, is_view);                                                       \
                    break;                                                                                                                 \
                case 'd':                                                                                                                  \
                    copy_and_cast_##NAME<std::complex<double>>(buffer_info, is_view);                                                      \
                    break;                                                                                                                 \
                case 'g':                                                                                                                  \
                    copy_and_cast_##NAME<std::complex<long double>>(buffer_info, is_view);                                                 \
                    break;                                                                                                                 \
                }                                                                                                                          \
                [[fallthrough]];                                                                                                           \
            default:                                                                                                                       \
                throw EINSUMSEXCEPTION("Can not convert format descriptor " + buffer_info.format + " to " + type_name<T>() + "!");         \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=, mult)
    OPERATOR(/=, div)
    OPERATOR(+=, add)
    OPERATOR(-=, sub)

#undef OPERATOR

    virtual auto dim(int d) const -> size_t override {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims[d];
    }
    virtual auto dims() const -> std::vector<size_t> { return _dims; }

    virtual auto vector_data() const -> const Vector & { return _data.get(); }
    virtual auto vector_data() -> Vector & { return _data.get(); }

    [[nodiscard]] virtual auto stride(int d) const noexcept -> size_t override {
        if (d < 0) {
            d += _rank;
        }
        return _strides[d];
    }

    virtual auto strides() const noexcept -> std::vector<size_t> { return _strides; }

    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return RuntimeTensorView<T>{*this, dim};
    }

    // Returns the linear size of the tensor
    [[nodiscard]] virtual auto size() const -> size_t { return _data.size(); }

    virtual auto full_view_of_underlying() const noexcept -> bool override { return true; }

    virtual const std::string &name() const override { return _name; };

    virtual void set_name(const std::string &new_name) override { _name = new_name; };

  private:
    T &subscript_to_val(const pybind11::tuple &args) {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    const T &subscript_to_val(const pybind11::tuple &args) const {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) const {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    void set_value_at(T value, const std::vector<size_t> &index) {
        T &target = this->operator()(index);
        target    = value;
        return target;
    }

    void assign_to_view(const pybind11::buffer &view, const pybind11::tuple &args) {
        auto this_view = subscript_to_view(args);

        this_view = view;
    }

    void assign_to_view(T value, const pybind11::tuple &args) {
        auto this_view = subscript_to_view(args);

        this_view = value;
    }

  public:
    virtual pybind11::object subscript(const pybind11::tuple &args) {
        if (args.size() < _rank) {
            return pybind11::cast(subscript_to_view(args));
        }
        if (args.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                return pybind11::cast(subscript_to_view(args));
            }
        }

        return pybind11::cast(subscript_to_val(args));
    }

    virtual const pybind11::object subscript(const pybind11::tuple &args) const {
        if (args.size() < _rank) {
            return pybind11::cast(subscript_to_view(args));
        }
        if (args.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                return pybind11::cast(subscript_to_view(args));
            }
        }

        return pybind11::cast(subscript_to_val(args));
    }

    pybind11::object assign_values(const pybind11::buffer &value, const pybind11::tuple &index) {
        if (index.size() < _rank) {
            return pybind11::cast(assign_to_view(value, index));
        }
        if (index.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < index.size(); i++) {
            const auto &arg = index[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                assign_to_view(value, index);
                return subscript(index);
            }
        }
        throw EINSUMSEXCEPTION("Can not assign buffer object to a single position!");
    }

    pybind11::object assign_values(T value, const pybind11::tuple &index) {
        if (index.size() < _rank) {
            return pybind11::cast(assign_to_view(value, index));
        }
        if (index.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < index.size(); i++) {
            const auto &arg = index[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                assign_to_view(value, index);
                return subscript(index);
            }
        }

        T &target = subscript_to_val(index);
        target    = value;

        return subscript(index);
    }

    virtual RuntimeTensorView<T> subscript(const pybind11::slice &arg) {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(arg).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass);
    }

    virtual const RuntimeTensorView<T> subscript(const pybind11::slice &arg) const {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(arg).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass);
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, const pybind11::slice &index) {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(index).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass) = value;
    }

    RuntimeTensorView<T> assign_values(T value, const pybind11::slice &index) {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(index).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass) = value;
    }

    virtual pybind11::object subscript(int index) {
        if (_rank == 1) {
            return pybind11::cast(this->operator()(index));
        } else {
            return pybind11::cast(this->operator()(std::vector<Range>{Range{-1, index}}));
        }
    }

    virtual const pybind11::object subscript(int index) const {
        if (_rank == 1) {
            return pybind11::cast(this->operator()(index));
        } else {
            return pybind11::cast(this->operator()(std::vector<Range>{Range{-1, index}}));
        }
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, int index) {
        if (_rank <= 1) {
            throw EINSUMSEXCEPTION("Can not assign buffer to a single position!");
        }

        return this->operator()(std::vector<Range>{Range{-1, index}}) = value;
    }

    pybind11::object assign_values(T value, int index) {
        if (_rank <= 1) {
            T &target = this->operator()({index});
            target    = value;
            return pybind11::cast(target);
        }

        auto view = this->operator()(std::vector<Range>{Range{-1, index}});
        view      = value;
        return pybind11::cast(view);
    }

    virtual size_t rank() const { return _rank; }
}; // namespace einsums

/**
 * @class RuntimeTensorView
 *
 * @brief Represents a view of a tensor whose properties can be determined at runtime but not compile time.
 */
template <typename T>
class RuntimeTensorView : public virtual tensor_props::TensorViewBase<RuntimeTensor<T>>,
                          virtual tensor_props::TypedTensorBase<T>,
                          virtual tensor_props::BasicTensorBase,
                          std::enable_shared_from_this<RuntimeTensorView<T>> {
  private:
    T                  *_data;
    std::string         _name{"(unnamed view)"};
    std::vector<size_t> _dims, _strides, _index_strides;
    size_t              _rank{0}, _size{0}, _alloc_size{0};
    bool                _full_view{false};

  public:
    RuntimeTensorView() = default;

    RuntimeTensorView(const RuntimeTensorView<T> &copy) = default;

    RuntimeTensorView(const SharedRuntimeTensorView<T> &copy) : RuntimeTensorView(*copy) {}

    RuntimeTensorView(RuntimeTensor<T> &view)
        : _data{view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(const RuntimeTensor<T> &view)
        : _data{view.data()}, _name{view.name()}, _dims{view.dims()}, _strides{view.strides()}, _rank{view.rank()}, _size{view.size()},
          _full_view{true}, _index_strides(view.rank()) {
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

    RuntimeTensorView(const RuntimeTensor<T> &other, const std::vector<size_t> &dims)
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

        _data = other.data();
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

        _data = other.data(offsets);
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

        _data = other.data(offsets);
        tensor_algebra::detail::dims_to_strides(_dims, _index_strides);
    }

    RuntimeTensorView(pybind11::buffer &buffer) {
        pybind11::buffer_info buffer_info = buffer.request(true);

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            _data = buffer_info.ptr;
        } else {
            throw EINSUMSEXCEPTION("Can not create RuntimeTensorView from buffer whose type does not match!");
        }

        _rank = buffer_info.ndim;
        _dims.resize(_rank);
        _strides.resize(_rank);
        _index_strides.resize(_rank);
        _size       = 1;
        _alloc_size = buffer_info.shape[0] * buffer_info.strides[0];

        for (int i = _rank - 1; i >= 0; i--) {
            _strides[i]       = buffer_info.strides[i] / buffer_info.itemsize;
            _dims[i]          = buffer_info.shape[i];
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    template <size_t Rank>
    explicit RuntimeTensorView(TensorView<T, Rank> &copy)
        : _data{copy.data()}, _dims{copy.dims()}, _strides{copy.strides()}, _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank; i >= 0; i--) {
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    template <size_t Rank>
    explicit RuntimeTensorView(const TensorView<T, Rank> &copy)
        : _data{copy.data()}, _dims{copy.dims()}, _strides{copy.strides()}, _rank{Rank}, _full_view{copy.full_view_of_underlying()} {
        _index_strides.resize(Rank);

        _size = 1;
        for (int i = Rank; i >= 0; i--) {
            _index_strides[i] = _size;
            _size *= _dims[i];
        }
    }

    template <size_t Rank>
    explicit RuntimeTensorView(Tensor<T, Rank> &copy)
        : _data{copy.data()}, _dims{copy.dims()}, _strides{copy.strides()}, _rank{Rank}, _full_view{true}, _index_strides{copy.strides()},
          _size{copy.size()} {}

    template <size_t Rank>
    explicit RuntimeTensorView(const Tensor<T, Rank> &copy)
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

    T *data() override { return _data; }

    const T *data() const override { return _data; }

    template <typename Storage>
    T *data(const Storage &index) {
        return &(_data[indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    template <typename Storage>
    const T *data(const Storage &index) const {
        return &(_data[indices_to_sentinel_negative_check(_strides, _dims, index)]);
    }

    template <typename Storage>
    T &operator()(const Storage &index) {
        return _data[indices_to_sentinel_negative_check(_strides, _dims, index)];
    }

    template <typename Storage>
    const T &operator()(const Storage &index) const {
        return _data[indices_to_sentinel_negative_check(_strides, _dims, index)];
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

                    if (start < 0 || end < 0 || start >= _dims[i] || end >= _dims[i] || start > end) {
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

                    if (start < 0 || end < 0 || start >= _dims[i] || end >= _dims[i] || start > end) {
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

  private:
#define COPY_CAST_OP(OP, NAME)                                                                                                             \
    template <typename TOther>                                                                                                             \
    void copy_and_cast_##NAME(const pybind11::buffer_info &buffer_info) {                                                                  \
        TOther *buffer_data = (TOther *)buffer_info.ptr;                                                                                   \
        EINSUMS_OMP_PARALLEL_FOR                                                                                                           \
        for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                         \
            size_t buffer_sent = 0, hold = sentinel, ord = 0;                                                                              \
            for (int i = 0; i < _rank; i++) {                                                                                              \
                ord += _strides[i] * (hold / _index_strides[i]);                                                                           \
                buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / _strides[i]);                                     \
                hold %= _strides[i];                                                                                                       \
            }                                                                                                                              \
            if constexpr (IsComplexV<T> && !IsComplexV<TOther> && !std::is_same_v<RemoveComplexT<T>, TOther>) {                            \
                _data[ord] OP(T)(RemoveComplexT<T>) buffer_data[buffer_sent];                                                              \
            } else if constexpr (!IsComplexV<T> && IsComplexV<TOther>) {                                                                   \
                _data[ord] OP(T) buffer_data[buffer_sent].real();                                                                          \
            } else if constexpr (IsComplexV<T> && IsComplexV<TOther>) {                                                                    \
                _data[ord].real() OP(RemoveComplexT<T>) buffer_data[buffer_sent].real();                                                   \
                _data[ord].imag() OP(RemoveComplexT<T>) buffer_data[buffer_sent].imag();                                                   \
            } else {                                                                                                                       \
                _data[ord] OP(T) buffer_data[buffer_sent];                                                                                 \
            }                                                                                                                              \
        }                                                                                                                                  \
    }

    COPY_CAST_OP(=, assign)
    COPY_CAST_OP(+=, add)
    COPY_CAST_OP(-=, sub)
    COPY_CAST_OP(*=, mult)
    COPY_CAST_OP(/=, div)
#undef COPY_CAST_OP

  public:
    virtual RuntimeTensorView<T> &operator=(const pybind11::buffer &buffer) {
        auto buffer_info = buffer.request();

        if (rank() != buffer_info.ndim) {
            throw EINSUMSEXCEPTION("Can not change the rank of a runtime tensor view when assigning!");
        }

        for (int i = buffer_info.ndim; i >= 0; i--) {
            if (_dims[i] != buffer_info.shape[i]) {
                throw EINSUMSEXCEPTION("Can not assign buffer to runtime tensor view with different shapes!");
            }
        }

        if (buffer_info.item_type_is_equivalent_to<T>()) {
            T *buffer_data = (T *)buffer_info.ptr;
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t sentinel = 0; sentinel < size(); sentinel++) {
                size_t buffer_sent = 0, hold = sentinel, ord = 0;
                for (int i = 0; i < _rank; i++) {
                    ord += _strides[i] * (hold / _index_strides[i]);
                    buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / _index_strides[i]);
                    hold %= _index_strides[i];
                }
                _data[ord] = buffer_data[buffer_sent];
            }
        } else {
            switch (buffer_info.format[0]) {
            case 'b':
                copy_and_cast_assign<char>(buffer_info);
                break;
            case 'B':
                copy_and_cast_assign<unsigned char>(buffer_info);
                break;
            case 'h':
                copy_and_cast_assign<short>(buffer_info);
                break;
            case 'H':
                copy_and_cast_assign<unsigned short>(buffer_info);
                break;
            case 'i':
                copy_and_cast_assign<int>(buffer_info);
                break;
            case 'I':
                copy_and_cast_assign<unsigned int>(buffer_info);
                break;
            case 'q':
                copy_and_cast_assign<long>(buffer_info);
                break;
            case 'Q':
                copy_and_cast_assign<unsigned long>(buffer_info);
                break;
            case 'f':
                copy_and_cast_assign<float>(buffer_info);
                break;
            case 'd':
                copy_and_cast_assign<double>(buffer_info);
                break;
            case 'g':
                copy_and_cast_assign<long double>(buffer_info);
                break;
            case 'Z':
                switch (buffer_info.format[1]) {
                case 'f':
                    copy_and_cast_assign<std::complex<float>>(buffer_info);
                    break;
                case 'd':
                    copy_and_cast_assign<std::complex<double>>(buffer_info);
                    break;
                case 'g':
                    copy_and_cast_assign<std::complex<long double>>(buffer_info);
                    break;
                }
                [[fallthrough]];
            default:
                throw EINSUMSEXCEPTION("Can not convert format descriptor " + buffer_info.format + " to " + type_name<T>() + "!");
            }
        }
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
    }                                                                                                                                      \
    virtual RuntimeTensor<T> &operator OP(const pybind11::buffer & buffer) {                                                               \
        auto buffer_info = buffer.request();                                                                                               \
                                                                                                                                           \
        if (rank() != buffer_info.ndim) {                                                                                                  \
            throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different rank!");                                     \
        }                                                                                                                                  \
        for (int i = buffer_info.ndim; i >= 0; i--) {                                                                                      \
            if (_dims[i] != buffer_info.shape[i]) {                                                                                        \
                throw EINSUMSEXCEPTION("Can not perform " #OP " with buffer object with different dimensions!");                           \
            }                                                                                                                              \
        }                                                                                                                                  \
                                                                                                                                           \
        if (buffer_info.item_type_is_equivalent_to<T>()) {                                                                                 \
            T *buffer_data = (T *)buffer_info.ptr;                                                                                         \
            EINSUMS_OMP_PARALLEL_FOR                                                                                                       \
            for (size_t sentinel = 0; sentinel < size(); sentinel++) {                                                                     \
                size_t buffer_sent = 0, hold = sentinel, ord = 0;                                                                          \
                for (int i = 0; i < _rank; i++) {                                                                                          \
                    ord += _strides[i] * (hold / _index_strides[i]);                                                                       \
                    buffer_sent += (buffer_info.strides[i] / buffer_info.itemsize) * (hold / _index_strides[i]);                           \
                    hold %= _index_strides[i];                                                                                             \
                }                                                                                                                          \
                _data[ord] OP buffer_data[buffer_sent];                                                                                    \
            }                                                                                                                              \
        } else {                                                                                                                           \
            switch (buffer_info.format[0]) {                                                                                               \
            case 'b':                                                                                                                      \
                copy_and_cast_##NAME<char>(buffer_info);                                                                                   \
                break;                                                                                                                     \
            case 'B':                                                                                                                      \
                copy_and_cast_##NAME<unsigned char>(buffer_info);                                                                          \
                break;                                                                                                                     \
            case 'h':                                                                                                                      \
                copy_and_cast_##NAME<short>(buffer_info);                                                                                  \
                break;                                                                                                                     \
            case 'H':                                                                                                                      \
                copy_and_cast_##NAME<unsigned short>(buffer_info);                                                                         \
                break;                                                                                                                     \
            case 'i':                                                                                                                      \
                copy_and_cast_##NAME<int>(buffer_info);                                                                                    \
                break;                                                                                                                     \
            case 'I':                                                                                                                      \
                copy_and_cast_##NAME<unsigned int>(buffer_info);                                                                           \
                break;                                                                                                                     \
            case 'q':                                                                                                                      \
                copy_and_cast_##NAME<long>(buffer_info);                                                                                   \
                break;                                                                                                                     \
            case 'Q':                                                                                                                      \
                copy_and_cast_##NAME<unsigned long>(buffer_info);                                                                          \
                break;                                                                                                                     \
            case 'f':                                                                                                                      \
                copy_and_cast_##NAME<float>(buffer_info);                                                                                  \
                break;                                                                                                                     \
            case 'd':                                                                                                                      \
                copy_and_cast_##NAME<double>(buffer_info);                                                                                 \
                break;                                                                                                                     \
            case 'g':                                                                                                                      \
                copy_and_cast_##NAME<long double>(buffer_info);                                                                            \
                break;                                                                                                                     \
            case 'Z':                                                                                                                      \
                switch (buffer_info.format[1]) {                                                                                           \
                case 'f':                                                                                                                  \
                    copy_and_cast_##NAME<std::complex<float>>(buffer_info);                                                                \
                    break;                                                                                                                 \
                case 'd':                                                                                                                  \
                    copy_and_cast_##NAME<std::complex<double>>(buffer_info);                                                               \
                    break;                                                                                                                 \
                case 'g':                                                                                                                  \
                    copy_and_cast_##NAME<std::complex<long double>>(buffer_info);                                                          \
                    break;                                                                                                                 \
                }                                                                                                                          \
                [[fallthrough]];                                                                                                           \
            default:                                                                                                                       \
                throw EINSUMSEXCEPTION("Can not convert format descriptor " + buffer_info.format + " to " + type_name<T>() + "!");         \
            }                                                                                                                              \
        }                                                                                                                                  \
        return *this;                                                                                                                      \
    }

    OPERATOR(*=, mult)
    OPERATOR(/=, div)
    OPERATOR(+=, add)
    OPERATOR(-=, sub)

#undef OPERATOR

    virtual auto dim(int d) const -> size_t override {
        // Add support for negative indices.
        if (d < 0) {
            d += _rank;
        }
        return _dims[d];
    }
    virtual auto dims() const -> std::vector<size_t> { return _dims; }

    virtual auto stride(int d) const noexcept -> size_t override {
        if (d < 0) {
            d += _rank;
        }
        return _strides[d];
    }

    virtual auto strides() const noexcept -> std::vector<size_t> { return _strides; }

    virtual auto to_rank_1_view() const -> RuntimeTensorView<T> {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return RuntimeTensorView<T>{*this, dim};
    }

    // Returns the linear size of the tensor
    virtual auto size() const -> size_t { return _size; }

    virtual auto full_view_of_underlying() const noexcept -> bool override { return true; }

    virtual const std::string &name() const override { return _name; };

    virtual void set_name(const std::string &new_name) override { _name = new_name; };

  private:
    T &subscript_to_val(const pybind11::tuple &args) {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    const T &subscript_to_val(const pybind11::tuple &args) const {
        std::vector<size_t> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            pass[i] = pybind11::cast<size_t>(arg);
        }
        return this->operator()(pass);
    }

    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    RuntimeTensorView<T> subscript_to_view(const pybind11::tuple &args) const {
        std::vector<Range> pass(args.size());

        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::int_>(arg)) {
                pass[i] = Range{-1, pybind11::cast<size_t>(arg)};
            } else if (pybind11::isinstance<pybind11::slice>(arg)) {
                size_t start, stop, step, slice_length;
                (pybind11::cast<pybind11::slice>(arg)).compute(this->dim(i), &start, &stop, &step, &slice_length);
                if (step != 1) {
                    throw EINSUMSEXCEPTION("Can not handle slices with step sizes other than 1!");
                }
                pass[i] = Range{start, stop};
            }
        }
        return this->operator()(pass);
    }

    void set_value_at(T value, const std::vector<size_t> &index) {
        T &target = this->operator()(index);
        target    = value;
        return target;
    }

    void assign_to_view(const pybind11::buffer &view, const pybind11::tuple &args) {
        auto this_view = subscript_to_view(args);

        this_view = view;
    }

    void assign_to_view(T value, const pybind11::tuple &args) {
        auto this_view = subscript_to_view(args);

        this_view = value;
    }

  public:
    virtual pybind11::object subscript(const pybind11::tuple &args) {
        if (args.size() < _rank) {
            return pybind11::cast(subscript_to_view(args));
        }
        if (args.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                return pybind11::cast(subscript_to_view(args));
            }
        }

        return pybind11::cast(subscript_to_val(args));
    }

    virtual const pybind11::object subscript(const pybind11::tuple &args) const {
        if (args.size() < _rank) {
            return pybind11::cast(subscript_to_view(args));
        }
        if (args.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < args.size(); i++) {
            const auto &arg = args[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                return pybind11::cast(subscript_to_view(args));
            }
        }

        return pybind11::cast(subscript_to_val(args));
    }

    pybind11::object assign_values(const pybind11::buffer &value, const pybind11::tuple &index) {
        if (index.size() < _rank) {
            return pybind11::cast(assign_to_view(value, index));
        }
        if (index.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < index.size(); i++) {
            const auto &arg = index[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                assign_to_view(value, index);
                return subscript(index);
            }
        }
        throw EINSUMSEXCEPTION("Can not assign buffer object to a single position!");
    }

    pybind11::object assign_values(T value, const pybind11::tuple &index) {
        if (index.size() < _rank) {
            return pybind11::cast(assign_to_view(value, index));
        }
        if (index.size() > _rank) {
            throw EINSUMSEXCEPTION("Too many indices passed to tensor!");
        }
        for (int i = 0; i < index.size(); i++) {
            const auto &arg = index[i];

            if (pybind11::isinstance<pybind11::slice>(arg)) {
                assign_to_view(value, index);
                return subscript(index);
            }
        }

        T &target = subscript_to_val(index);
        target    = value;

        return subscript(index);
    }

    virtual RuntimeTensorView<T> subscript(const pybind11::slice &arg) {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(arg).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass);
    }

    virtual const RuntimeTensorView<T> subscript(const pybind11::slice &arg) const {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(arg).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass);
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, const pybind11::slice &index) {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(index).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass) = value;
    }

    RuntimeTensorView<T> assign_values(T value, const pybind11::slice &index) {
        size_t start, end, step, length;

        pybind11::cast<pybind11::slice>(index).compute(_dims[0], &start, &end, &step, &length);

        if (step != 1) {
            throw EINSUMSEXCEPTION("Can not handle slices with steps not equal to 1!");
        }

        std::vector<Range> pass{Range{start, end}};

        return this->operator()(pass) = value;
    }

    virtual pybind11::object subscript(int index) {
        if (_rank == 1) {
            return pybind11::cast(this->operator()(index));
        } else {
            return pybind11::cast(this->operator()(std::vector<Range>{Range{-1, index}}));
        }
    }

    virtual const pybind11::object subscript(int index) const {
        if (_rank == 1) {
            return pybind11::cast(this->operator()(index));
        } else {
            return pybind11::cast(this->operator()(std::vector<Range>{Range{-1, index}}));
        }
    }

    RuntimeTensorView<T> assign_values(const pybind11::buffer &value, int index) {
        if (_rank <= 1) {
            throw EINSUMSEXCEPTION("Can not assign buffer to a single position!");
        }

        return this->operator()(std::vector<Range>{Range{-1, index}}) = value;
    }

    pybind11::object assign_values(T value, int index) {
        if (_rank <= 1) {
            T &target = this->operator()({index});
            target    = value;
            return pybind11::cast(target);
        }

        auto view = this->operator()(std::vector<Range>{Range{-1, index}});
        view      = value;
        return pybind11::cast(view);
    }

    virtual size_t rank() const { return _rank; }
};

} // namespace einsums