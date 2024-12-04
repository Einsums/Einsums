#pragma once

#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/H5.hpp>

#include "Einsums/Errors/Error.hpp"
#include "Einsums/Errors/ThrowException.hpp"
#include "Einsums/TypeSupport/AreAllConvertible.hpp"
#include "Einsums/TypeSupport/Arguments.hpp"

namespace einsums {

/**
 * @struct DiskTensor
 *
 * @brief Represents a tensor stored on disk.
 *
 * @tparam T The data type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct DiskTensor final : public virtual tensor_base::DiskTensor,
                          virtual tensor_base::Tensor<T, Rank>,
                          virtual tensor_base::LockableTensor {
    DiskTensor() = default;

    DiskTensor(DiskTensor const &) = default;

    DiskTensor(DiskTensor &&) noexcept = default;

    ~DiskTensor() override = default;

    template <typename... Dims>
    explicit DiskTensor(h5::fd_t &file, std::string name, Chunk<sizeof...(Dims)> chunk, Dims... dims)
        : _file{file}, _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
        struct Stride {
            size_t value{1};

            Stride() = default;

            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                println("Unable to open disk tensor '{}'", _name);
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, h5::current_dims{static_cast<size_t>(dims)...},
                                      h5::chunk{chunk} /* | h5::gzip{9} | h5::fill_value<T>(0.0) */);
            } catch (std::exception &e) {
                println("Unable to create disk tensor '{}': {}", _name, e.what());
                std::abort();
            }
        }
    }

    template <typename... Dims, typename = std::enable_if_t<are_all_convertible_v<size_t, Dims...>::value>>
    explicit DiskTensor(h5::fd_t &file, std::string name, Dims... dims)
        : _file{file}, _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
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

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());

        std::array<size_t, Rank> chunk_temp{};
        chunk_temp[0] = 1;
        for (int i = 1; i < Rank; i++) {
            constexpr size_t chunk_min{64};
            if (_dims[i] < chunk_min)
                chunk_temp[i] = _dims[i];
            else
                chunk_temp[i] = chunk_min;
        }

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                println("Unable to open disk tensor '{}'", _name);
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, h5::current_dims{static_cast<size_t>(dims)...},
                                      h5::chunk{chunk_temp} /* | h5::gzip{9} | h5::fill_value<T>(0.0) */);
            } catch (std::exception &e) {
                println("Unable to create disk tensor '{}': {}", _name, e.what());
                std::abort();
            }
        }
    }

    /// Constructs a DiskTensor shaped like the provided Tensor. Data from the provided tensor
    /// is NOT saved.
    explicit DiskTensor(h5::fd_t &file, Tensor<T, Rank> const &tensor) : _file{file}, _name{tensor.name()} {
        // Save dimension information from the provided tensor.
        h5::current_dims cdims;
        for (int i = 0; i < Rank; i++) {
            _dims[i] = tensor.dim(i);
            cdims[i] = _dims[i];
        }

        struct Stride {
            size_t value{1};

            Stride() = default;

            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());

        std::array<size_t, Rank> chunk_temp{};
        chunk_temp[0] = 1;
        for (int i = 1; i < Rank; i++) {
            constexpr size_t chunk_min{64};
            if (_dims[i] < chunk_min)
                chunk_temp[i] = _dims[i];
            else
                chunk_temp[i] = chunk_min;
        }

        // Check to see if the data set exists
        if (H5Lexists(_file, _name.c_str(), H5P_DEFAULT) > 0) {
            _existed = true;
            try {
                _disk = h5::open(_file, _name);
            } catch (std::exception &e) {
                println("Unable to open disk tensor '%s'", _name.c_str());
                std::abort();
            }
        } else {
            _existed = false;
            // Use h5cpp create data structure on disk.  Refrain from allocating any memory
            try {
                _disk = h5::create<T>(_file, _name, cdims, h5::chunk{chunk_temp} /*| h5::gzip{9} | h5::fill_value<T>(0.0)*/);
            } catch (std::exception &e) {
                println("Unable to create disk tensor '%s'", _name.c_str());
                std::abort();
            }
        }
    }

    // Provides ability to store another tensor to a part of a disk tensor.

    size_t    dim(int d) const override { return _dims[d]; }
    Dim<Rank> dims() const override { return _dims; }

    [[nodiscard]] auto existed() const -> bool { return _existed; }

    [[nodiscard]] auto disk() -> h5::ds_t & { return _disk; }

    // void _write(Tensor<T, Rank> &data) { h5::write(disk(), data); }

    std::string const &name() const override { return _name; }

    void set_name(std::string const &new_name) override { _name = new_name; }

    size_t stride(int d) const noexcept { return _strides[d]; }

    // This creates a Disk object with its Rank being equal to the number of All{} parameters
    // Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index)
        -> std::enable_if_t<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0,
                            DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>> {
        // Get positions of All
        auto all_positions = get_array_from_tuple<std::array<int, count_of_type<AllT, MultiIndex...>()>>(
            arguments::positions_of_type<AllT, MultiIndex...>());
        auto index_positions = get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(
            arguments::positions_of_type<size_t, MultiIndex...>());
        auto range_positions = get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(
            arguments::positions_of_type<Range, MultiIndex...>());

        auto const &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Stride<Rank> strides{};
        Count<Rank>  counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            strides[value] = _strides[value];
            counts[value]  = _dims[value];
            // dims_all[i] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = get_from_tuple<Range>(indices, value)[0];
            counts[value]  = get_from_tuple<Range>(indices, value)[1] - get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                               offsets, strides);
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const
        -> std::enable_if_t<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0,
                            DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank> const> {
        // Get positions of All
        auto all_positions = get_array_from_tuple<std::array<int, count_of_type<AllT, MultiIndex...>()>>(
            arguments::positions_of_type<AllT, MultiIndex...>());
        auto index_positions = get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(
            arguments::positions_of_type<size_t, MultiIndex...>());
        auto range_positions = get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(
            arguments::positions_of_type<Range, MultiIndex...>());

        auto const &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Stride<Rank> strides{};
        Count<Rank>  counts{};

        std::fill(counts.begin(), counts.end(), 1.0);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = get_from_tuple<size_t>(indices, value);
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            strides[value] = _strides[value];
            counts[value]  = _dims[value];
            // dims_all[i] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = get_from_tuple<Range>(indices, value)[0];
            counts[value]  = get_from_tuple<Range>(indices, value)[1] - get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (auto cnt : counts) {
            if (cnt > 1) {
                dims_all[dims_index++] = cnt;
            }
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>(), Rank>(*this, dims_all, counts,
                                                                                                               offsets, strides);
    }

  private:
    h5::fd_t &_file;

    std::string  _name;
    Dim<Rank>    _dims;
    Stride<Rank> _strides;

    h5::ds_t _disk;

    // Did the entry already exist on disk? Doesn't indicate validity of the data just the existence of the entry.
    bool _existed{false};
};

/**
 * @struct DiskView
 *
 * @brief Represents a view of a DiskTensor.
 *
 * @tparam T The data type stored by the tensor.
 * @tparam ViewRank The rank of the view.
 * @tparam Rank The rank of the DiskTensor being viewed.
 */
template <typename T, size_t ViewRank, size_t Rank>
struct DiskView final : virtual tensor_base::DiskTensor,
                        virtual tensor_base::TensorView<T, ViewRank, DiskTensor<T, Rank>>,
                        virtual tensor_base::LockableTensor {
    DiskView(einsums::DiskTensor<T, Rank> &parent, Dim<ViewRank> const &dims, Count<Rank> const &counts, Offset<Rank> const &offsets,
             Stride<Rank> const &strides)
        : _parent(parent), _dims(dims), _counts(counts), _offsets(offsets), _strides(strides), _tensor{_dims} {
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    };

    DiskView(einsums::DiskTensor<T, Rank> const &parent, Dim<ViewRank> const &dims, Count<Rank> const &counts, Offset<Rank> const &offsets,
             Stride<Rank> const &strides)
        : _parent(const_cast<einsums::DiskTensor<T, Rank> &>(parent)), _dims(dims), _counts(counts), _offsets(offsets), _strides(strides),
          _tensor{_dims} {
        // Section const section("DiskView constructor");
        h5::read<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
        set_read_only(true);
    };

    DiskView(DiskView const &) = default;

    DiskView(DiskView &&) noexcept = default;

    ~DiskView() { put(); }

    void set_read_only(bool readOnly) { _readOnly = readOnly; }

    auto operator=(T const *other) -> DiskView & {
        // Can't perform checks on data. Assume the user knows what they're doing.
        // This function is used when interfacing with libint2.

        // Save the data to disk.
        h5::write<T>(_parent.disk(), other, h5::count{_counts}, h5::offset{_offsets});

        return *this;
    }

    // TODO: I'm not entirely sure if a TensorView can be sent to the disk.
    template <template <typename, size_t> typename TType>
    auto operator=(TType<T, ViewRank> const &other) -> DiskView & {
        if (_readOnly) {
            EINSUMS_THROW_EXCEPTION(Error::bad_parameter, "Attempting to write data to a read only disk view.");
        }

        // Check dims
        for (int i = 0; i < ViewRank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(Error::bad_parameter, "dims do not match (i {} dim {} other {})", i, _dims[i], other.dim(i));
            }
        }

        // Performing the write here will cause a double write to occur. The destructor above will call put to save
        // the data to disk.
        // Sync the data to disk and into our internal tensor.
        // h5::write<T>(_parent.disk(), other.data(), h5::count{_counts}, h5::offset{_offsets});
        _tensor = other;

        return *this;
    }

    // Does not perform a disk read. That was handled by the constructor.
    auto get() -> Tensor<T, ViewRank> & { return _tensor; }

    void put() {
        if (!_readOnly)
            h5::write<T>(_parent.disk(), _tensor.data(), h5::count{_counts}, h5::offset{_offsets});
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> T const & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) -> T & {
        return _tensor(std::forward<MultiIndex>(index)...);
    }

    size_t        dim(int d) const override { return _tensor.dim(d); }
    Dim<ViewRank> dims() const override { return _tensor.dims(); }

    std::string const &name() const override { return _name; }
    void               set_name(std::string const &new_name) override { _name = new_name; }

    operator Tensor<T, ViewRank> &() const { return _tensor; }       // NOLINT
    operator Tensor<T, ViewRank> const &() const { return _tensor; } // NOLINT

    void zero() { _tensor.zero(); }
    void set_all(T value) { _tensor.set_all(value); }

    bool full_view_of_underlying() const override {
        size_t prod = 1;
        for (int i = 0; i < ViewRank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < Rank; i++) {
            prod2 *= _parent.dim(i);
        }

        return prod == prod2;
    }

  private:
    einsums::DiskTensor<T, Rank> &_parent;
    Dim<ViewRank>                 _dims;
    Count<Rank>                   _counts;
    Offset<Rank>                  _offsets;
    Stride<Rank>                  _strides;
    Tensor<T, ViewRank>           _tensor;
    std::string                   _name{"(unnamed)"};

    bool _readOnly{false};

    // std::unique_ptr<Tensor<ViewRank, T>> _tensor;
};

#ifdef __cpp_deduction_guides
template <typename... Dims>
DiskTensor(h5::fd_t &file, std::string name, Dims... dims) -> DiskTensor<double, sizeof...(Dims)>;

template <typename... Dims>
DiskTensor(h5::fd_t &file, std::string name, Chunk<sizeof...(Dims)> chunk, Dims... dims) -> DiskTensor<double, sizeof...(Dims)>;
#endif

/**
 * @brief Create a disk tensor object.
 *
 * Creates a new tensor that lives on disk. This function does not create any tensor in memory
 * but the tensor is "created" on the HDF5 @p file handle.
 *
 * @code
 * auto a = create_disk_tensor(handle, "a", 3, 3);           // auto -> DiskTensor<double, 2>
 * auto b = create_disk_tensor<float>(handle, "b", 4, 5, 6); // auto -> DiskTensor<float, 3>
 * @endcode
 *
 * @tparam Type The datatype of the underlying disk tensor. Defaults to double.
 * @tparam Args The datatypes of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param file The HDF5 file descriptor
 * @param name The name of the tensor. Stored in the file under this name.
 * @param args The arguments needed to constructor the tensor
 * @return A new disk tensor.
 */
template <typename Type = double, typename... Args>
auto create_disk_tensor(h5::fd_t &file, std::string const &name, Args... args) -> DiskTensor<Type, sizeof...(Args)> {
    return DiskTensor<Type, sizeof...(Args)>{file, name, args...};
}

/**
 * @brief Create a disk tensor object.
 *
 * Creates a new tensor that lives on disk. This function does not create any tensor in memory
 * but the tensor is "created" on the HDF5 @p file handle. Data from the provided tensor is NOT
 * saved.
 *
 * @code
 * auto mem_a = create_tensor("a", 3, 3");           // auto -> Tensor<double, 2>
 * auto a = create_disk_tensor_like(handle, mem_a);  // auto -> DiskTensor<double, 2>
 * @endcode
 *
 * @tparam Type The datatype of the underlying disk tensor.
 * @tparam Rank The datatypes of the calling parameters. In almost all cases you should not need to worry about this parameter.
 * @param file The HDF5 file descriptor
 * @param tensor The tensor to reference for size and name.
 * @return A new disk tensor.
 */
template <typename T, size_t Rank>
auto create_disk_tensor_like(h5::fd_t &file, Tensor<T, Rank> const &tensor) -> DiskTensor<T, Rank> {
    return DiskTensor(file, tensor);
}

} // namespace einsums