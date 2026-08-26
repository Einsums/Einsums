//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/ModuleVars.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/AreAllConvertible.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/Utilities.hpp>

#include <H5Dpublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <cstdio>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <string>

namespace einsums {
/**
 * @struct DiskView
 *
 * @brief Represents a view of a DiskTensor.
 *
 * @tparam T The data type stored by the tensor.
 * @tparam ViewRank The rank of the view.
 * @tparam Rank The rank of the DiskTensor being viewed.
 */
template <typename T, size_t rank>
struct DiskView final : tensor_base::DiskTensor, design_pats::Lockable<std::recursive_mutex> {
    /**
     * @typedef ValueType
     *
     * @brief Holds the type of data stored by this tensor.
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the view.
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef underlying_type
     *
     * @brief Holds the tensor type that this object views. It will be a DiskTensor in this case.
     */
    using underlying_type = einsums::DiskTensor<T, rank>;

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskTensor<T, BaseRank> &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2);

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }

        _size = prod;
    }

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskTensor<T, BaseRank> const &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2);

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }

        set_read_only(true);
        _size = prod;
    }

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskView<T, BaseRank> &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2) && parent.full_view_of_underlying();

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }
        _size = prod;
    }

    /**
     * Construct a view of a tensor with the given dimensions and with the given dataset and dataspace.
     */
    template <size_t BaseRank>
    DiskView(einsums::DiskView<T, BaseRank> const &parent, Dim<rank> const &dims, hid_t dataset, hid_t dataspace)
        : _dims(dims), _dataset(dataset), _dataspace(dataspace) {

        _data_type = H5I_INVALID_HID;

        if constexpr (std::is_same_v<T, float>) {
            _data_type = H5T_NATIVE_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            _data_type = H5T_NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<T, std::complex<float>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().float_complex_type;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            _data_type = detail::Einsums_Tensor_vars::get_singleton().double_complex_type;
        }

        size_t prod = 1;
        for (int i = 0; i < rank; i++) {
            prod *= _dims[i];
        }

        size_t prod2 = 1;
        for (int i = 0; i < rank; i++) {
            prod2 *= parent.dim(i);
        }

        _full_view = (prod == prod2) && parent.full_view_of_underlying();

        _mem_dataspace =
            H5Screate_simple(rank, reinterpret_cast<hsize_t const *>(dims.data()), reinterpret_cast<hsize_t const *>(dims.data()));

        if (_data_type == H5I_INVALID_HID || dataspace == H5I_INVALID_HID || dataset == H5I_INVALID_HID ||
            _mem_dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not initialize disk view!");
        }

        set_read_only(true);

        _size = prod;
    }

    /**
     * Default copy constructor
     */
    DiskView(DiskView const &) = default;

    /**
     * Default move constructor
     */
    DiskView(DiskView &&) noexcept = default;

    /**
     * Destructor.
     */
    ~DiskView() {
        put();

        if (_dataspace != H5I_INVALID_HID) {
            H5Sclose(_dataspace);
            _dataspace = H5I_INVALID_HID;
        }

        if (_mem_dataspace != H5I_INVALID_HID && _mem_dataspace != H5S_ALL) {
            H5Sclose(_mem_dataspace);
            _mem_dataspace = H5S_ALL;
        }
    }

    /**
     * Make the tensor view read only.
     */
    void set_read_only(bool readOnly) { _readOnly = readOnly; }

    /**
     * Copy data from a pointer to the view.
     *
     * @attention This is an expert method only. If you are using this, then you must know what you are doing!
     */
    auto operator=(T const *other) -> DiskView & {
        if (_readOnly) {
            EINSUMS_THROW_EXCEPTION(access_denied, "Attempting to write data to a read only disk view.");
        }

        std::lock_guard lock{*this};

        fetch();

        std::memcpy(_tensor->data(), other, _tensor->size() * sizeof(T));

        return *this;
    }

    /**
     * Copy a tensor into disk.
     */
    template <TensorConcept TType>
        requires SameUnderlyingAndRank<TType, DiskView>
    auto operator=(TType const &other) -> DiskView & {
        if (_readOnly) {
            EINSUMS_THROW_EXCEPTION(access_denied, "Attempting to write data to a read only disk view.");
        }

        std::lock_guard lock{*this};

        // Check dims
        for (int i = 0; i < rank; i++) {
            if (_dims[i] != other.dim(i)) {
                EINSUMS_THROW_EXCEPTION(dimension_error, "dims do not match (i {} dim {} other {})", i, _dims[i], other.dim(i));
            }
        }

        // Performing the write here will cause a double write to occur. The destructor above will call put to save
        // the data to disk.
        // Sync the data to disk and into our internal tensor.
        // h5::write<T>(_parent.disk(), other.data(), h5::count{_counts}, h5::offset{_offsets});
        get() = other;

        return *this;
    }

    /**
     * If the buffered tensor has not been created, create it and fill it. If it has already been created,
     * do nothing.
     *
     * @sa fetch
     *
     * @versionadded 2.0.0
     */
    void cache() const {
        auto lock = std::lock_guard(*this);
        if (!_tensor) {
            _tensor = std::make_unique<BufferTensor<T, Rank>>(true, _dims);

            auto err = H5Dread(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor->data());

            if (err < 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
            }
        }
    }

    /**
     * If the buffered tensor has not been created, create it and fill it. If it has already been created,
     * update its contents with the data stored on disk.
     *
     * @sa cache
     *
     * @versionadded 2.0.0
     */
    void fetch() const {
        auto lock = std::lock_guard(*this);
        if (!_tensor) {
            _tensor = std::make_unique<BufferTensor<T, Rank>>(true, _dims);
        }

        auto err = H5Dread(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor->data());

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not read tensor data!");
        }
    }

    /**
     * Gets the underlying tensor holding the data.
     */
    [[nodiscard]] auto get() -> BufferTensor<T, rank> & {
        cache();
        return *_tensor;
    }

    /**
     * Gets the underlying tensor holding the data.
     */
    [[nodiscard]] auto get() const -> BufferTensor<T, rank> const & {
        cache();
        return *_tensor;
    }

    /**
     * Gets the underlying tensor holding the data. If the tensor has already been created,
     * update it with what is stored on disk.
     */
    [[nodiscard]] auto get_update() -> BufferTensor<T, rank> & {
        fetch();
        return *_tensor;
    }

    /**
     * Gets the underlying tensor holding the data. If the tensor has already been created,
     * update it with what is stored on disk.
     */
    [[nodiscard]] auto get_update() const -> BufferTensor<T, rank> const & {
        fetch();
        return *_tensor;
    }

    /**
     * Writes the buffered tensor's data to the disk then destroys the buffered tensor.
     * There is no const version.
     */
    void unget() {
        auto lock = std::lock_guard(*this);
        if (_tensor) {
            put();

            _tensor.reset();
        }
    }

    /**
     * Push any changes to the view to the disk.
     */
    void put() {
        auto lock = std::lock_guard(*this);
        if (!_readOnly && _tensor) {
            _tensor->tensor_from_gpu();
            H5Dwrite(_dataset, _data_type, _mem_dataspace, _dataspace, H5P_DEFAULT, _tensor->data());
        }
    }

    /**
     * Subscript into the tensor.
     */
    template <typename... MultiIndex>
        requires(NoneOfType<AllT, MultiIndex...> && NoneOfType<Range, MultiIndex...>)
    auto operator()(MultiIndex &&...index) const -> T const & {
        return get()(std::forward<MultiIndex>(index)...);
    }

    /**
     * Subscript into the tensor.
     */
    template <typename... MultiIndex>
        requires(NoneOfType<AllT, MultiIndex...> && NoneOfType<Range, MultiIndex...>)
    auto operator()(MultiIndex &&...index) -> T & {
        return get()(std::forward<MultiIndex>(index)...);
    }

    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index) -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
        // Get positions of All
        auto all_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<AllT, MultiIndex...>()>>(
            arguments::positions_of_type<AllT, MultiIndex...>());
        auto index_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(
            arguments::positions_of_type<size_t, MultiIndex...>());
        auto range_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(
            arguments::positions_of_type<Range, MultiIndex...>());

        auto const &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Count<Rank>  counts{};
        Dim<Rank>    block{};

        std::fill(counts.begin(), counts.end(), 1);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
            block[value]   = 1;
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            block[value] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            block[value]   = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (int i = 0; i < Rank; i++) {
            if (!is_in(i, index_positions)) {
                dims_all[dims_index] = block[i];
                dims_index++;
            }
        }

        hid_t dataspace = H5Scopy(_dataspace);

        if (dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create a copy of the data space for view creation!");
        }

        int parent_rank = H5Sget_simple_extent_ndims(_dataspace);

        if (parent_rank < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        std::vector<hsize_t> start_vec(parent_rank), count_vec(parent_rank), block_vec(parent_rank);

        auto err = H5Sget_regular_hyperslab(dataspace, start_vec.data(), NULL, count_vec.data(), block_vec.data());

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        for (int i = 0, j = 0; i < parent_rank && j < rank; i++) {
            if (block_vec[i] != 1) {
                start_vec[i] += offsets[j];
                j++;
            }

            while (j < rank && block[j] == 1) {
                j++;
            }
        }

        err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, reinterpret_cast<hsize_t const *>(offsets.data()), NULL,
                                  reinterpret_cast<hsize_t const *>(counts.data()), reinterpret_cast<hsize_t const *>(block.data()));

        if (err < 0) {
            H5Sclose(dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Disk view creation failed!");
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(*this, dims_all, _dataset,
                                                                                                         dataspace);
    }

    /// This creates a Disk object with its Rank being equal to the number of All{} parameters
    /// Range is not inclusive. Range{10, 11} === size of 1
    template <typename... MultiIndex>
        requires(count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>() != 0)
    auto operator()(MultiIndex... index) const
        -> DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> const {
        // Get positions of All
        auto all_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<AllT, MultiIndex...>()>>(
            arguments::positions_of_type<AllT, MultiIndex...>());
        auto index_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<size_t, MultiIndex...>()>>(
            arguments::positions_of_type<size_t, MultiIndex...>());
        auto range_positions = arguments::get_array_from_tuple<std::array<int, count_of_type<Range, MultiIndex...>()>>(
            arguments::positions_of_type<Range, MultiIndex...>());

        auto const &indices = std::forward_as_tuple(index...);

        // Need the offset and stride into the large tensor
        Offset<Rank> offsets{};
        Count<Rank>  counts{};
        Dim<Rank>    block{};

        std::fill(counts.begin(), counts.end(), 1);

        // Need the dim of the smaller tensor
        Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> dims_all{};

        for (auto [i, value] : enumerate(index_positions)) {
            // printf("i, value: %d %d\n", i, value);
            offsets[value] = arguments::get_from_tuple<size_t>(indices, value);
            block[value]   = 1;
        }
        for (auto [i, value] : enumerate(all_positions)) {
            // println("here");
            block[value] = _dims[value];
        }
        for (auto [i, value] : enumerate(range_positions)) {
            offsets[value] = arguments::get_from_tuple<Range>(indices, value)[0];
            block[value]   = arguments::get_from_tuple<Range>(indices, value)[1] - arguments::get_from_tuple<Range>(indices, value)[0];
        }

        // Go through counts and anything that isn't equal to 1 is copied to the dims_all
        int dims_index = 0;
        for (int i = 0; i < Rank; i++) {
            if (!is_in(i, index_positions)) {
                dims_all[dims_index] = block[i];
                dims_index++;
            }
        }

        hid_t dataspace = H5Scopy(_dataspace);

        if (dataspace == H5I_INVALID_HID) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not create a copy of the data space for view creation!");
        }

        int parent_rank = H5Sget_simple_extent_ndims(_dataspace);

        if (parent_rank < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        std::vector<hsize_t> start_vec(parent_rank), count_vec(parent_rank), block_vec(parent_rank);

        auto err = H5Sget_regular_hyperslab(dataspace, start_vec.data(), NULL, count_vec.data(), block_vec.data());

        if (err < 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Could not query properties of the dataspace!");
        }

        for (int i = 0, j = 0; i < parent_rank && j < rank; i++) {
            if (block_vec[i] != 1) {
                start_vec[i] += offsets[j];
                j++;
            }

            while (j < rank && block[j] == 1) {
                j++;
            }
        }

        err = H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, reinterpret_cast<hsize_t const *>(offsets.data()), NULL,
                                  reinterpret_cast<hsize_t const *>(counts.data()), reinterpret_cast<hsize_t const *>(block.data()));

        if (err < 0) {
            H5Sclose(dataspace);
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "Disk view creation failed!");
        }

        return DiskView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>(*this, dims_all, _dataset,
                                                                                                         dataspace);
    }

    /**
     * Get the dimension along a given axis.
     */
    [[nodiscard]] size_t dim(int d) const {
        if (d < 0) {
            d += rank;
        }
        return _dims.at(d);
    }

    /**
     * Get all the dimensions of the view.
     */
    [[nodiscard]] Dim<rank> dims() const { return _dims; }

    /**
     * Get the size of the view.
     */
    [[nodiscard]] size_t size() const { return _size; }

    /**
     * Get the name of the tensor.
     */
    [[nodiscard]] std::string const &name() const { return _name; }

    /**
     * Set the name of the tensor.
     */
    void set_name(std::string const &new_name) { _name = new_name; }

    /**
     * Cast the tensor to Tensor<T,ViewRank>.
     */
    operator BufferTensor<T, rank> &() { return get(); } // NOLINT

    /**
     * Cast the tensor to Tensor<T,ViewRank>.
     */
    operator BufferTensor<T, rank> const &() const { return get(); } // NOLINT

    /**
     * Set all of the values of the tensor to zero.
     */
    void zero() {
        if (!_tensor) {
            fetch();
        }

        _tensor->zero();
    }

    /**
     * Set all of the values of the tensor to the passed value.
     */
    void set_all(T value) {
        if (!_tensor) {
            fetch();
        }
        _tensor->set_all(value);
    }

    /**
     * Gets whether the view is showing the whole tensor.
     */
    [[nodiscard]] bool full_view_of_underlying() const { return _full_view; }

  private:
    /**
     * @var _dataspace
     *
     * The dataspace that contains the parameters for the view.
     */
    hid_t _dataspace{H5I_INVALID_HID};

    /**
     * @var _mem_dataspace
     *
     * The dataspace that specifies the parameters for the core tensor.
     */
    hid_t _mem_dataspace{H5S_ALL};

    /**
     * @var _dataset
     *
     * The data set that this tensor is viewing.
     */
    hid_t _dataset{H5I_INVALID_HID};

    /**
     * @var _data_type
     *
     * The data type identifier.
     */
    hid_t _data_type{H5I_INVALID_HID};

    /**
     * @var _dims
     *
     * Holds the dimensions of the tensor.
     */
    Dim<rank> _dims;

    /**
     * @var _size
     *
     * The size of the tensor view.
     */
    size_t _size;

    /**
     * @var _tensor
     *
     * This is the in-core representation of the view.
     */
    mutable std::unique_ptr<BufferTensor<T, rank>> _tensor{nullptr};

    /**
     * @var _name
     *
     * This is the name of the view used for printing.
     */
    std::string _name{"(unnamed)"};

    /**
     * @var _readOnly
     *
     * Indicates whether the view is read-only or read-write.
     */
    bool _readOnly{false};

    /**
     * @var _full_view
     *
     * Indicates whether this view sees the entire data space that it was built on.
     */
    bool _full_view{false};
};

TENSOR_EXPORT(DiskView)

} // namespace einsums
