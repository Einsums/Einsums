//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#ifndef BACKENDS_DEVICE_TENSOR_HPP
#define BACKENDS_DEVICE_TENSOR_HPP

// If this is included on its own, we should not include DeviceTensorView.hpp here.
// It depends on functions in this file, and tests break if it is included first.
#include <Einsums/Assert.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Iterator/Enumerate.hpp>
#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TypeSupport/GPUCast.hpp>
#include <Einsums/TypeSupport/GPUComplex.hpp>

#include <cstring>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace einsums {

#ifndef DOXYGEN
template <typename T, size_t rank>
DeviceTensor<T, rank>::DeviceTensor(DeviceTensor<T, rank> const &copy, detail::HostToDeviceMode mode) {
    *this = copy;
}

template <typename T, size_t rank>
DeviceTensor<T, rank>::~DeviceTensor() {
    using namespace einsums::gpu;

    gpu::device_synchronize();

    switch (_mode) {
    case einsums::detail::MAPPED:
        if (this->_host_data == nullptr) {
            return;
        }
        hip_catch(hipHostUnregister((void *)this->_host_data));
        delete[] this->_host_data;
        break;
    case einsums::detail::PINNED:
        hip_catch(hipHostFree((void *)this->_host_data));
        break;
    case einsums::detail::DEV_ONLY:
        hip_catch(hipFree((void *)this->_data));
        break;
    default:
        break;
    }

    if (this->_gpu_dims != nullptr) {
        hip_catch(hipFree((void *)this->_gpu_dims));
    }
}

template <typename T, size_t rank>
template <typename... Dims>
    requires requires {
        requires(sizeof...(Dims) == rank);
        requires(!std::is_same_v<detail::HostToDeviceMode, Dims> && ...);
    }
DeviceTensor<T, rank>::DeviceTensor(std::string name, einsums::detail::HostToDeviceMode mode, Dims... dims)
    : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...}, _mode{mode} {
    using namespace einsums::gpu;
    static_assert(rank == sizeof...(dims), "Declared rank does not match provided dims");

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
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    switch (mode) {
    case einsums::detail::MAPPED:
        this->_host_data = new T[size];
        hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::PINNED:
        hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::DEV_ONLY:
        this->_host_data = nullptr;
        hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
        break;
    default:
        EINSUMS_THROW_EXCEPTION(enum_error, "Did not understand the provided mode!");
    }

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 2 * sizeof(size_t) * rank));
    this->_gpu_strides = this->_gpu_dims + rank;

    EINSUMS_ASSERT(this->_gpu_dims != nullptr && this->_dims.data() != nullptr);

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));

    gpu::device_synchronize();
}

template <typename T, size_t rank>
template <typename... Dims>
    requires requires {
        requires(sizeof...(Dims) == rank);
        requires(!std::is_same_v<detail::HostToDeviceMode, Dims> && ...);
    }
DeviceTensor<T, rank>::DeviceTensor(std::string name, Dims... dims)
    : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...}, _mode{detail::DEV_ONLY} {
    using namespace einsums::gpu;
    static_assert(rank == sizeof...(dims), "Declared rank does not match provided dims");

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
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    switch (_mode) {
    case einsums::detail::MAPPED:
        this->_host_data = new T[size];
        hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::PINNED:
        hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::DEV_ONLY:
        this->_host_data = nullptr;
        hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
        break;
    default:
        EINSUMS_THROW_EXCEPTION(enum_error, "Did not understand the provided mode!");
    }

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 2 * sizeof(size_t) * rank));
    this->_gpu_strides = this->_gpu_dims + rank;

    EINSUMS_ASSERT(this->_gpu_dims != nullptr && this->_dims.data() != nullptr);

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));

    gpu::device_synchronize();
}

template <typename T, size_t rank>
template <size_t OtherRank, typename... Dims>
DeviceTensor<T, rank>::DeviceTensor(DeviceTensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims)
    : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
    using namespace einsums::gpu;
    if (existingTensor._mode == einsums::detail::UNKNOWN) {
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "DeviceTensor being copied has not been initialized!");
    }
    this->_host_data = existingTensor._host_data;
    this->_data      = existingTensor._data;
    this->_mode      = existingTensor._mode;

    existingTensor._host_data = nullptr;
    existingTensor._data      = nullptr;
    if constexpr (OtherRank != 0) {
        existingTensor._gpu_dims    = nullptr;
        existingTensor._gpu_strides = nullptr;
    }

    static_assert(rank == sizeof...(dims), "Declared rank does not match provided dims");

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
    for (auto [i, dim] : einsums::enumerate(_dims)) {
        if (dim == -1) {
            nfound++;
            location = i;
        }
    }

    if (nfound > 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "More than one -1 was provided.");
    }

    if (nfound == 1) {
        size_t size{1};
        for (auto [i, dim] : einsums::enumerate(_dims)) {
            if (i != location)
                size *= dim;
        }
        if (size > existingTensor.size()) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Size of new tensor is larger than the parent tensor.");
        }
        _dims[location] = existingTensor.size() / size;
    }

    // Row-major order of dimensions
    std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    // Check size
    if (existingTensor.size() != size) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Provided dims to not match size of parent tensor");
    }

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 2 * sizeof(size_t) * rank));
    this->_gpu_strides = this->_gpu_dims + rank;

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    gpu::device_synchronize();
}

template <typename T, size_t rank>
DeviceTensor<T, rank>::DeviceTensor(Dim<rank> dims, einsums::detail::HostToDeviceMode mode) : _dims{std::move(dims)}, _mode{mode} {
    using namespace einsums::gpu;
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
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    switch (mode) {
    case einsums::detail::MAPPED:
        this->_host_data = new T[size];
        hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::PINNED:
        hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::DEV_ONLY:
        this->_host_data = nullptr;
        hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
        break;
    default:
        EINSUMS_THROW_EXCEPTION(enum_error, "Could not understand the provided mode!");
    }

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 2 * sizeof(size_t) * rank));
    this->_gpu_strides = this->_gpu_dims + rank;

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    gpu::device_synchronize();
}
#endif

namespace detail {

/**
 * Kernel to copy a DeviceTensorView into a DeviceTensor object.
 */
template <typename T, size_t rank>
__global__ void copy_to_tensor(T *to_data, size_t const *index_strides, size_t const *to_strides, T const *from_data,
                               size_t const *from_strides, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t inds[rank];

    for (size_t curr_element = worker; curr_element < elements; curr_element += kernel_size) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the view.
        size_t from_ind = einsums::indices_to_sentinel<rank>(from_strides, (size_t *)inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, (size_t *)inds);

        // Do the copy.
        to_data[to_ind] = from_data[from_ind];
    }

    __threadfence();
}

/**
 * Kernel to copy a DeviceTensorView into a DeviceTensor object. Converts as well.
 */
template <typename T, size_t rank, typename TOther>
__global__ void copy_to_tensor_conv(T *to_data, size_t const *index_strides, size_t const *to_strides, TOther const *from_data,
                                    size_t const *from_strides, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    for (size_t curr_element = worker; curr_element < elements; curr_element += kernel_size) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the view.
        size_t from_ind = einsums::indices_to_sentinel<rank>(inds, (size_t const *)from_strides);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(inds, (size_t const *)to_strides);

        // Do the copy.
        to_data[to_ind] = (T)from_data[from_ind];
    }

    __threadfence();
}

} // namespace detail

#ifndef DOXYGEN

template <typename T, size_t rank>
DeviceTensor<T, rank>::DeviceTensor(DeviceTensorView<T, rank> const &other) : _name{other.name()}, _dims{other.dims()} {
    using namespace einsums::gpu;
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
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    this->_mode      = einsums::detail::DEV_ONLY;
    this->_host_data = nullptr;
    hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 2 * sizeof(size_t) * rank));
    this->_gpu_strides = this->_gpu_dims + rank;

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));

    gpu::device_synchronize();

    einsums::detail::copy_to_tensor<dev_datatype, rank><<<blocks(size), block_size(size), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, other.gpu_data(), other.gpu_strides(), size);

    gpu::stream_wait();
}

#endif

namespace detail {

// Set every entry in a tensor to an element.
template <typename T, size_t rank>
__global__ void set_all(T *data, size_t const *index_strides, size_t const *strides, T value, size_t elements) {
    using namespace einsums::gpu;
    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    // Wrap around to help save work load.
    size_t block_size = blockDim.x * blockDim.y * blockDim.z;

    size_t elements_adjusted;

    if (elements % block_size == 0) {
        elements_adjusted = elements;
    } else {
        elements_adjusted = elements + (block_size - (elements % block_size));
    }

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the tensor.
        size_t ind = einsums::indices_to_sentinel<rank>(strides, inds);

        // Do the copy.
        data[ind] = value;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

} // namespace detail

#ifndef DOXYGEN
template <typename T, size_t rank>
void DeviceTensor<T, rank>::resize(Dim<rank> dims) {
    using namespace einsums::gpu;

    if (dims == _dims) {
        return;
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

    size_t old_size = size();

    _dims = dims;

    // Row-major order of dimensions
    std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    if (size == old_size) {
        return;
    }

    switch (_mode) {
    case einsums::detail::MAPPED:
        hip_catch(hipHostUnregister((void *)this->_host_data));

        if (this->_host_data != nullptr) {
            delete[] this->_host_data;
        }
        this->_host_data = new T[size];

        hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));

        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::PINNED:
        if (this->_host_data != nullptr) {
            hip_catch(hipHostFree((void *)this->_host_data));
        }

        hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));

        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::DEV_ONLY:
        if (this->_data != nullptr) {
            hip_catch(hipFree((void *)this->_data));
        }

        hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
        break;
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor has not been initialized!");
    }

    if (this->_gpu_dims == nullptr) {
        hip_catch(hipMalloc((void **)&this->_gpu_dims, 2 * sizeof(size_t) * rank));
        _gpu_strides = _gpu_dims + rank;
    }

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    gpu::device_synchronize();
}

template <typename T, size_t rank>
void DeviceTensor<T, rank>::zero() {
    using namespace einsums::gpu;
    einsums::detail::set_all<dev_datatype, rank><<<blocks(this->size()), block_size(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, einsums::HipCast<dev_datatype, double>::cast(0.0), _dims[0] * _strides[0]);
    stream_wait();
}

template <typename T, size_t rank>
void DeviceTensor<T, rank>::set_all(T value) {
    using namespace einsums::gpu;
    einsums::detail::set_all<dev_datatype, rank><<<blocks(this->size()), block_size(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, HipCast<dev_datatype, T>::cast(value), _dims[0] * _strides[0]);
    stream_wait();
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires requires {
        requires NoneOfType<AllT, MultiIndex...>;
        requires NoneOfType<Range, MultiIndex...>;
    }
DeviceTensor<T, rank>::dev_datatype *DeviceTensor<T, rank>::gpu_data(MultiIndex... index) {
    using namespace einsums::gpu;
#    if !defined(DOXYGEN)
    EINSUMS_ASSERT(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = std::array{static_cast<std::int64_t>(index)...};
    for (auto [i, _index] : einsums::enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
#    endif
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires requires {
        requires NoneOfType<AllT, MultiIndex...>;
        requires NoneOfType<Range, MultiIndex...>;
    }
DeviceTensor<T, rank>::dev_datatype const *DeviceTensor<T, rank>::gpu_data(MultiIndex... index) const {
    using namespace einsums::gpu;
#    if !defined(DOXYGEN)
    EINSUMS_ASSERT(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = std::array{static_cast<std::int64_t>(index)...};
    for (auto [i, _index] : einsums::enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
#    endif
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires requires {
        requires NoneOfType<AllT, MultiIndex...>;
        requires NoneOfType<Range, MultiIndex...>;
    }
DeviceTensor<T, rank>::host_datatype *DeviceTensor<T, rank>::data(MultiIndex... index) {
    using namespace einsums::gpu;
#    if !defined(DOXYGEN)
    EINSUMS_ASSERT(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = std::array{static_cast<std::int64_t>(index)...};
    for (auto [i, _index] : einsums::enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _host_data + ordinal;
#    endif
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires requires {
        requires NoneOfType<AllT, MultiIndex...>;
        requires NoneOfType<Range, MultiIndex...>;
    }
DeviceTensor<T, rank>::host_datatype const *DeviceTensor<T, rank>::data(MultiIndex... index) const {
    using namespace einsums::gpu;
#    if !defined(DOXYGEN)
    EINSUMS_ASSERT(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = std::array{static_cast<std::int64_t>(index)...};
    for (auto [i, _index] : einsums::enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _host_data + ordinal;
#    endif
}

template <typename T, size_t rank>
__host__ void DeviceTensor<T, rank>::read(std::vector<T> const &data) {
    using namespace einsums::gpu;
    EINSUMS_ASSERT(data.size() <= this->size());

    switch (_mode) {
    case einsums::detail::MAPPED:
    case einsums::detail::PINNED:
        std::memcpy((void *)this->_host_data, (void const *)data.data(), data.size() * sizeof(T));
        break;
    case einsums::detail::DEV_ONLY:
        hip_catch(hipMemcpy((void *)this->_data, (void const *)data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
        gpu::device_synchronize();
        break;
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
    }
}

template <typename T, size_t rank>
void DeviceTensor<T, rank>::write(std::vector<T> &data) {
    using namespace einsums::gpu;
    if (data.size() != this->size()) {
        data.resize(this->size());
    }

    switch (_mode) {
    case einsums::detail::MAPPED:
    case einsums::detail::PINNED:
        std::memcpy((void *)data.data(), (void const *)this->_host_data, this->size() * sizeof(T));
        break;
    case einsums::detail::DEV_ONLY:
        hip_catch(hipMemcpy((void *)data.data(), (void const *)this->_data, this->size() * sizeof(T), hipMemcpyDeviceToHost));
        // No sync
        break;
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
    }
}

template <typename T, size_t rank>
__host__ void DeviceTensor<T, rank>::read(T const *data) {
    using namespace einsums::gpu;

    switch (_mode) {
    case einsums::detail::MAPPED:
    case einsums::detail::PINNED:
        std::memcpy((void *)this->_host_data, (void const *)data, this->size() * sizeof(T));
        break;
    case einsums::detail::DEV_ONLY:
        hip_catch(hipMemcpy((void *)this->_data, (void const *)data, this->size() * sizeof(T), hipMemcpyHostToDevice));
        gpu::device_synchronize();
        break;
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
    }
}

template <typename T, size_t rank>
void DeviceTensor<T, rank>::write(T *data) {
    using namespace einsums::gpu;

    switch (_mode) {
    case einsums::detail::MAPPED:
    case einsums::detail::PINNED:
        std::memcpy((void *)data, (void const *)this->_host_data, this->size() * sizeof(T));
        break;
    case einsums::detail::DEV_ONLY:
        hip_catch(hipMemcpy((void *)data, (void const *)this->_data, this->size() * sizeof(T), hipMemcpyDeviceToHost));
        // No sync
        break;
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
    }
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
auto DeviceTensor<T, rank>::operator()(MultiIndex &&...index) const -> T {
    using namespace einsums::gpu;

    static_assert(sizeof...(MultiIndex) == rank);

    T      out;
    size_t ordinal = einsums::indices_to_sentinel(_strides, std::forward<MultiIndex>(index)...);

    switch (_mode) {
    case einsums::detail::MAPPED:
    case einsums::detail::PINNED:
        EINSUMS_ASSERT(sizeof...(MultiIndex) <= _dims.size());

        return this->_host_data[ordinal];
    case einsums::detail::DEV_ONLY:
        hip_catch(hipMemcpy((void *)&out, (void const *)(this->_data + ordinal), sizeof(T), hipMemcpyDeviceToHost));
        // no sync
        return out;
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
    }
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
auto DeviceTensor<T, rank>::operator()(MultiIndex &&...index) -> HostDevReference<T> {
    using namespace einsums::gpu;
    static_assert(sizeof...(MultiIndex) == rank);
    constexpr bool test = requires { requires !(std::is_same_v<einsums::AllT, std::remove_cvref_t<MultiIndex>> || ...); };
    static_assert(test);

    size_t ordinal = einsums::indices_to_sentinel(_strides, std::forward<MultiIndex>(index)...);

    if (ordinal > this->size()) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "Array index out of range!");
    }

    switch (_mode) {
    case einsums::detail::MAPPED:
    case einsums::detail::PINNED:
        return HostDevReference<T>(this->_host_data + ordinal, true);
    case einsums::detail::DEV_ONLY:
        return HostDevReference<host_datatype>((host_datatype *)_data + ordinal, false);
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
    }
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires AtLeastOneOfType<AllT, MultiIndex...>
auto DeviceTensor<T, rank>::operator()(MultiIndex... index)
    -> DeviceTensorView<T, count_of_type<einsums::AllT, MultiIndex...>() + count_of_type<einsums::Range, MultiIndex...>()> {
    using namespace einsums::gpu;
    // Construct a TensorView using the indices provided as the starting point for the view.
    // e.g.:
    //    Tensor T{"Big Tensor", 7, 7, 7, 7};
    //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
    // println("Here");
    auto const &indices = std::forward_as_tuple(index...);

    Offset<rank>                                                                         offsets;
    Stride<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> strides{};
    Dim<count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>    dims{};

    int counter{0};
    for_sequence<sizeof...(MultiIndex)>([&](auto i) {
        // println("looking at {}", i);
        if constexpr (std::is_convertible_v<std::tuple_element_t<i, std::tuple<MultiIndex...>>, std::int64_t>) {
            auto tmp = static_cast<std::int64_t>(std::get<i>(indices));
            if (tmp < 0)
                tmp = _dims[i] + tmp;
            offsets[i] = tmp;
        } else if constexpr (std::is_same_v<AllT, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
            strides[counter] = _strides[i];
            dims[counter]    = _dims[i];
            offsets[i]       = 0;
            counter++;

        } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
            auto range = std::get<i>(indices);
            offsets[i] = range[0];
            if (range[1] < 0) {
                auto temp = _dims[i] + range[1];
                range[1]  = temp;
            }
            dims[counter]    = range[1] - range[0];
            strides[counter] = _strides[i];
            counter++;
        }
    });

    return DeviceTensorView<T, count_of_type<einsums::AllT, MultiIndex...>() + count_of_type<einsums::Range, MultiIndex...>()>{
        *this, std::move(dims), offsets, strides};
}

template <typename T, size_t rank>
template <typename... MultiIndex>
    requires NumOfType<einsums::Range, rank, MultiIndex...>
auto DeviceTensor<T, rank>::operator()(MultiIndex... index) const -> DeviceTensorView<T, rank> {
    using namespace einsums::gpu;
    Dim<rank>    dims{};
    Offset<rank> offset{};
    Stride<rank> stride = _strides;

    auto ranges = arguments::get_array_from_tuple<std::array<Range, rank>>(std::forward_as_tuple(index...));

    for (int r = 0; r < rank; r++) {
        auto range = ranges[r];
        offset[r]  = range[0];
        if (range[1] < 0) {
            auto temp = _dims[r] + range[1];
            range[1]  = temp;
        }
        dims[r] = range[1] - range[0];
    }

    return DeviceTensorView<T, rank>{*this, std::move(dims), std::move(offset), std::move(stride)};
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::assign(DeviceTensor<T, rank> const &other) {
    using namespace einsums::gpu;
    bool realloc{_data == nullptr};
    for (int i = 0; i < rank; i++) {
        if (dim(i) == 0 || (dim(i) != other.dim(i)))
            realloc = true;
    }

    if (realloc) {
        gpu::device_synchronize();
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

        switch (_mode) {
        case einsums::detail::MAPPED:
            hip_catch(hipHostUnregister((void *)this->_host_data));

            delete[] this->_host_data;
            this->_host_data = new T[size];

            hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));

            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case einsums::detail::PINNED:
            hip_catch(hipHostFree((void *)this->_host_data));

            hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));

            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case einsums::detail::DEV_ONLY:
            hip_catch(hipFree((void *)this->_data));

            hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
            break;
        default:
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor being assigned to is not initialized!");
        }

        if (this->_gpu_dims == nullptr) {
            hip_catch(hipMalloc((void **)&this->_gpu_dims, 2 * rank * sizeof(size_t)));
            if (this->_gpu_strides != nullptr) {
                hip_catch(hipFree((void *)this->_gpu_strides));
            }
            this->_gpu_strides = this->_gpu_dims + rank;
        }

        hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));

        gpu::device_synchronize();

        einsums::detail::copy_to_tensor<dev_datatype, rank><<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
            this->_data, this->_gpu_strides, this->_gpu_strides, other._data, other._gpu_strides, size);
    } else {
        einsums::detail::copy_to_tensor<dev_datatype, rank><<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
            this->_data, this->_gpu_strides, this->_gpu_strides, other._data, other._gpu_strides, _strides[0] * _dims[0]);
    }

    gpu::stream_wait();

    return *this;
}

template <typename T, size_t rank>
template <typename TOther>
    requires(!std::same_as<T, TOther>)
auto DeviceTensor<T, rank>::assign(DeviceTensor<TOther, rank> const &other) -> DeviceTensor<T, rank> & {
    using namespace einsums::gpu;
    bool realloc{false};
    for (int i = 0; i < rank; i++) {
        if (dim(i) == 0 || (dim(i) != other.dim(i)))
            realloc = true;
    }

    if (realloc) {
        gpu::device_synchronize();
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

        switch (_mode) {
        case einsums::detail::MAPPED:
            hip_catch(hipHostUnregister((void *)this->_host_data));

            delete[] this->_host_data;
            this->_host_data = new T[size];

            hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));

            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case einsums::detail::PINNED:
            hip_catch(hipHostFree((void *)this->_host_data));

            hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));

            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case einsums::detail::DEV_ONLY:
            hip_catch(hipFree((void *)this->_data));

            hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
            break;
        default:
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor being assigned to was not initialized!");
        }

        if (this->_gpu_dims == nullptr) {
            hip_catch(hipMalloc((void **)&this->_gpu_dims, 2 * rank * sizeof(size_t)));
            if (this->_gpu_strides != nullptr) {
                hip_catch(hipFree((void *)this->_gpu_strides));
            }
            this->_gpu_strides = this->_gpu_dims + rank;
        }

        hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
        gpu::device_synchronize();

        einsums::detail::copy_to_tensor_conv<typename DeviceTensor<T, rank>::dev_datatype, rank,
                                             typename DeviceTensor<TOther, rank>::dev_datatype>
            <<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(this->_data, this->_gpu_strides, this->_gpu_strides,
                                                                                  other._data, other._gpu_strides, size);
    } else {
        einsums::detail::copy_to_tensor_conv<typename DeviceTensor<T, rank>::dev_datatype, rank,
                                             typename DeviceTensor<TOther, rank>::dev_datatype>
            <<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(this->_data, this->_gpu_strides, this->_gpu_strides,
                                                                                  other._data, other._gpu_strides, _strides[0] * _dims[0]);
    }

    gpu::stream_wait();

    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::assign(Tensor<T, rank> const &other) {
    using namespace einsums::gpu;
    bool realloc{false};
    println(other);
    for (int i = 0; i < rank; i++) {
        if (dim(i) == 0 || (dim(i) != other.dim(i)))
            realloc = true;
    }

    if (realloc) {
        gpu::device_synchronize();
        struct Stride {
            size_t value{1};
            Stride() = default;
            auto operator()(size_t dim) -> size_t {
                auto old_value = value;
                value *= dim;
                return old_value;
            }
        };

        _dims = other.dims();

        // Row-major order of dimensions
        std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

        switch (_mode) {
        case einsums::detail::MAPPED:
            hip_catch(hipHostUnregister((void *)this->_host_data));

            delete[] this->_host_data;
            this->_host_data = new T[size];

            hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));

            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case einsums::detail::PINNED:
            hip_catch(hipHostFree((void *)this->_host_data));

            hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));

            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case einsums::detail::DEV_ONLY:
            hip_catch(hipFree((void *)this->_data));

            hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
            break;
        default:
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor being assinged to was not initialized!");
        }

        hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
        gpu::device_synchronize();
    }

    if (other.is_row_major()) {
        if (this->_mode == einsums::detail::DEV_ONLY) {
            hip_catch(hipMemcpy((void *)this->_data, (void const *)other.data(), this->size() * sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        } else if (_mode == einsums::detail::MAPPED || _mode == einsums::detail::PINNED) {
            std::memcpy(this->_host_data, other.data(), this->size() * sizeof(T));
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor being assigned to was not initialized!");
        }
    } else {
        if (_mode == einsums::detail::DEV_ONLY) {
            auto lock = other.gpu_cache_tensor();
            dev_datatype const*other_ptr = (dev_datatype const*)other.get_gpu_pointer();

            size_t *other_strides;

            hip_catch(hipMalloc((void **)&other_strides, rank * sizeof(size_t)));
            {
                auto other_strides_temp = other.strides();

                hip_catch(
                    hipMemcpy((void *)other_strides, (void const *)other_strides_temp.data(), rank * sizeof(size_t), hipMemcpyHostToDevice));
            }

            einsums::detail::copy_to_tensor<dev_datatype, rank><<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
                this->_data, this->_gpu_strides, this->_gpu_strides, other_ptr, other_strides, other.size());

            gpu::device_synchronize();

            hip_catch(hipFree((void *)other_strides));
        } else if (_mode == einsums::detail::MAPPED || _mode == einsums::detail::PINNED) {
            auto other_strides = other.strides();
            for(size_t index = 0; index < other.size(); index++) {
                size_t this_index, other_index;
                sentinel_to_sentinels(index, _strides, _strides, this_index, other_strides, other_index);
                _host_data[this_index] = other.data()[other_index];
            }
        }
    }

    return *this;
}

template <typename T, size_t rank>
template <typename TOther>
auto DeviceTensor<T, rank>::assign(DeviceTensorView<TOther, rank> const &other) -> DeviceTensor<T, rank> & {
    using namespace einsums::gpu;
    einsums::detail::copy_to_tensor_conv<typename DeviceTensor<T, rank>::dev_datatype, rank,
                                         typename DeviceTensor<TOther, rank>::dev_datatype>
        <<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
            this->_data, this->_gpu_strides, this->_gpu_strides, other.gpu_data(), other.gpu_strides(), _strides[0] * _dims[0]);

    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::init(DeviceTensor<T, rank> const &copy, einsums::detail::HostToDeviceMode mode) {
    using namespace einsums::gpu;

    gpu::device_synchronize();

    if (_mode != einsums::detail::UNKNOWN) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "Double initialization of device tensor!");
    }

    this->_name    = copy._name;
    this->_dims    = copy._dims;
    this->_strides = copy._strides;
    this->_mode    = mode;

    if (mode == einsums::detail::UNKNOWN) {
        _mode = copy._mode;
    }

    size_t size = copy.dim(0) * copy.stride(0);

    switch (_mode) {
    case einsums::detail::MAPPED:
        this->_host_data = new T[size];
        hip_catch(hipHostRegister((void *)this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::PINNED:
        hip_catch(hipHostMalloc((void **)&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
        break;
    case einsums::detail::DEV_ONLY:
        this->_host_data = nullptr;
        hip_catch(hipMalloc((void **)&(this->_data), size * sizeof(T)));
        break;
    default:
        EINSUMS_THROW_EXCEPTION(enum_error, "Did not understand the provided mode!");
    }

    switch (copy._mode) {
    case einsums::detail::DEV_ONLY:
        hip_catch(hipMemcpy((void *)this->_data, (void const *)copy._data, copy.size() * sizeof(T), hipMemcpyDeviceToDevice));
        break;
    case einsums::detail::PINNED:
    case einsums::detail::MAPPED:
        hip_catch(hipMemcpy((void *)this->_data, (void const *)copy._host_data, copy.size() * sizeof(T), hipMemcpyHostToDevice));
        break;
    default:
        EINSUMS_THROW_EXCEPTION(uninitialized_error, "DeviceTensor being copied has not been initialized!");
    }

    // This will almost certainly allocate a full 4 kB, so optimize the packing.
    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 2 * sizeof(size_t) * rank));
    this->_gpu_strides = this->_gpu_dims + rank;

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));

    gpu::device_synchronize();
    return *this;
}

template <typename T, size_t rank>
auto DeviceTensor<T, rank>::operator=(DeviceTensor<T, rank> const &other) -> DeviceTensor<T, rank> & {
    if (_mode == einsums::detail::UNKNOWN) {
        return init(other);
    } else {
        return this->assign(other);
    }
}

template <typename T, size_t rank>
template <typename TOther>
    requires(!std::same_as<T, TOther>)
auto DeviceTensor<T, rank>::operator=(DeviceTensor<TOther, rank> const &other) -> DeviceTensor<T, rank> & {
    return this->assign(other);
}

template <typename T, size_t rank>
template <typename TOther>
auto DeviceTensor<T, rank>::operator=(DeviceTensorView<TOther, rank> const &other) -> DeviceTensor<T, rank> & {
    return this->assign(other);
}

template <typename T, size_t rank>
auto DeviceTensor<T, rank>::operator=(Tensor<T, rank> const &other) -> DeviceTensor<T, rank> & {
    return this->assign(other);
}

template <typename T, size_t rank>
auto DeviceTensor<T, rank>::operator=(T const &fill_value) -> DeviceTensor<T, rank> & {
    using namespace einsums::gpu;
    this->set_all(fill_value);
    return *this;
}
#endif

namespace detail {
/**
 * Kernel to do assignment and operation. This is addition.
 */
template <typename T, size_t rank>
__global__ void add_and_assign(T *to_data, size_t const *index_strides, size_t const *to_strides, T const *from_data,
                               size_t const *from_strides, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];
    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the view.
        size_t from_ind = einsums::indices_to_sentinel<rank>(from_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = to_data[to_ind] + from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and operation. This is subtraction.
 */
template <typename T, size_t rank>
__global__ void sub_and_assign(T *to_data, size_t const *index_strides, size_t const *to_strides, T const *from_data,
                               size_t const *from_strides, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the view.
        size_t from_ind = einsums::indices_to_sentinel<rank>(from_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = to_data[to_ind] - from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and operation. This is multiplication.
 */
template <typename T, size_t rank>
__global__ void mul_and_assign(T *to_data, size_t const *index_strides, size_t const *to_strides, T const *from_data,
                               size_t const *from_strides, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the view.
        size_t from_ind = einsums::indices_to_sentinel<rank>(from_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = gpu_ops::mult(to_data[to_ind], from_data[from_ind]);

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and operation. This is division.
 */
template <typename T, size_t rank>
__global__ void div_and_assign(T *to_data, size_t const *index_strides, size_t const *to_strides, T const *from_data,
                               size_t const *from_strides, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the view.
        size_t from_ind = einsums::indices_to_sentinel<rank>(from_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = gpu_ops::div(to_data[to_ind], from_data[from_ind]);

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and scalar operation. This is addition.
 */
template <typename T, size_t rank>
__global__ void add_and_assign_scal(T *to_data, size_t const *index_strides, size_t const *to_strides, T scalar, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = to_data[to_ind] + scalar;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and scalar operation. This is subtraction.
 */
template <typename T, size_t rank>
__global__ void sub_and_assign_scal(T *to_data, size_t const *index_strides, size_t const *to_strides, T scalar, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = to_data[to_ind] - scalar;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and scalar operation. This is multiplication.
 */
template <typename T, size_t rank>
__global__ void mul_and_assign_scal(T *to_data, size_t const *index_strides, size_t const *to_strides, T scalar, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = gpu_ops::mult(to_data[to_ind], scalar);

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}
/**
 * Kernel to do assignment and scalar operation. This is division.
 */
template <typename T, size_t rank>
__global__ void div_and_assign_scal(T *to_data, size_t const *index_strides, size_t const *to_strides, T scalar, size_t elements) {
    using namespace einsums::gpu;

    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    size_t inds[rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<rank>(curr_element, index_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = einsums::indices_to_sentinel<rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = gpu_ops::div(to_data[to_ind], scalar);

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

} // namespace detail

#ifndef DOXYGEN

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::add_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::add_and_assign_scal<dev_datatype, rank><<<blocks(this->size()), block_size(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, HipCast<dev_datatype, T>::cast(other), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::sub_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::sub_and_assign_scal<dev_datatype, rank><<<blocks(this->size()), block_size(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, HipCast<dev_datatype, T>::cast(other), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::mult_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::mul_and_assign_scal<dev_datatype, rank><<<blocks(this->size()), block_size(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, HipCast<dev_datatype, T>::cast(other), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::div_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::div_and_assign_scal<dev_datatype, rank><<<blocks(this->size()), block_size(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, HipCast<dev_datatype, T>::cast(other), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::add_assign(DeviceTensor<T, rank> const &other) {
    using namespace einsums::gpu;
    einsums::detail::add_and_assign<dev_datatype, rank><<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, other.gpu_data(), other.gpu_strides(), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::sub_assign(DeviceTensor<T, rank> const &other) {
    using namespace einsums::gpu;
    einsums::detail::sub_and_assign<dev_datatype, rank><<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, other.gpu_data(), other.gpu_strides(), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::mult_assign(DeviceTensor<T, rank> const &other) {
    using namespace einsums::gpu;
    einsums::detail::mul_and_assign<dev_datatype, rank><<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, other.gpu_data(), other.gpu_strides(), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank> &DeviceTensor<T, rank>::div_assign(DeviceTensor<T, rank> const &other) {
    using namespace einsums::gpu;
    einsums::detail::div_and_assign<dev_datatype, rank><<<blocks(other.size()), block_size(other.size()), 0, get_stream()>>>(
        this->_data, this->_gpu_strides, this->_gpu_strides, other.gpu_data(), other.gpu_strides(), _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t rank>
DeviceTensor<T, rank>::DeviceTensor(Tensor<T, rank> const &copy, einsums::detail::HostToDeviceMode mode) {
    using namespace einsums::gpu;
    this->_name = copy.name();
    this->_dims = copy.dims();
    dims_to_strides(_dims, _strides, true);
    this->_mode = mode;

    if (mode == einsums::detail::MAPPED) {
        this->_host_data = new T[copy.size()];
        hip_catch(hipHostRegister((void *)this->_host_data, copy.size() * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
    } else if (mode == einsums::detail::PINNED) {
        hip_catch(hipHostMalloc((void **)&(this->_host_data), copy.size() * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
    } else if (mode == einsums::detail::DEV_ONLY) {
        this->_host_data = nullptr;
        hip_catch(hipMalloc((void **)&(this->_data), copy.size() * sizeof(T)));
        hip_catch(hipMemcpy((void *)this->_data, (void const *)copy.data(), copy.size() * sizeof(T), hipMemcpyHostToDevice));
    } else {
        EINSUMS_THROW_EXCEPTION(enum_error, "Unknown occupancy mode!");
    }

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 2 * sizeof(size_t) * rank));
    this->_gpu_strides = this->_gpu_dims + rank;

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
    gpu::device_synchronize();

    if (copy.is_row_major()) {
        if (this->_mode == einsums::detail::DEV_ONLY) {
            hip_catch(hipMemcpy((void *)this->_data, (void const *)copy.data(), this->size() * sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        } else if (_mode == einsums::detail::MAPPED || _mode == einsums::detail::PINNED) {
            std::memcpy(this->_host_data, copy.data(), this->size() * sizeof(T));
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor being assigned to was not initialized!");
        }
    } else {
        if (_mode == einsums::detail::DEV_ONLY) {
            auto lock = copy.gpu_cache_tensor();
            dev_datatype const *copy_ptr = (dev_datatype const *)copy.get_gpu_pointer();

            size_t *copy_strides;

            hip_catch(hipMalloc((void **)&copy_strides, rank * sizeof(size_t)));
            {
                auto copy_strides_temp = copy.strides();

                hip_catch(
                    hipMemcpy((void *)copy_strides, (void const *)copy_strides_temp.data(), rank * sizeof(size_t), hipMemcpyHostToDevice));
            }

            einsums::detail::copy_to_tensor<dev_datatype, rank><<<blocks(copy.size()), block_size(copy.size()), 0, get_stream()>>>(
                this->_data, this->_gpu_strides, this->_gpu_strides, copy_ptr, copy_strides, copy.size());

            gpu::device_synchronize();

            hip_catch(hipFree((void *)copy_strides));
        } else if (_mode == einsums::detail::MAPPED || _mode == einsums::detail::PINNED) {
            auto copy_strides = copy.strides();
            for(size_t index = 0; index < copy.size(); index++) {
                size_t this_index, copy_index;
                sentinel_to_sentinels(index, _strides, _strides, this_index, copy_strides, copy_index);
                _host_data[this_index] = copy.data()[copy_index];
            }
        }
    }
}

template <typename T, size_t rank>
DeviceTensor<T, rank>::operator Tensor<T, rank>() const {
    using namespace einsums::gpu;
    Tensor<T, rank> out(true, this->_dims);

    out.set_name(this->_name);

    hip_catch(hipMemcpy((void *)out.data(), (void const *)this->_data, this->size() * sizeof(T), hipMemcpyDeviceToHost));
    // no sync needed

    return out;
}

template <typename T, size_t rank>
DeviceTensor<T, rank>::operator DeviceTensorView<T, rank>() {
    return DeviceTensorView<T, rank>(*this, _dims);
}

template <typename T, size_t rank>
DeviceTensor<T, rank>::operator DeviceTensorView<T, rank> const() const {
    return DeviceTensorView<T, rank>(*this, _dims);
}

#endif

} // namespace einsums

#endif
