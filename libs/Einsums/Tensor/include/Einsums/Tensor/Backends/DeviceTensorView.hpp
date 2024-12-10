#ifndef BACKENDS_DEVICE_TENSOR_VIEW_HPP
#define BACKENDS_DEVICE_TENSOR_VIEW_HPP

#include <Einsums/Tensor/Backends/DeviceTensor.hpp>
#include <Einsums/Tensor/DeviceTensor.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>

namespace einsums {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T, size_t Rank>
DeviceTensorView<T, Rank>::DeviceTensorView(DeviceTensorView<T, Rank> const &copy) {
    using namespace einsums::gpu;

    this->_dims    = copy.dims();
    this->_strides = copy.strides();
    this->_name    = copy.name();
    hip_catch(hipMalloc((void **)&_data, sizeof(T) * _dims[0] * _strides[0]));
    hip_catch(hipMemcpy((void *)this->_data, (void *)copy.gpu_data(), sizeof(T) * _dims[0] * _strides[0], hipMemcpyDeviceToDevice));
    this->_full_view_of_underlying = copy.full_view_of_underlying();

    dims_to_strides(_dims, _index_strides);

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 3 * sizeof(size_t) * Rank));
    this->_gpu_strides       = this->_gpu_dims + Rank;
    this->_gpu_index_strides = this->_gpu_dims + 2 * Rank;

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * Rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * Rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_index_strides, (void const *)this->_index_strides.data(), sizeof(size_t) * Rank,
                        hipMemcpyHostToDevice));
    gpu::device_synchronize();
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank>::~DeviceTensorView() {
    using namespace einsums::gpu;

    gpu::device_synchronize();
    hip_catch(hipFree((void *)this->_gpu_dims));

    if (_free_dev_data) {
        hip_catch(hipHostUnregister(_host_data));
    }
}

#endif

namespace detail {
template <typename T, size_t Rank>
__global__ void copy_to_tensor_array(T *to_data, size_t const *index_strides, size_t const *to_strides, T const *from_data,
                                     size_t elements) {
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

    size_t inds[Rank];

    while (curr_element < elements) {

        // Convert index into index combination.
        sentinel_to_indices<Rank>(curr_element, index_strides, inds);

        // Map index combination onto the tensor.
        size_t to_ind = indices_to_sentinel<Rank>(to_strides, inds);

        // Do the copy.
        to_data[to_ind] = from_data[curr_element];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}
} // namespace detail

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::assign(T const *data) {
    using namespace einsums::gpu;
    T *gpu_ptr;

    hip_catch(hipHostRegister((void *)data, this->size() * sizeof(T), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer((void **)&gpu_ptr, (void *)data, 0));

    einsums::detail::copy_to_tensor_array<T, Rank><<<block_size(this->size()), blocks(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_index_strides, this->_gpu_strides, gpu_ptr, this->size());

    hip_catch(hipStreamSynchronize(get_stream()));

    hip_catch(hipHostUnregister(data));

    return *this;
}

template <typename T, size_t Rank>
template <template <typename, size_t> typename AType>
    requires DeviceRankTensor<AType<T, Rank>, Rank, T>
auto DeviceTensorView<T, Rank>::assign(AType<T, Rank> const &other) -> DeviceTensorView<T, Rank> & {
    using namespace einsums::gpu;

    einsums::detail::copy_to_tensor<T, Rank><<<block_size(this->size()), blocks(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_index_strides, this->_gpu_strides, other.gpu_data(), other.gpu_strides(), this->size());

    gpu::stream_wait();

    return *this;
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::operator=(DeviceTensorView<T, Rank> const &other) -> DeviceTensorView<T, Rank> & {
    return this->assign(other);
}

template <typename T, size_t Rank>
void DeviceTensorView<T, Rank>::set_all(T const &fill_value) {
    using namespace einsums::gpu;
    einsums::detail::set_all<T, Rank><<<block_size(this->size()), blocks(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_index_strides, this->_gpu_strides, fill_value, _index_strides[0] * _dims[0]);
    gpu::stream_wait();
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::operator=(T const *other) -> DeviceTensorView<T, Rank> & {
    return this->assign(other);
}

template <typename T, size_t Rank>
template <template <typename, size_t> typename AType>
    requires DeviceRankTensor<AType<T, Rank>, Rank, T>
auto DeviceTensorView<T, Rank>::operator=(AType<T, Rank> const &other) -> DeviceTensorView<T, Rank> & {
    return this->assign(other);
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::add_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::add_and_assign_scal<T, Rank><<<block_size(this->size()), blocks(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_index_strides, this->_gpu_strides, other, _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::sub_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::sub_and_assign_scal<T, Rank><<<block_size(this->size()), blocks(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_index_strides, this->_gpu_strides, other, _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::mult_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::mul_and_assign_scal<T, Rank><<<block_size(this->size()), blocks(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_index_strides, this->_gpu_strides, other, _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::div_assign(T const &other) {
    using namespace einsums::gpu;
    einsums::detail::div_and_assign_scal<T, Rank><<<block_size(this->size()), blocks(this->size()), 0, get_stream()>>>(
        this->_data, this->_gpu_index_strides, this->_gpu_strides, other, _strides[0] * _dims[0]);
    gpu::stream_wait();
    return *this;
}

template <typename T, size_t Rank>
template <typename... MultiIndex>
auto DeviceTensorView<T, Rank>::gpu_data(MultiIndex... index) -> DeviceTensorView<T, Rank>::dev_datatype * {
    using namespace einsums::gpu;
    assert(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = ::std::array{static_cast<::std::int64_t>(index)...};
    for (auto [i, _index] : enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = ::std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
}

template <typename T, size_t Rank>
template <typename... MultiIndex>
auto DeviceTensorView<T, Rank>::gpu_data(MultiIndex... index) const -> DeviceTensorView<T, Rank>::dev_datatype const * {
    using namespace einsums::gpu;
    assert(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = ::std::array{static_cast<::std::int64_t>(index)...};
    for (auto [i, _index] : enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = ::std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::gpu_data_array(::std::array<size_t, Rank> const &index_list) -> DeviceTensorView<T, Rank>::dev_datatype * {
    using namespace einsums::gpu;
    size_t ordinal = ::std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::gpu_data_array(::std::array<size_t, Rank> const &index_list) const
    -> DeviceTensorView<T, Rank>::dev_datatype const * {
    using namespace einsums::gpu;
    size_t ordinal = ::std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
}

template <typename T, size_t Rank>
template <typename... MultiIndex>
auto DeviceTensorView<T, Rank>::operator()(MultiIndex... index) const -> T {
    using namespace einsums::gpu;
    T out;

    hip_catch(hipMemcpy(&out, this->gpu_data(index...), sizeof(T), hipMemcpyDeviceToHost));
    // no sync

    return out;
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::to_rank_1_view() const -> DeviceTensorView<T, 1> {
    using namespace einsums::gpu;
    if constexpr (Rank == 1) {
        return *this;
    } else {
        if (_strides[Rank - 1] != 1) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Creating a Rank-1 TensorView for this Tensor(View) is not supported.");
        }
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

#    if defined(EINSUMS_SHOW_WARNING)
        println("Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!");
#    endif

        return DeviceTensorView<T, 1>{*this, dim, Stride<1>{1}};
    }
}

template <typename T, size_t Rank>
template <template <typename, size_t> typename TensorType, size_t OtherRank, typename... Args>
auto DeviceTensorView<T, Rank>::common_initialization(TensorType<T, OtherRank> const &other, Args &&...args)
    -> ::std::enable_if_t<::std::is_base_of_v<::einsums::tensor_base::Tensor<T, OtherRank>, TensorType<T, OtherRank>>> {
    using namespace einsums::gpu;

    static_assert(Rank <= OtherRank, "A TensorView must be the same Rank or smaller than the Tensor being viewed.");

    set_mutex(other.get_mutex());

    Stride<Rank>      default_strides{};
    Offset<OtherRank> default_offsets{};
    Stride<Rank>      error_strides{};
    error_strides[0] = -1;

    // Check to see if the user provided a dim of "-1" in one place. If found then the user requests that we compute this
    // dimensionality for them.
    int nfound{0};
    int location{-1};
    for (auto [i, dim] : enumerate(_dims)) {
        if (dim == -1) {
            nfound++;
            location = i;
        }
    }

    if (nfound > 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "More than one -1 was provided.");
    }

    if (nfound == 1 && Rank == 1) {
        default_offsets.fill(0);
        default_strides.fill(1);

        auto offsets = arguments::get(default_offsets, args...);
        auto strides = arguments::get(default_strides, args...);

        _dims[location] = static_cast<std::int64_t>(std::ceil((other.size() - offsets[0]) / static_cast<float>(strides[0])));
    }

    if (nfound == 1 && Rank > 1) {
        EINSUMS_THROW_EXCEPTION(todo_error, "Haven't coded up this case yet.");
    }

    // If the Ranks are the same then use "other"s stride information
    if constexpr (Rank == OtherRank) {
        default_strides = other._strides;
        // Else since we're different Ranks we cannot automatically determine our stride and the user MUST
        // provide the information
    } else {
        if (::std::accumulate(_dims.begin(), _dims.end(), 1, ::std::multiplies<>()) == other.size()) {
            dims_to_strides(_dims, default_strides);
        } else {
            // Stride information cannot be automatically deduced.  It must be provided.
            default_strides = arguments::get(error_strides, args...);
            if (default_strides[0] == static_cast<size_t>(-1)) {
                EINSUMS_THROW_EXCEPTION(bad_logic, "Unable to automatically deduce stride information. Stride must be passed in.");
            }
        }
    }

    default_offsets.fill(0);

    // Use default_* unless the caller provides one to use.
    _strides                         = arguments::get(default_strides, args...);
    Offset<OtherRank> const &offsets = arguments::get(default_offsets, args...);

    dims_to_strides(_dims, _index_strides);

    // Determine the ordinal using the offsets provided (if any) and the strides of the parent
    size_t ordinal = ::std::inner_product(offsets.begin(), offsets.end(), other._strides.begin(), size_t{0});
    _data          = other._data + ordinal;
    if (other._host_data != nullptr) {
        _host_data = other._host_data + ordinal;
    } else {
        _host_data = nullptr;
    }

    hip_catch(hipMalloc((void **)&(this->_gpu_dims), 3 * sizeof(size_t) * Rank));
    this->_gpu_strides       = this->_gpu_dims + Rank;
    this->_gpu_index_strides = this->_gpu_dims + 2 * Rank;

    hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * Rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * Rank, hipMemcpyHostToDevice));
    hip_catch(hipMemcpy((void *)this->_gpu_index_strides, (void const *)this->_index_strides.data(), sizeof(size_t) * Rank,
                        hipMemcpyHostToDevice));
    gpu::device_synchronize();
}

#endif

} // namespace einsums
#endif
