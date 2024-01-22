#pragma once

#include "einsums/DeviceTensor.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"

namespace einsums {

namespace gpu {

template <typename T, size_t Rank>
DeviceTensorView<T, Rank>::DeviceTensorView(const DeviceTensorView<T, Rank> &copy) {

    this->_dims                    = copy.dims();
    this->_strides                 = copy.strides();
    this->_name                    = copy.name();
    this->_data                    = copy.data();
    this->_full_view_of_underlying = copy.full_view_of_underlying();

    hip_catch(hipMalloc(&(this->_gpu_dims), sizeof(Dim<Rank>)));
    hip_catch(hipMalloc(&(this->_gpu_strides), sizeof(Stride<Rank>)));

    hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank>::~DeviceTensorView() {
    hip_catch(hipFree(this->_gpu_dims));
    hip_catch(hipFree(this->_gpu_strides));
}

namespace detail {
template <typename T, size_t Rank>
__global__ void copy_to_tensor_array(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, const T *from_data,
                                     size_t elements) {

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

    while (curr_element < elements_adjusted) {
        std::array<int, Rank> inds;

        // Convert index into index combination.
        index_to_combination(curr_element % elements, *to_dims, inds);

        // Map index combination onto the tensor.
        size_t to_ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        to_data[to_ind] = from_data[curr_element % elements];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}
} // namespace detail

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::operator=(const host_ptr T *data) {
    T *gpu_ptr;

    hip_catch(hipHostRegister(data, this->size() * sizeof(T), hipHostRegisterDefault));
    hip_catch(hipHostGetDevicePointer(&gpu_ptr, data, 0));

    detail::copy_to_tensor_array<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(
        this->_data, this->_gpu_dims, this->_gpu_strides, gpu_ptr, this->size());

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipHostUnregister(data));

    return *this;
}

template <typename T, size_t Rank>
template <template <typename, size_t> typename AType>
    requires detail::DeviceRankTensor<AType<T, Rank>, Rank, T>
auto DeviceTensorView<T, Rank>::operator=(const AType<T, Rank> &other) -> DeviceTensorView<T, Rank> & {

    detail::copy_to_tensor<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(
        this->_data, this->_gpu_dims, this->_gpu_strides, other.data(), other.gpu_dims(), other.gpu_strides(), this->size());

    hip_catch(hipDeviceSynchronize());

    return *this;
}

template <typename T, size_t Rank>
template <template <typename, size_t> typename AType>
    requires detail::DeviceRankTensor<AType<T, Rank>, Rank, T>
auto DeviceTensorView<T, Rank>::operator=(const AType<T, Rank> &&other) -> DeviceTensorView<T, Rank> & {

    detail::copy_to_tensor<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(
        this->_data, this->_gpu_dims, this->_gpu_strides, other.data(), other.gpu_dims(), other.gpu_strides(), this->size());

    hip_catch(hipDeviceSynchronize());

    return *this;
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::operator=(const T &fill_value) -> DeviceTensorView<T, Rank> & {
    detail::set_all<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(this->_data, this->_gpu_dims, this->_gpu_strides,
                                                                                          fill_value, this->size());

    hip_catch(hipDeviceSynchronize());
}
template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::operator+=(const T &other) {
    detail::add_and_assign_scal<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(
        this->_data, this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::operator-=(const T &other) {
    detail::sub_and_assign_scal<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(
        this->_data, this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::operator*=(const T &other) {
    detail::mul_and_assign_scal<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(
        this->_data, this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensorView<T, Rank> &DeviceTensorView<T, Rank>::operator/=(const T &other) {
    detail::div_and_assign_scal<T, Rank><<<detail::constructor_threads, detail::constructor_blocks>>>(
        this->_data, this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
template <typename... MultiIndex>
auto DeviceTensorView<T, Rank>::data(MultiIndex... index) const -> device_ptr T * {
    assert(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = std::array{static_cast<std::int64_t>(index)...};
    for (auto [i, _index] : enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::data_array(const std::array<size_t, Rank> &index_list) const -> device_ptr T * {
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
}

template <typename T, size_t Rank>
template <typename... MultiIndex>
auto DeviceTensorView<T, Rank>::operator()(MultiIndex... index) const -> T {
    T out;
    hip_catch(hipMemcpy(&out, this->data(index...), sizeof(T), hipMemcpyDeviceToHost));

    return out;
}

template <typename T, size_t Rank>
auto DeviceTensorView<T, Rank>::to_rank_1_view() const -> DeviceTensorView<T, 1> {
    if constexpr (Rank == 1) {
        return *this;
    } else {
        if (_strides[Rank - 1] != 1) {
            throw std::runtime_error("Creating a Rank-1 TensorView for this Tensor(View) is not supported.");
        }
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

#if defined(EINSUMS_SHOW_WARNING)
        println("Creating a Rank-1 TensorView of an existing TensorView may not work. Be careful!");
#endif

        return DeviceTensorView<T, 1>{*this, dim, Stride<1>{1}};
    }
}

template <typename T, size_t Rank>
template <template <typename, size_t> typename TensorType, size_t OtherRank, typename... Args>
auto DeviceTensorView<T, Rank>::common_initialization(TensorType<T, OtherRank> &other, Args &&...args)
    -> std::enable_if_t<std::is_base_of_v<einsums::detail::TensorBase<T, OtherRank>, TensorType<T, OtherRank>>> {

    static_assert(Rank <= OtherRank, "A TensorView must be the same Rank or smaller that the Tensor being viewed.");

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
        throw std::runtime_error("More than one -1 was provided.");
    }

    if (nfound == 1 && Rank == 1) {
        default_offsets.fill(0);
        default_strides.fill(1);

        auto offsets = arguments::get(default_offsets, args...);
        auto strides = arguments::get(default_strides, args...);

        _dims[location] = static_cast<std::int64_t>(std::ceil((other.size() - offsets[0]) / static_cast<float>(strides[0])));
    }

    if (nfound == 1 && Rank > 1) {
        throw std::runtime_error("Haven't coded up this case yet.");
    }

    // If the Ranks are the same then use "other"s stride information
    if constexpr (Rank == OtherRank) {
        default_strides = other._strides;
        // Else since we're different Ranks we cannot automatically determine our stride and the user MUST
        // provide the information
    } else {
        if (std::accumulate(_dims.begin(), _dims.end(), 1.0, std::multiplies<>()) ==
            std::accumulate(other._dims.begin(), other._dims.end(), 1.0, std::multiplies<>())) {
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
            std::transform(_dims.rbegin(), _dims.rend(), default_strides.rbegin(), Stride());
            size_t size = default_strides.size() == 0 ? 0 : default_strides[0] * _dims[0];
        } else {
            // Stride information cannot be automatically deduced.  It must be provided.
            default_strides = arguments::get(error_strides, args...);
            if (default_strides[0] == static_cast<size_t>(-1)) {
                throw std::runtime_error("Unable to automatically deduce stride information. Stride must be passed in.");
            }
        }
    }

    default_offsets.fill(0);

    // Use default_* unless the caller provides one to use.
    _strides                         = arguments::get(default_strides, args...);
    const Offset<OtherRank> &offsets = arguments::get(default_offsets, args...);

    // Determine the ordinal using the offsets provided (if any) and the strides of the parent
    size_t ordinal = std::inner_product(offsets.begin(), offsets.end(), other._strides.begin(), size_t{0});
    _data          = other._data + ordinal;

    hip_catch(hipMalloc(&(this->_gpu_dims), sizeof(Dim<Rank>)));
    hip_catch(hipMalloc(&(this->_gpu_strides), sizeof(Stride<Rank>)));

    hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));
}

} // namespace gpu

} // namespace einsums
