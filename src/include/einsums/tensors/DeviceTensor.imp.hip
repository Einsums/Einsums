#pragma once

#include "einsums/_GPUUtils.hpp"

#include "einsums/DeviceTensor.hpp"

#include <__clang_hip_runtime_wrapper.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime_api.h>

namespace einsums {
namespace gpu {

template <typename T, size_t Rank>
DeviceTensor<T, Rank>::DeviceTensor(const DeviceTensor<T, Rank> &copy) {
    this->_name    = copy._name;
    this->_dims    = copy._dims;
    this->_strides = copy._strides;
    this->mode     = detail::DEV_ONLY;

    hip_catch(hipMalloc(&(this->_data, copy.size() * sizeof(T))));

    if (copy._mode == detail::DEV_ONLY) {
        hip_catch(hipMemcpy(this->_data, copy._data, copy.size() * sizeof(T), hipMemcpyDeviceToDevice));
    } else {
        hip_catch(hipMemcpy(this->_data, copy._host_data, copy.size() * sizeof(T), hipMemcpyHostToDevice));
    }

    this->_host_data = nullptr;

    hip_catch(hipMalloc(&(this->_gpu_dims), sizeof(Dim<Rank>)));
    hip_catch(hipMalloc(&(this->_gpu_strides), sizeof(Stride<Rank>)));

    hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank>::~DeviceTensor() {
    if (this->_mode == detail::MAPPED) {
        if (this->_host_data == nullptr) {
            return;
        }
        hip_catch(hipHostUnregister(this->_host_data));
        delete[] this->_host_data;
    } else if (this->mode == detail::DEV_ONLY) {
        if (this->_data == nullptr) {
            return;
        }
        hip_catch(hipFree(this->_data));
    } else if (this->mode == detail::PINNED) {
        if (this->_host_data == nullptr) {
            return;
        }
        hip_catch(hipHostFree(this->_host_data));
    }

    if (this->_gpu_dims != nullptr) {
        hip_catch(hipFree(this->_gpu_dims));
    }

    if (this->_gpu_strides != nullptr) {
        hip_catch(hipFree(this->_gpu_strides));
    }
}

template <typename T, size_t Rank, typename... Dims>
DeviceTensor<T, Rank>::DeviceTensor(std::string name, detail::HostToDeviceMode mode, Dims... dims)
    : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
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
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    if (mode == detail::MAPPED) {
        this->_host_data = new T[size];
        hip_catch(hipHostRegister(this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::PINNED) {
        hip_catch(hipHostMalloc(&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::DEV_ONLY) {
        this->_host_data = nullptr;
        hip_catch(hipMalloc(&(this->_data), size * sizeof(T)));
    }

    hip_catch(hipMalloc(&(this->_gpu_dims), sizeof(<Rank>)));
    hip_catch(hipMalloc(&(this->_gpu_strides), sizeof(Stride<Rank>)));

    hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));
}

template <typename T, size_t Rank, size_t OtherRank, typename... Dims>
DeviceTensor<T, Rank>::DeviceTensor(DeviceTensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims)
    : _name{std::move(name)}, _dims{static_cast<size_t>(dims)...} {
    this->_host_data   = existingTensor._host_data;
    this->_data        = existingTensor._data;
    this->_gpu_dims    = existingTensor._gpu_dims;
    this->_gpu_strides = existingTensor._gpu_strides;

    this->_mode = existingTensor._mode;

    existingTensor._host_data   = nullptr;
    existingTensor._data        = nullptr;
    existingTensor._gpu_dims    = nullptr;
    existingTensor._gpu_strides = nullptr;

    static_assert(Rank == sizeof...(dims), "Declared rank does not match provided dims");

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
    for (auto [i, dim] : enumerate(_dims)) {
        if (dim == -1) {
            nfound++;
            location = i;
        }
    }

    if (nfound > 1) {
        throw std::runtime_error("More than one -1 was provided.");
    }

    if (nfound == 1) {
        size_t size{1};
        for (auto [i, dim] : enumerate(_dims)) {
            if (i != location)
                size *= dim;
        }
        if (size > existingTensor.size()) {
            throw std::runtime_error("Size of new tensor is larger than the parent tensor.");
        }
        _dims[location] = existingTensor.size() / size;
    }

    // Row-major order of dimensions
    std::transform(_dims.rbegin(), _dims.rend(), _strides.rbegin(), Stride());
    size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];

    // Check size
    if (_data.size() != size) {
        throw std::runtime_error("Provided dims to not match size of parent tensor");
    }
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank>::DeviceTensor(Dim<Rank> dims, detail::HostToDeviceMode mode) : _dims{std::move(dims)} {
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

    if (mode == detail::MAPPED) {
        this->_host_data = new T[size];
        hip_catch(hipHostRegister(this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::PINNED) {
        hip_catch(hipHostMalloc(&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::DEV_ONLY) {
        this->_host_data = nullptr;
        hip_catch(hipMalloc(&(this->_data), size * sizeof(T)));
    }

    hip_catch(hipMalloc(&(this->_gpu_dims), sizeof(Dim<Rank>)));
    hip_catch(hipMalloc(&(this->_gpu_strides), sizeof(Stride<Rank>)));

    hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));
}

namespace detail {

/**
 * Turns a single numerical index into a list of indices into the tensor.
 */
__host__ __device__ template <size_t Rank>
void     index_to_combination(size_t index, const Dim<Rank> &dims, std::array<int, Rank> &out) {
    size_t quot = index;

    for (ssize_t i = Rank - 1; i >= 0; i--) {
        out[i] = quot % dims[i];
        quot /= dims[i];
    }
}

/**
 * Turns a list of indices into a single numerical index.
 */
__host__ __device__ template <size_t Rank>
size_t   combination_to_index(const std::array<int, Rank> &inds, const Dim<Rank> &dims, const Stride<Rank> &strides) {
    size_t out = 0;
    for (ssize_t i = 0; i < Rank; i++) {
        int ind = inds[i];

        if (ind < 0) {
            ind += dims[i];
        }

        out += strides[i] * ind;
    }

    return out;
}

/**
 * Kernel to copy a DeviceTensorView into a DeviceTensor object.
 */
__global__ template <typename T, size_t Rank>
void copy_to_tensor(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, const T *from_data, const Dim<Rank> *from_dims,
                    const Stride<Rank> *from_strides, size_t elements) {

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

        // Map index combination onto the view.
        size_t from_ind = combination_to_index(inds, *from_dims, *from_strides);

        // Map index combination onto the tensor.
        size_t to_ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        to_data[to_ind] = from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to copy a DeviceTensorView into a DeviceTensor object. Converts as well.
 */
__global__ template <typename T, size_t Rank, typename TOther>
void copy_to_tensor_conv(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, const TOther *from_data,
                         const Dim<Rank> *from_dims, const Stride<Rank> *from_strides, size_t elements) {

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

        // Map index combination onto the view.
        size_t from_ind = combination_to_index(inds, *from_dims, *from_strides);

        // Map index combination onto the tensor.
        size_t to_ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        to_data[to_ind] = (T)from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

consteval dim3 constructor_blocks = dim3(32), constructor_threads = dim3(32);

} // namespace detail

template <typename T, size_t Rank>
DeviceTensor<T, Rank>::DeviceTensor(const DeviceTensorView<T, Rank> &other) : _name{other.name()}, _dims{other.dims()} {
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

    if (mode == detail::MAPPED) {
        this->_host_data = new T[size];
        hip_catch(hipHostRegister(this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::PINNED) {
        hip_catch(hipHostMalloc(&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::DEV_ONLY) {
        this->_host_data = nullptr;
        hip_catch(hipMalloc(&(this->_data), size * sizeof(T)));
    }

    hip_catch(hipMalloc(&(this->_gpu_dims), sizeof(Dim<Rank>)));
    hip_catch(hipMalloc(&(this->_gpu_strides), sizeof(Stride<Rank>)));

    hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(deta<T, Rank>, constructor_threads, constructor_blocks, 0, 0, this->_data, this->_gpu_dims, this->_gpu_strides,
                       other.data(), other.gpu_dims(), other.gpu_strides(), size);

    hip_catch(hipDeviceSynchronize());
}

namespace detail {

// Set every entry in a tensor to an element.
__global__ template <typename T, size_t Rank>
void set_all(T *data, const Dim<Rank> *dims, const Stride<Rank> *strides, T value) {
    int worker, kernel_size;

    get_worker_info(worker, kernel_size);

    size_t curr_element = worker;

    // Wrap around to help save work load.
    size_t block_size = blockDim.x * blockDim.y * blockDim.z;

    size_t elements = strides->size() == 0 ? 0 : (*strides)[0] * (*dims)[0];
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
        size_t ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        data[ind] = value;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

} // namespace detail

template <typename T, size_t Rank>
void DeviceTensor<T, Rank>::zero() {
    hipLaunchKernelGGL(detail::set_all<T, Rank>, constructor_threads, constructor_blocks, 0, 0, this->_data, this->_gpu_dims,
                       this->_gpu_strides, (T)0);

    hip_catch(hipDeviceSynchronize());
}

template <typename T, size_t Rank>
void DeviceTensor<T, Rank>::set_all(T value) {
    hipLaunchKernelGGL(detail::set_all<T, Rank>, constructor_threads, constructor_blocks, 0, 0, this->_data, this->_gpu_dims,
                       this->_gpu_strides, value);

    hip_catch(hipDeviceSynchronize());
}

template <typename T, size_t Rank, typename... MultiIndex>
    requires requires {
        requires NoneOfType<AllT, MultiIndex...>;
        requires NoneOfType<Range, MultiIndex...>;
    }
device_ptr T *DeviceTensor<T, Rank>::data(MultiIndex... index) {
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
    assert(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = std::array{static_cast<std::int64_t>(index)...};
    for (auto [i, _index] : enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
    return _data + ordinal;
#endif
}

template <typename T, size_t Rank, typename... MultiIndex>
    requires requires {
        requires NoneOfType<AllT, MultiIndex...>;
        requires NoneOfType<Range, MultiIndex...>;
    }
T DeviceTensor<T, Rank>::operator()(MultiIndex... index) const {

    if (this->_mode == detail::MAPPED || this->_mode == detail::PINNED) {
        assert(sizeof...(MultiIndex) <= _dims.size());

        auto index_list = std::array{static_cast<std::int64_t>(index)...};
        for (auto [i, _index] : enumerate(index_list)) {
            if (_index < 0) {
                index_list[i] = _dims[i] + _index;
            }
        }
        size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});
        return this->_host_data[ordinal];
    } else if (this->_mode == detail::DEV_ONLY) {
        T out;
        hip_catch(hipMemcpy(&out, this->data(index...), sizeof(T), hipMemcpyHostToDevice));
        return out;
    }
}

template <typename T, size_t Rank, typename... MultiIndex>
    requires requires {
        requires NoneOfType<AllT, MultiIndex...>;
        requires NoneOfType<Range, MultiIndex...>;
    }
HostDevReference<T> &DeviceTensor<T, Rank>::operator()(MultiIndex... index) {
    assert(sizeof...(MultiIndex) <= _dims.size());

    auto index_list = std::array{static_cast<std::int64_t>(index)...};
    for (auto [i, _index] : enumerate(index_list)) {
        if (_index < 0) {
            index_list[i] = _dims[i] + _index;
        }
    }
    size_t ordinal = std::inner_product(index_list.begin(), index_list.end(), _strides.begin(), size_t{0});

    if (this->_mode == detail::MAPPED || this->_mode == detail::PINNED) {
        return *new HostDevReference<T>(this->_host_data + ordinal, true);
    } else if (this->_mode == detail::DEV_ONLY) {
        return *new HostDevReference<T>(this->_data + ordinal, false);
    }
}

template <typename T, size_t Rank, typename... MultiIndex>
    requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
auto DeviceTensor<T, Rank>::operator()(MultiIndex... index)
    -> DeviceTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()> {
    // Construct a TensorView using the indices provided as the starting point for the view.
    // e.g.:
    //    Tensor T{"Big Tensor", 7, 7, 7, 7};
    //    T(0, 0) === T(0, 0, :, :) === TensorView{T, Dims<2>{7, 7}, Offset{0, 0}, Stride{49, 1}} ??
    // println("Here");
    const auto &indices = std::forward_as_tuple(index...);

    Offset<Rank>                                                                         offsets;
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
            counter++;

        } else if constexpr (std::is_same_v<Range, std::tuple_element_t<i, std::tuple<MultiIndex...>>>) {
            auto range       = std::get<i>(indices);
            offsets[counter] = range[0];
            if (range[1] < 0) {
                auto temp = _dims[i] + range[1];
                range[1]  = temp;
            }
            dims[counter]    = range[1] - range[0];
            strides[counter] = _strides[i];
            counter++;
        }
    });

    return DeviceTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>{*this, std::move(dims),
                                                                                                             offsets, strides};
}

template <typename T, size_t Rank, typename... MultiIndex>
    requires NumOfType<Range, Rank, MultiIndex...>
auto DeviceTensor<T, Rank>::operator()(MultiIndex... index) const -> DeviceTensorView<T, Rank> {
    Dim<Rank>    dims{};
    Offset<Rank> offset{};
    Stride<Rank> stride = _strides;

    auto ranges = get_array_from_tuple<std::array<Range, Rank>>(std::forward_as_tuple(index...));

    for (int r = 0; r < Rank; r++) {
        auto range = ranges[r];
        offset[r]  = range[0];
        if (range[1] < 0) {
            auto temp = _dims[r] + range[1];
            range[1]  = temp;
        }
        dims[r] = range[1] - range[0];
    }

    return DeviceTensorView<T, Rank>{*this, std::move(dims), std::move(offset), std::move(stride)};
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator=(const DeviceTensor<T, Rank> &other) {
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

        if (this->_mode == detail::MAPPED) {
            hip_catch(hipHostUnregister(this->_host_data));

            delete[] this->_host_data;
            this->_host_data = new T[size];

            hip_catch(hipHostRegister(this->_host_data, size * sizeof(T), hipHostRegisterDefault));

            hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
        } else if (this->_mode == detail::PINNED) {
            hip_catch(hipHostFree(this->_host_data));

            hip_catch(hipHostMalloc(&(this->_host_data), size * sizeof(T), 0));

            hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
        } else if (this->_mode == detail::DEV_ONLY) {
            hip_catch(hipFree(this->_data));

            hip_catch(hipMalloc(&(this->_data), size * sizeof(T)));
        }

        hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
        hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(detail::copy_to_tensor<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                           this->_gpu_dims, this->_gpu_strides, other._data, other._gpu_dims, other._gpu_strides, size);
    } else {
        hipLaunchKernelGGL(detail::copy_to_tensor<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                           this->_gpu_dims, this->_gpu_strides, other._data, other._gpu_dims, other._gpu_strides, _strides[0] * _dims[0]);
    }

    hip_catch(hipDeviceSynchronize());

    return *this;
}

template <typename T, size_t Rank, typename TOther>
    requires(!std::same_as<T, TOther>)
auto DeviceTensor<T, Rank>::operator=(const DeviceTensor<TOther, Rank> &other) -> DeviceTensor<T, Rank> & {
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

        if (this->_mode == detail::MAPPED) {
            hip_catch(hipHostUnregister(this->_host_data));

            delete[] this->_host_data;
            this->_host_data = new T[size];

            hip_catch(hipHostRegister(this->_host_data, size * sizeof(T), hipHostRegisterDefault));

            hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
        } else if (this->_mode == detail::PINNED) {
            hip_catch(hipHostFree(this->_host_data));

            hip_catch(hipHostMalloc(&(this->_host_data), size * sizeof(T), 0));

            hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
        } else if (this->_mode == detail::DEV_ONLY) {
            hip_catch(hipFree(this->_data));

            hip_catch(hipMalloc(&(this->_data), size * sizeof(T)));
        }

        hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
        hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(detail::copy_to_tensor_conv<T, Rank, TOther>, detail::constructor_threads, detail::constructor_blocks, 0, 0,
                           this->_data, this->_gpu_dims, this->_gpu_strides, other._data, other._gpu_dims, other._gpu_strides, size);
    } else {
        hipLaunchKernelGGL(detail::copy_to_tensor_conv<T, Rank, TOther>, detail::constructor_threads, detail::constructor_blocks, 0, 0,
                           this->_data, this->_gpu_dims, this->_gpu_strides, other._data, other._gpu_dims, other._gpu_strides,
                           _strides[0] * _dims[0]);
    }

    hip_catch(hipDeviceSynchronize());

    return *this;
}

template <typename T, size_t Rank, typename TOther>
auto DeviceTensor<T, Rank>::operator=(const DeviceTensorView<TOther, Rank> &other) -> DeviceTensor<T, Rank> & {
    hipLaunchKernelGGL(detail::copy_to_tensor_conv<T, Rank, TOther>, detail::constructor_threads, detail::constructor_blocks, 0, 0,
                       this->_data, this->_gpu_dims, this->_gpu_strides, other.data(), other.gpu_dims(), other.gpu_strides(),
                       _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
auto DeviceTensor<T, Rank>::operator=(const T &fill_value) -> DeviceTensor<T, Rank> & {
    this->set_all(fill_value);
    return *this;
}

namespace detail {
/**
 * Kernel to do assignment and operation. This is addition.
 */
__global__ template <typename T, size_t Rank>
void add_and_assign(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, const T *from_data, const Dim<Rank> *from_dims,
                    const Stride<Rank> *from_strides, size_t elements) {

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

        // Map index combination onto the view.
        size_t from_ind = combination_to_index(inds, *from_dims, *from_strides);

        // Map index combination onto the tensor.
        size_t to_ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        to_data[to_ind] += from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and operation. This is subtraction.
 */
__global__ template <typename T, size_t Rank>
void sub_and_assign(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, const T *from_data, const Dim<Rank> *from_dims,
                    const Stride<Rank> *from_strides, size_t elements) {

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

        // Map index combination onto the view.
        size_t from_ind = combination_to_index(inds, *from_dims, *from_strides);

        // Map index combination onto the tensor.
        size_t to_ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        to_data[to_ind] -= from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and operation. This is multiplication.
 */
__global__ template <typename T, size_t Rank>
void mul_and_assign(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, const T *from_data, const Dim<Rank> *from_dims,
                    const Stride<Rank> *from_strides, size_t elements) {

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

        // Map index combination onto the view.
        size_t from_ind = combination_to_index(inds, *from_dims, *from_strides);

        // Map index combination onto the tensor.
        size_t to_ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        to_data[to_ind] *= from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and operation. This is division.
 */
__global__ template <typename T, size_t Rank>
void div_and_assign(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, const T *from_data, const Dim<Rank> *from_dims,
                    const Stride<Rank> *from_strides, size_t elements) {

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

        // Map index combination onto the view.
        size_t from_ind = combination_to_index(inds, *from_dims, *from_strides);

        // Map index combination onto the tensor.
        size_t to_ind = combination_to_index(inds, *to_dims, *to_strides);

        // Do the copy.
        to_data[to_ind] /= from_data[from_ind];

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and scalar operation. This is addition.
 */
__global__ template <typename T, size_t Rank>
void add_and_assign_scal(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, T scalar, size_t elements) {

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
        to_data[to_ind] += scalar;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and scalar operation. This is subtraction.
 */
__global__ template <typename T, size_t Rank>
void sub_and_assign_scal(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, T scalar, size_t elements) {

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
        to_data[to_ind] -= scalar;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and scalar operation. This is multiplication.
 */
__global__ template <typename T, size_t Rank>
void mul_and_assign_scal(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, T scalar, size_t elements) {

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
        to_data[to_ind] *= scalar;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

/**
 * Kernel to do assignment and scalar operation. This is division.
 */
__global__ template <typename T, size_t Rank>
void div_and_assign_scal(T *to_data, const Dim<Rank> *to_dims, const Stride<Rank> *to_strides, T scalar, size_t elements) {

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
        to_data[to_ind] /= scalar;

        // Increment.
        curr_element += kernel_size;
    }

    __threadfence();
}

} // namespace detail

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator+=(const T &other) {
    hipLaunchKernelGGL(detail::add_and_assign_scalar<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator-=(const T &other) {
    hipLaunchKernelGGL(detail::sub_and_assign_scalar<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator*=(const T &other) {
    hipLaunchKernelGGL(detail::mul_and_assign_scalar<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator/=(const T &other) {
    hipLaunchKernelGGL(detail::div_and_assign_scalar<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other, _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator+=(const DeviceTensor<T, Rank> &other) {
    hipLaunchKernelGGL(detail::add_and_assign<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other.data(), other.gpu_dims(), other.gpu_strides(), _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator-=(const DeviceTensor<T, Rank> &other) {
    hipLaunchKernelGGL(detail::sub_and_assign<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other.data(), other.gpu_dims(), other.gpu_strides(), _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator*=(const DeviceTensor<T, Rank> &other) {
    hipLaunchKernelGGL(detail::mul_and_assign<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other.data(), other.gpu_dims(), other.gpu_strides(), _strides[0] * _dims[0]);
    return *this;
}

template <typename T, size_t Rank>
DeviceTensor<T, Rank> &DeviceTensor<T, Rank>::operator/=(const DeviceTensor<T, Rank> &other) {
    hipLaunchKernelGGL(detail::div_and_assign<T, Rank>, detail::constructor_threads, detail::constructor_blocks, 0, 0, this->_data,
                       this->_gpu_dims, this->_gpu_strides, other.data(), other.gpu_dims(), other.gpu_strides(), _strides[0] * _dims[0]);
    return *this;
}

template<typename T, size_t Rank>
DeviceTensor<T, Rank>::DeviceTensor(const Tensor<T, Rank> &copy, detail::HostToDeviceMode mode) {
    this->_name    = copy._name;
    this->_dims    = copy._dims;
    this->_strides = copy._strides;
    this->_mode = mode;

    if (mode == detail::MAPPED) {
        this->_host_data = new T[size];
        hip_catch(hipHostRegister(this->_host_data, size * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::PINNED) {
        hip_catch(hipHostMalloc(&(this->_host_data), size * sizeof(T), 0));
        hip_catch(hipHostGetDevicePointer(&(this->_data), this->_host_data, 0));
    } else if (mode == detail::DEV_ONLY) {
        this->_host_data = nullptr;
        hip_catch(hipMalloc(&(this->_data), size * sizeof(T)));
    }

    hip_catch(hipMemcpy(this->_data, copy.data(), copy.size() * sizeof(T), hipMemcpyHostToDevice));

    hip_catch(hipMalloc(&(this->_gpu_dims), sizeof(<Rank>)));
    hip_catch(hipMalloc(&(this->_gpu_strides), sizeof(Stride<Rank>)));

    hip_catch(hipMemcpy(this->_gpu_dims, this->_dims, sizeof(Dim<Rank>), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(this->_gpu_strides, this->_strides, sizeof(Stride<Rank>), hipMemcpyHostToDevice));
}

} // namespace gpu
} // namespace einsums