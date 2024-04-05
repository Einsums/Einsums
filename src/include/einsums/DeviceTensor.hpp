#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/Tensor.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

namespace einsums {

template <typename T, size_t Rank>
struct DeviceTensorView;

template <typename T, size_t Rank>
struct DeviceTensor;

template <typename T, size_t Rank>
struct BlockDeviceTensor;

namespace detail {

/**
 * @enum HostToDeviceMode
 *
 * @brief Enum that specifies how device tensors store data and make it available to the GPU.
 */
enum HostToDeviceMode { DEV_ONLY, MAPPED, PINNED };

/**
 * Turns a single sentinel value into an index combination.
 *
 * @param index The sentinel value.
 * @param dims The dimensions of the tensor along each axis.
 * @param out The output array.
 */
template <size_t Rank>
__host__ __device__ void index_to_combination(size_t index, const size_t *dims, size_t *out);

/**
 * Turns a single sentinel value into an index combination.
 *
 * @param index The sentinel value.
 * @param dims The dimensions of the tensor along each axis.
 * @param out The output array.
 */
template <size_t Rank>
__host__ __device__ void index_to_combination(size_t index, const einsums::Dim<Rank> &dims, std::array<int, Rank> &out);

/**
 * @brief Turns a combination of indices into a single sentinel value that can be used to index into an array.
 *
 * @param inds The indices for each dimension.
 * @param dims The dimensions of the tensor.
 * @param strides The strides for each dimension.
 *
 * @return A one-dimensional index for a tensor's data array.
 */
template <size_t Rank>
__host__ __device__ size_t combination_to_index(const std::array<size_t, Rank> &inds, const einsums::Dim<Rank> &dims,
                                                const einsums::Stride<Rank> &strides);
/**
 * @brief Turns a combination of indices into a single sentinel value that can be used to index into an array.
 *
 * @param inds The indices for each dimension.
 * @param dims The dimensions of the tensor.
 * @param strides The strides for each dimension.
 *
 * @return A one-dimensional index for a tensor's data array.
 */
template <size_t Rank>
__host__ __device__ size_t combination_to_index(const size_t *inds, const size_t *dims, const size_t *strides);

} // namespace detail

/**
 * @class HostDevReference
 *
 * @brief Wraps some functionality of a reference to allow host-device communication.
 * This class provides some functionality of a reference, but the data may actually be stored on the device.
 * Data is copied back and forth with each call.
 *
 * @note It is best to avoid needing this class, as a whole bunch of small memory transfers is very slow.
 */
template <typename T>
class HostDevReference {
  private:
    /**
     * @property _ptr
     *
     * @brief The pointer held by this object.
     */
    T *_ptr;

    /**
     * @property is_on_host
     *
     * @brief True if the pointer is a host pointer. False if it is a device pointer.
     */
    bool is_on_host;

  public:
    /**
     * Construct an empty reference.
     */
    HostDevReference() : _ptr{nullptr}, is_on_host{true} {}

    /**
     * Construct a reference wrapping the specified pointer.
     */
    HostDevReference(T *ptr, bool is_host) : _ptr{ptr}, is_on_host{is_host} {}

    /**
     * Delete the reference. Because the data is managed by something else, don't acutally free the pointer.
     */
    ~HostDevReference() { _ptr = nullptr; }

    /**
     * Get the value of the reference.
     */
    T get() const {
        if (_ptr == nullptr) {
            return T{0};
        }
        if (is_on_host) {
            return *_ptr;
        } else {
            T out;
            gpu::hip_catch(hipMemcpy((void *)&out, (const void *)_ptr, sizeof(T), hipMemcpyDeviceToHost));
            return out;
        }
    }

    /**
     * Copy some data to the reference.
     */
    HostDevReference<T> &operator=(const T &other) {
        assert(_ptr != nullptr);
        if (is_on_host) {
            *_ptr = other;
        } else {
            T temp = other;
            gpu::hip_catch(hipMemcpy((void *)_ptr, (const void *)&temp, sizeof(T), hipMemcpyHostToDevice));
        }
        return *this;
    }

    /**
     * Copy some data to the reference.
     */
    HostDevReference<T> &operator=(const HostDevReference<T> &other) {
        if (is_on_host) {
            *_ptr = other.get();
        } else {
            if (other.is_on_host) {
                T temp = other.get();
                gpu::hip_catch(hipMemcpy((void *)_ptr, (const void *)&temp, sizeof(T), hipMemcpyHostToDevice));
            } else if (this->_ptr != other._ptr) {
                gpu::hip_catch(hipMemcpy((void *)_ptr, (const void *)other._ptr, sizeof(T), hipMemcpyDeviceToDevice));
            }
        }
        return *this;
    }

    /**
     * Get the address handled by the reference.
     */
    T *operator&() { return _ptr; }

    /**
     * Convert to the underlying type.
     */
    operator T() { return this->get(); }
};

/**
 * @struct DeviceTensor
 *
 * @brief Makes tensor functionality available to the GPU.
 *
 * @tparam T The type of the data managed by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct DeviceTensor : public ::einsums::detail::TensorBase<T, Rank> {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    /**
     * @typedef host_datatype
     *
     * @brief The data type stored on the host. It is an alias of T.
     */
    using host_datatype = T;

  private:
    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(Unnamed)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the tensor.
     */
    einsums::Dim<Rank> _dims;

    /**
     * @property _gpu_dims
     *
     * @brief The dimensions of the tensor made available to the GPU.
     */
    size_t *_gpu_dims;

    /**
     * @property _strides
     *
     * @brief The strides of the tensor.
     */
    einsums::Stride<Rank> _strides;

    /**
     * @property _gpu_strides
     *
     * @brief The strides of the tensor made available to the GPU.
     */
    size_t *_gpu_strides;

    /**
     * @property _data
     *
     * @brief A device pointer to the data on the device.
     */
    __device_ptr__ dev_datatype *_data;

    /**
     * @property _host_data
     *
     * @brief If the tensor is mapped or pinned, this is the data on the host.
     */
    __host_ptr__ host_datatype *_host_data;

    /**
     * @property _mode
     *
     * @brief The storage mode of the tensor.
     */
    detail::HostToDeviceMode _mode;

    friend struct DeviceTensorView<T, Rank>;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensorView;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensor;

  public:
    /**
     * @brief Construct a new tensor on the GPU.
     */
    DeviceTensor() = default;

    /**
     * @brief Copy construct a new GPU tensor.
     */
    DeviceTensor(const DeviceTensor<T, Rank> &);

    /**
     * @brief Destructor.
     */
    ~DeviceTensor();

    /**
     * @brief Construct a new DeviceTensor object with the given name and dimensions.
     *
     * Constructs a new DeviceTensor object using the information provided in \p name and \p dims .
     *
     * @code
     * auto A = DeviceTensor("A", 3, 3);
     * @endcode
     *
     * The newly constructed DeviceTensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling DeviceTensor.zero() or zero(DeviceTensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param dims The dimensions of each rank of the tensor.
     */
    template <typename... Dims>
    explicit DeviceTensor(std::string name, detail::HostToDeviceMode mode, Dims... dims);

    // Once this is called "otherTensor" is no longer a valid tensor.
    /**
     * @brief Construct a new DeviceTensor object. Moving \p existingTensor data to the new tensor.
     *
     * This constructor is useful for reshaping a tensor. It does not modify the underlying
     * tensor data. It only creates new mapping arrays for how the data is viewed.
     *
     * @code
     * auto A = DeviceTensor("A", 27); // Creates a rank-1 tensor of 27 elements
     * auto B = DeviceTensor(std::move(A), "B", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     * // At this point A is no longer valid.
     * @endcode
     *
     * Supports using -1 for one of the ranks to automatically compute the dimensional of it.
     *
     * @code
     * auto A = DeviceTensor("A", 27);
     * auto B = DeviceTensor(std::move(A), "B", 3, -1, 3); // Automatically determines that -1 should be 3.
     * @endcode
     *
     * @tparam OtherRank The rank of \p existingTensor can be different than the rank of the new tensor
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param existingTensor The existing tensor that holds the tensor data.
     * @param name The name of the new tensor
     * @param dims The dimensionality of each rank of the new tensor.
     */
    template <size_t OtherRank, typename... Dims>
    explicit DeviceTensor(DeviceTensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims);

    /**
     * @brief Construct a new DeviceTensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     */
    explicit DeviceTensor(Dim<Rank> dims, detail::HostToDeviceMode mode = detail::DEV_ONLY);

    /**
     * @brief Construct a new Tensor object from a TensorView.
     *
     * Data is explicitly copied from the view to the new tensor.
     *
     * @param other The tensor view to copy.
     */
    DeviceTensor(const DeviceTensorView<T, Rank> &other);

    /**
     * @brief Zeroes out the tensor data.
     */
    void zero();

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     */
    void set_all(T value);

    /**
     * @brief Returns a pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return T* A pointer to the data.
     */
    auto data() -> dev_datatype * { return _data; }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    auto data() const -> const dev_datatype * { return _data; }

    /**
     * @brief Returns a pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return T* A pointer to the data.
     */
    auto host_data() -> host_datatype * { return _host_data; }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    auto host_data() const -> const host_datatype * { return _host_data; }

    /**
     * Returns a pointer into the tensor at the given location.
     *
     * @code
     * auto A = DeviceTensor("A", 3, 3, 3); // Creates a rank-3 tensor of 27 elements
     *
     * double* A_pointer = A.data(1, 2, 3); // Returns the pointer to element (1, 2, 3) in A.
     * @endcode
     *
     *
     * @tparam MultiIndex The datatypes of the passed parameters. Must be castable to
     * @param index The explicit desired index into the tensor. Must be castable to std::int64_t.
     * @return A pointer into the tensor at the requested location.
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> dev_datatype *;

    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto data(MultiIndex... index) const -> const dev_datatype *;

    /**
     * Sends data from the host to the device.
     *
     * @param data The vector data.
     */
    void read(const std::vector<T> &data);

    /**
     * Sends data from the device to the host.
     *
     * @param data The vector that will be filled.
     */
    void write(std::vector<T> &data);

    /**
     * Sends data from the host to the device.
     *
     * @param data The vector data.
     */
    void read(const T *data);

    /**
     * Sends data from the device to the host.
     *
     * @param data The vector that will be filled.
     */
    void write(T *data);

    /**
     * Assignments with a stream.
     */
    DeviceTensor<T, Rank> &assign(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &assign(const Tensor<T, Rank> &other);

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    DeviceTensor<T, Rank> &assign(const DeviceTensor<TOther, Rank> &other);

    DeviceTensor<T, Rank> &assign(const DeviceTensorView<T, Rank> &other);

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    DeviceTensor<T, Rank> &assign(const DeviceTensorView<TOther, Rank> &other);

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) const -> T;

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return const T&
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto operator()(MultiIndex... index) -> HostDevReference<T> &;

    // WARNING: Chances are this function will not work if you mix All{}, Range{} and explicit indexes.
    /**
     * @brief Subscripts into the tensor and creates a view.
     */
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
    auto operator()(MultiIndex... index)
        -> DeviceTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>;

    /**
     * @brief Subscripts into the tensor and creates a view.
     */
    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> DeviceTensorView<T, Rank>;

    /**
     * @brief Copy data from one tensor to another.
     */
    auto operator=(const DeviceTensor<T, Rank> &other) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Copy data from one tensor to another, and convert types.
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const DeviceTensor<TOther, Rank> &other) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Copy data from a tensor view into a tensor.
     */
    template <typename TOther>
    auto operator=(const DeviceTensorView<TOther, Rank> &other) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Copy data from one tensor to another.
     */
    auto operator=(const Tensor<T, Rank> &other) -> DeviceTensor<T, Rank> &;

    /**
     * Fill a tensor with a value.
     */
    auto operator=(const T &fill_value) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Operate and assign every element with a scalar.
     */
    DeviceTensor<T, Rank> &add_assign(const T &other);
    DeviceTensor<T, Rank> &mult_assign(const T &other);
    DeviceTensor<T, Rank> &sub_assign(const T &other);
    DeviceTensor<T, Rank> &div_assign(const T &other);

    DeviceTensor<T, Rank> &operator*=(const T &other) { return this->mult_assign(other); }
    DeviceTensor<T, Rank> &operator+=(const T &other) { return this->add_assign(other); }
    DeviceTensor<T, Rank> &operator-=(const T &other) { return this->sub_assign(other); }
    DeviceTensor<T, Rank> &operator/=(const T &other) { return this->div_assign(other); }

    /**
     * @brief Operate and assign two tensors element-wise.
     */
    DeviceTensor<T, Rank> &add_assign(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &mult_assign(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &sub_assign(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &div_assign(const DeviceTensor<T, Rank> &other);

    DeviceTensor<T, Rank> &operator*=(const DeviceTensor<T, Rank> &other) { return this->mult_assign(other); }
    DeviceTensor<T, Rank> &operator+=(const DeviceTensor<T, Rank> &other) { return this->add_assign(other); }
    DeviceTensor<T, Rank> &operator-=(const DeviceTensor<T, Rank> &other) { return this->sub_assign(other); }
    DeviceTensor<T, Rank> &operator/=(const DeviceTensor<T, Rank> &other) { return this->div_assign(other); }

    /**
     * @brief Get the dimension for the given rank.
     */
    [[nodiscard]] auto dim(int d) const -> size_t {
        // Add support for negative indices.
        if (d < 0)
            d += Rank;
        return _dims[d];
    }

    /**
     * @brief Get all the dimensions.
     */
    auto dims() const -> Dim<Rank> { return _dims; }

    /**
     * @brief Get the dimensions available to the GPU.
     */
    __device_ptr__ size_t *gpu_dims() { return _gpu_dims; }

    /**
     * @brief Get the dimensions available to the GPU.
     */
    const __device_ptr__ size_t *gpu_dims() const { return _gpu_dims; }

    ALIAS_TEMPLATE_FUNCTION(shape, dims);

    /**
     * @brief Get the name of the tensor.
     */
    [[nodiscard]] auto name() const -> const std::string & { return _name; }

    /**
     * @brief Set the name of the tensor.
     */
    void set_name(const std::string &name) { _name = name; }

    /**
     * @brief Get the stride of the given rank.
     */
    [[nodiscard]] auto stride(int d) const noexcept -> size_t {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    /**
     * @brief Get all the strides.
     */
    auto strides() const noexcept -> const auto & { return _strides; }

    /**
     * @brief Get the strides available to the GPU.
     */
    __device_ptr__ size_t *gpu_strides() { return _gpu_strides; }

    /**
     * @brief Get the strides available to the GPU.
     */
    const __device_ptr__ size_t *gpu_strides() const { return _gpu_strides; }

    /**
     * Convert to a rank 1 tensor view.
     */
    auto to_rank_1_view() const -> DeviceTensorView<T, 1> {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return DeviceTensorView<T, 1>{*this, dim};
    }

    /// @brief Returns the linear size of the tensor
    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

    /**
     * @brief Whether this object is the full view.
     */
    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }

    /**********************************************
     * Interface between device and host tensors. *
     **********************************************/

    /**
     * @brief Copy a host tensor to the device.
     */
    explicit DeviceTensor(const Tensor<T, Rank> &, detail::HostToDeviceMode mode = detail::MAPPED);

    /**
     * @brief Copy a device tensor to the host.
     */
    explicit operator Tensor<T, Rank>() const;
};

/**
 * @struct DeviceTensor<T, 0>
 *
 * Implementation for a zero-rank tensor.
 */
template <typename T>
struct DeviceTensor<T, 0> : public einsums::detail::TensorBase<T, 0> {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    /**
     * @typedef host_datatype
     *
     * @brief The data type stored on the host. It is an alias of T.
     */
    using host_datatype = T;

  private:
    /**
     * @property _name
     *
     * @brief The name of the tensor.
     */
    std::string _name{"(Unnamed)"};

    /**
     * @property _data
     *
     * @brief A device pointer to the data on the device.
     */
    __device_ptr__ dev_datatype *_data{nullptr};

    /**
     * @property _host_data
     *
     * @brief If the tensor is mapped or pinned, this is the data on the host.
     */
    __host_ptr__ host_datatype *_host_data{nullptr};

    /**
     * @property _mode
     *
     * @brief The storage mode of the tensor.
     */
    detail::HostToDeviceMode _mode;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensorView;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensor;

  public:
    /**
     * @brief Construct a new tensor on the GPU.
     */
    DeviceTensor() : _mode(detail::DEV_ONLY) { gpu::hip_catch(hipMallocAsync((void **)&_data, sizeof(T), gpu::get_stream())); }

    /**
     * @brief Copy construct a new GPU tensor.
     */
    DeviceTensor(const DeviceTensor<T, 0> &other, detail::HostToDeviceMode mode = detail::DEV_ONLY) : _mode{mode} {
        if (mode == detail::DEV_ONLY) {
            gpu::hip_catch(hipMallocAsync((void **)&_data, sizeof(T), gpu::get_stream()));
            gpu::hip_catch(hipMemcpyAsync((void *)_data, (const void *)other.data(), sizeof(T), hipMemcpyDeviceToDevice, gpu::get_stream()));
        } else if (mode == detail::MAPPED) {
            _host_data = new T((T)other);
            gpu::hip_catch(hipHostRegister((void *)_host_data, sizeof(T), hipHostRegisterDefault));
            gpu::hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        } else {
            gpu::hip_catch(hipHostMalloc((void **)&_host_data, sizeof(T), 0));
            *_host_data = (T)other;
            gpu::hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        }
    }

    /**
     * @brief Destructor.
     */
    ~DeviceTensor() {
        if (this->_mode == detail::MAPPED) {
            if (this->_host_data == nullptr) {
                return;
            }
            gpu::hip_catch(hipHostUnregister((void *)this->_host_data));
            delete this->_host_data;
        } else if (this->_mode == detail::PINNED) {
            gpu::hip_catch(hipHostFree((void *)this->_host_data));
        } else if (this->_mode == detail::DEV_ONLY) {
            gpu::hip_catch(hipFreeAsync((void *)this->_data, gpu::get_stream()));
        }
    }

    /**
     * @brief Construct a new tensor by dims.
     */
    DeviceTensor(Dim<0> dims, detail::HostToDeviceMode mode = detail::DEV_ONLY) : _mode{mode} {
        if (mode == detail::DEV_ONLY) {
            gpu::hip_catch(hipMallocAsync((void **)&_data, sizeof(T), gpu::get_stream()));
        } else if (mode == detail::MAPPED) {
            _host_data = new T();
            gpu::hip_catch(hipHostRegister((void *)_host_data, sizeof(T), hipHostRegisterDefault));
            gpu::hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        } else {
            gpu::hip_catch(hipHostMalloc((void **)&_host_data, sizeof(T), 0));
            gpu::hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        }
    }

    /**
     * @brief Construct a new named zero-rank tensor on the GPU.
     */
    explicit DeviceTensor(std::string name, detail::HostToDeviceMode mode = detail::DEV_ONLY) : _name{std::move(name)}, _mode{mode} {
        switch (mode) {
        case detail::MAPPED:
            this->_host_data = new T();
            gpu::hip_catch(hipHostRegister((void *)this->_host_data, sizeof(T), hipHostRegisterDefault));
            gpu::hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case detail::PINNED:
            gpu::hip_catch(hipHostMalloc((void **)&(this->_host_data), sizeof(T), 0));
            gpu::hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case detail::DEV_ONLY:
            this->_host_data = nullptr;
            gpu::hip_catch(hipMallocAsync((void **)&(this->_data), sizeof(T), gpu::get_stream()));
            break;
        }
    }

    auto               data() -> dev_datatype               *{ return _data; }
    [[nodiscard]] auto data() const -> const dev_datatype * { return _data; }

    auto               host_data() -> host_datatype               *{ return _host_data; }
    [[nodiscard]] auto host_data() const -> const host_datatype * { return _host_data; }

    auto operator=(const DeviceTensor<T, 0> &other) -> DeviceTensor<T, 0> & {
        gpu::hip_catch(hipMemcpyAsync((void *)_data, (const void *)other.data(), sizeof(T), hipMemcpyDeviceToDevice, gpu::get_stream()));
        return *this;
    }

    auto operator=(const T &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data = other;
        } else {
            gpu::hip_catch(hipMemcpyAsync((void *)_data, (const void *)&other, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
        }
        return *this;
    }

    auto operator+=(const T &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data += other;
        } else {
            T temp;
            gpu::hip_catch(hipMemcpy((void *)&temp, (const void *)_data, sizeof(T), hipMemcpyDeviceToHost));
            temp += other;
            gpu::hip_catch(hipMemcpyAsync((void *)_data, (const void *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
        }
        return *this;
    }

    auto operator-=(const T &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data -= other;
        } else {
            T temp;
            gpu::hip_catch(hipMemcpy((void *)&temp, (const void *)_data, sizeof(T), hipMemcpyDeviceToHost));
            temp -= other;
            gpu::hip_catch(hipMemcpyAsync((void *)_data, (const void *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
        }
        return *this;
    }

    auto operator*=(const T &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data *= other;
        } else {
            T temp;
            gpu::hip_catch(hipMemcpy((void *)&temp, (const void *)_data, sizeof(T), hipMemcpyDeviceToHost));
            temp *= other;
            gpu::hip_catch(hipMemcpyAsync((void *)_data, (const void *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
        }
        return *this;
    }

    auto operator/=(const T &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data /= other;
        } else {
            T temp;
            gpu::hip_catch(hipMemcpy((void *)&temp, (const void *)_data, sizeof(T), hipMemcpyDeviceToHost));
            temp /= other;
            gpu::hip_catch(hipMemcpyAsync((void *)_data, (const void *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
        }
        return *this;
    }

    operator T() const {
        T out;
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            return *_host_data;
        } else {
            gpu::hip_catch(hipMemcpy((void *)&out, (const void *)_data, sizeof(T), hipMemcpyDeviceToHost));
            return out;
        }
    }

    [[nodiscard]] auto name() const -> const std::string & { return _name; }
    void               set_name(const std::string &name) { _name = name; }

    [[nodiscard]] auto dim(int) const -> size_t override { return 1; }

    [[nodiscard]] auto dims() const -> Dim<0> { return Dim<0>{}; }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }

    /**********************************************
     * Interface between device and host tensors. *
     **********************************************/

    /**
     * @brief Copy a host tensor to the device.
     */
    explicit DeviceTensor(const Tensor<T, 0> &other, detail::HostToDeviceMode mode = detail::MAPPED) {
        switch (mode) {
        case detail::MAPPED:
            this->_host_data = new T();
            *_host_data      = (T)other;
            gpu::hip_catch(hipHostRegister((void *)this->_host_data, sizeof(T), hipHostRegisterDefault));
            gpu::hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case detail::PINNED:
            gpu::hip_catch(hipHostMalloc((void **)&(this->_host_data), sizeof(T), 0));
            gpu::hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            *_host_data = (T)other;
            break;
        case detail::DEV_ONLY:
            this->_host_data = nullptr;
            gpu::hip_catch(hipMallocAsync((void **)&(this->_data), sizeof(T), gpu::get_stream()));
            gpu::hip_catch(hipMemcpyAsync((void *)this->_data, (const void *)other.data(), sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            break;
        }
    }

    /**
     * @brief Copy a device tensor to the host.
     */
    explicit operator Tensor<T, 0>() {
        Tensor<T, 0> out(std::move(_name));

        out = (T) * this;
        return out;
    }
};

template <typename T, size_t Rank>
struct DeviceTensorView : public ::einsums::detail::TensorBase<T, Rank> {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;

    /**
     * @typedef host_datatype
     *
     * @brief The data type stored on the host. It is an alias of T.
     */
    using host_datatype = T;

  private:
    /**
     * @property _name
     *
     * @brief The name of the view.
     */
    std::string _name{"(Unnamed View)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the view.
     */
    einsums::Dim<Rank> _dims;

    /**
     * @property _gpu_dims
     *
     * @brief The dimensions of the view made available to the GPU.
     */
    size_t *_gpu_dims;

    /**
     * @property _strides
     *
     * @brief The strides of the view.
     */
    einsums::Stride<Rank> _strides;

    /**
     * @property _gpu_strides
     *
     * @brief The strides of the view made available to the GPU.
     */
    size_t *_gpu_strides;
    // Offsets<Rank> _offsets;

    /**
     * @property _full_view_of_underlying
     *
     * @brief Whether the view captures all of the data.
     */
    bool _full_view_of_underlying{false};

    /**
     * @property _data
     *
     * @brief A pointer to the GPU data.
     */
    dev_datatype *_data;

  public:
    DeviceTensorView() = delete;

    /**
     * @brief Copy constructor.
     */
    DeviceTensorView(const DeviceTensorView &);

    /**
     * @brief Destructor.
     */
    ~DeviceTensorView();

    // std::enable_if doesn't work with constructors.  So we explicitly create individual
    // constructors for the types of tensors we support (Tensor and TensorView).  The
    // call to common_initialization is able to perform an enable_if check.
    /**
     * @brief Create a tensor view around the given tensor.
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(const DeviceTensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(const_cast<DeviceTensor<T, OtherRank> &>(other), args...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        common_initialization(other, args...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensorView<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(other, args...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(const DeviceTensorView<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(const_cast<DeviceTensorView<T, OtherRank> &>(other), args...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(std::string name, DeviceTensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{std::move(name)}, _dims{dim} {
        common_initialization(other, args...);
    }

    DeviceTensorView<T, Rank> &assign(const __host_ptr__ T *other);

    template <template <typename, size_t> typename AType>
        requires DeviceRankTensor<AType<T, Rank>, Rank, T>
    DeviceTensorView<T, Rank> &assign(const AType<T, Rank> &other);

    void set_all(const T &value);

    /**
     * @brief Copy as much data as is needed from the host pointer to the device.
     */
    auto operator=(const __host_ptr__ T *other) -> DeviceTensorView &;

    /**
     * @brief Copy data from another tensor.
     */
    template <template <typename, size_t> typename AType>
        requires DeviceRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(const AType<T, Rank> &other) -> DeviceTensorView &;

    /**
     * @brief Copy data from a tensor.
     */
    template <template <typename, size_t> typename AType>
        requires DeviceRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(const AType<T, Rank> &&other) -> DeviceTensorView &;

    /**
     * @brief Fill the view with a value.
     */
    auto operator=(const T &fill_value) -> DeviceTensorView & {
        this->set_all(fill_value);
        return *this;
    }

    DeviceTensorView<T, Rank> &mult_assign(const T &value);
    DeviceTensorView<T, Rank> &div_assign(const T &value);
    DeviceTensorView<T, Rank> &add_assign(const T &value);
    DeviceTensorView<T, Rank> &sub_assign(const T &value);

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator*=(const T &value) { return this->mult_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator/=(const T &value) { return this->div_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator+=(const T &value) { return this->add_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator-=(const T &value) { return this->sub_assign(value); }

    /**
     * @brief Get a pointer to the GPU data.
     */
    auto data() -> dev_datatype * { return _data; }

    /**
     * @brief Get a pointer to the GPU data.
     */
    auto data() const -> const dev_datatype * { return static_cast<const T *>(_data); }

    /**
     * @brief Get a pointer to an element in the view.
     */
    template <typename... MultiIndex>
    auto data(MultiIndex... index) const -> dev_datatype *;

    /**
     * @brief Get a pointer to an element in the view.
     */
    auto data_array(const std::array<size_t, Rank> &index_list) const -> __device_ptr__ T *;

    /**
     * @brief Get a value from the view.
     */
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> T;

    /**
     * @brief Get the dimension of the given rank.
     */
    [[nodiscard]] auto dim(int d) const -> size_t {
        if (d < 0)
            d += Rank;
        return _dims[d];
    }

    /**
     * @brief Get the dimensions of the view.
     */
    auto dims() const -> Dim<Rank> { return _dims; }

    /**
     * @brief Get the dimensions of the view made available to the GPU.
     */
    __device_ptr__ size_t *gpu_dims() { return _gpu_dims; }

    /**
     * @brief Get the dimensions of the view made available to the GPU.
     */
    const __device_ptr__ size_t *gpu_dims() const { return _gpu_dims; }

    /**
     * @brief Get the name of the view.
     */
    [[nodiscard]] auto name() const -> const std::string & { return _name; }

    /**
     * @brief Set the name of the view.
     */
    void set_name(const std::string &name) { _name = name; }

    /**
     * @brief Get the stride of the given rank.
     */
    [[nodiscard]] auto stride(int d) const noexcept -> size_t {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    /**
     * @brief Get the strides of the view.
     */
    auto strides() const noexcept -> const auto & { return _strides; }

    /**
     * @brief Get the strides of the view made available to the GPU.
     */
    __device_ptr__ size_t *gpu_strides() { return _gpu_strides; }

    /**
     * @brief Get the strides of the view made available to the GPU.
     */
    const __device_ptr__ size_t *gpu_strides() const { return _gpu_strides; }

    /**
     * @brief Convert the view to a one-dimensional array.
     */
    auto to_rank_1_view() const -> DeviceTensorView<T, 1>;

    /**
     * @brief Whether the view wraps all the data.
     */
    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return _full_view_of_underlying; }

    /**
     * @brief Get the size of the view.
     */
    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

  private:
    /**
     * @brief Method for initializing the view.
     */
    template <template <typename, size_t> typename TensorType, size_t OtherRank, typename... Args>
    auto common_initialization(TensorType<T, OtherRank> &other, Args &&...args)
        -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, OtherRank>, TensorType<T, OtherRank>>>;
};

} // namespace einsums

#include "einsums/gpu/DeviceTensor.hpp"
#include "einsums/gpu/DeviceTensorView.hpp"