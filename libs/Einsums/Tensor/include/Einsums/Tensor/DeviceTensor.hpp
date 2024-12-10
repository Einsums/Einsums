#ifndef DEVICE_TENSOR_HPP
#define DEVICE_TENSOR_HPP
// We use this so that the implementation headers work on their own.

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Tensor.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/AreAllConvertible.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#include <cstddef>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <numeric>
#include <vector>

namespace einsums {

template <typename T, size_t Rank>
struct DeviceTensorView;

template <typename T, size_t Rank>
struct DeviceTensor;

template <typename T, size_t Rank>
struct BlockDeviceTensor;

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
            hip_catch(hipMemcpy((void *)&out, (void const *)_ptr, sizeof(T), hipMemcpyDeviceToHost));
            // no sync

            return out;
        }
    }

    /**
     * Copy some data to the reference.
     */
    HostDevReference<T> &operator=(T const &other) {
        assert(_ptr != nullptr);
        if (is_on_host) {
            *_ptr = other;
        } else {
            T temp = other;
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * Copy some data to the reference.
     */
    HostDevReference<T> &operator=(HostDevReference<T> const &other) {
        if (is_on_host) {
            *_ptr = other.get();
        } else {
            if (other.is_on_host) {
                T temp = other.get();
                hip_catch(hipMemcpyAsync((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            } else if (this->_ptr != other._ptr) {
                hip_catch(hipMemcpyAsync((void *)_ptr, (void const *)other._ptr, sizeof(T), hipMemcpyDeviceToDevice, gpu::get_stream()));
            }
            gpu::stream_wait();
        }
        return *this;
    }

    /**
     * Add assignment.
     */
    HostDevReference<T> &operator+=(T const &other) {
        assert(_ptr != nullptr);
        if (is_on_host) {
            *_ptr += other;
        } else {
            T temp = other + get();
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * Add assignment.
     */
    HostDevReference<T> &operator+=(HostDevReference<T> const &other) {
        if (is_on_host) {
            *_ptr += other.get();
        } else {
            T temp = other.get() + get();
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * Sub assignment.
     */
    HostDevReference<T> &operator-=(T const &other) {
        assert(_ptr != nullptr);
        if (is_on_host) {
            *_ptr -= other;
        } else {
            T temp = get() - other;
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * Sub assignment.
     */
    HostDevReference<T> &operator-=(HostDevReference<T> const &other) {
        if (is_on_host) {
            *_ptr -= other.get();
        } else {
            T temp = get() - other.get();
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * Mult assignment.
     */
    HostDevReference<T> &operator*=(T const &other) {
        assert(_ptr != nullptr);
        if (is_on_host) {
            *_ptr *= other;
        } else {
            T temp = get() * other;
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * Mult assignment.
     */
    HostDevReference<T> &operator*=(HostDevReference<T> const &other) {
        if (is_on_host) {
            *_ptr *= other.get();
        } else {
            T temp = get() * other.get();
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * div assignment.
     */
    HostDevReference<T> &operator/=(T const &other) {
        assert(_ptr != nullptr);
        if (is_on_host) {
            *_ptr /= other;
        } else {
            T temp = get() / other;
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
        }
        return *this;
    }

    /**
     * Div assignment.
     */
    HostDevReference<T> &operator/=(HostDevReference<T> const &other) {
        if (is_on_host) {
            *_ptr /= other.get();
        } else {
            T temp = get() / other.get();
            hip_catch(hipMemcpy((void *)_ptr, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice));
            gpu::device_synchronize();
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
struct DeviceTensor : public virtual einsums::tensor_base::DeviceTensor,
                      virtual einsums::tensor_base::BasicTensor<T, Rank>,
                      virtual einsums::tensor_base::AlgebraOptimizedTensor,
                      virtual tensor_base::DeviceTypedTensor<T>,
                      virtual tensor_base::LockableTensor {
  public:
    using dev_datatype  = typename einsums::tensor_base::DeviceTypedTensor<T>::dev_datatype;
    using host_datatype = typename einsums::tensor_base::DeviceTypedTensor<T>::host_datatype;

    /**
     * @brief Construct a new tensor on the GPU.
     */
    DeviceTensor() = default;

    /**
     * @brief Copy construct a new GPU tensor.
     */
    DeviceTensor(DeviceTensor<T, Rank> const &other, detail::HostToDeviceMode mode = detail::UNKNOWN);

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
        requires requires {
            requires(sizeof...(Dims) == Rank);
            requires(!std::is_same_v<detail::HostToDeviceMode, Dims> && ...);
        }
    explicit DeviceTensor(std::string name, detail::HostToDeviceMode mode, Dims... dims);

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
        requires requires {
            requires(sizeof...(Dims) == Rank);
            requires(!std::is_same_v<detail::HostToDeviceMode, Dims> && ...);
        }
    explicit DeviceTensor(std::string name, Dims... dims);

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
    DeviceTensor(DeviceTensorView<T, Rank> const &other);

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    void resize(Dim<Rank> dims);

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     */
    template <typename... Dims>
    auto resize(Dims... dims) -> std::enable_if_t<(std::is_integral_v<Dims> && ... && (sizeof...(Dims) == Rank)), void> {
        resize(Dim<Rank>{static_cast<size_t>(dims)...});
    }

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
    auto gpu_data() -> dev_datatype * { return _data; }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return const T* An immutable pointer to the data.
     */
    auto gpu_data() const -> dev_datatype const * { return _data; }

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
    auto gpu_data(MultiIndex... index) -> dev_datatype *;

    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto gpu_data(MultiIndex... index) const -> dev_datatype const *;

    /**
     * @brief Returns a pointer to the host-readable data.
     *
     * Returns a pointer to the data as available to the host. If the tensor is in DEV_ONLY mode,
     * then this will return a null pointer. Otherwise, the pointer should be useable, but the data
     * may be outdated.
     *
     * @return T* A pointer to the data.
     */
    host_datatype *data() override { return _host_data; }

    /**
     * @brief Returns a pointer to the host-readable data.
     *
     * Returns a pointer to the data as available to the host. If the tensor is in DEV_ONLY mode,
     * then this will return a null pointer. Otherwise, the pointer should be useable, but the data
     * may be outdated.
     *
     * @return const T* An immutable pointer to the data.
     */
    host_datatype const *data() const override { return _host_data; }

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
    auto data(MultiIndex... index) -> host_datatype *;

    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto data(MultiIndex... index) const -> host_datatype const *;

    /**
     * Sends data from the host to the device.
     *
     * @param data The vector data.
     */
    void read(std::vector<T> const &data);

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
    void read(T const *data);

    /**
     * Sends data from the device to the host.
     *
     * @param data The vector that will be filled.
     */
    void write(T *data);

    /**
     * Assignments
     */
    DeviceTensor<T, Rank> &assign(DeviceTensor<T, Rank> const &other);
    DeviceTensor<T, Rank> &assign(Tensor<T, Rank> const &other);

    DeviceTensor<T, Rank> &init(DeviceTensor<T, Rank> const &other, einsums::detail::HostToDeviceMode mode = einsums::detail::UNKNOWN);

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    DeviceTensor<T, Rank> &assign(DeviceTensor<TOther, Rank> const &other);

    template <typename TOther>
    DeviceTensor<T, Rank> &assign(DeviceTensorView<TOther, Rank> const &other);

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
    auto operator()(MultiIndex... index) -> HostDevReference<T>;

    // WARNING: Chances are this function will not work if you mix All{}, Range{} and explicit indexes.
    /**
     * @brief Subscripts into the tensor and creates a view.
     */
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
    auto
    operator()(MultiIndex... index) -> DeviceTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>;

    /**
     * @brief Subscripts into the tensor and creates a view.
     */
    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> DeviceTensorView<T, Rank>;

    /**
     * @brief Copy data from one tensor to another.
     */
    auto operator=(DeviceTensor<T, Rank> const &other) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Copy data from one tensor to another, and convert types.
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(DeviceTensor<TOther, Rank> const &other) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Copy data from a tensor view into a tensor.
     */
    template <typename TOther>
    auto operator=(DeviceTensorView<TOther, Rank> const &other) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Copy data from one tensor to another.
     */
    auto operator=(Tensor<T, Rank> const &other) -> DeviceTensor<T, Rank> &;

    /**
     * Fill a tensor with a value.
     */
    auto operator=(T const &fill_value) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Operate and assign every element with a scalar.
     */
    DeviceTensor<T, Rank> &add_assign(T const &other);
    DeviceTensor<T, Rank> &mult_assign(T const &other);
    DeviceTensor<T, Rank> &sub_assign(T const &other);
    DeviceTensor<T, Rank> &div_assign(T const &other);

    DeviceTensor<T, Rank> &operator*=(T const &other) { return this->mult_assign(other); }
    DeviceTensor<T, Rank> &operator+=(T const &other) { return this->add_assign(other); }
    DeviceTensor<T, Rank> &operator-=(T const &other) { return this->sub_assign(other); }
    DeviceTensor<T, Rank> &operator/=(T const &other) { return this->div_assign(other); }

    /**
     * @brief Operate and assign two tensors element-wise.
     */
    DeviceTensor<T, Rank> &add_assign(DeviceTensor<T, Rank> const &other);
    DeviceTensor<T, Rank> &mult_assign(DeviceTensor<T, Rank> const &other);
    DeviceTensor<T, Rank> &sub_assign(DeviceTensor<T, Rank> const &other);
    DeviceTensor<T, Rank> &div_assign(DeviceTensor<T, Rank> const &other);

    DeviceTensor<T, Rank> &operator*=(DeviceTensor<T, Rank> const &other) { return this->mult_assign(other); }
    DeviceTensor<T, Rank> &operator+=(DeviceTensor<T, Rank> const &other) { return this->add_assign(other); }
    DeviceTensor<T, Rank> &operator-=(DeviceTensor<T, Rank> const &other) { return this->sub_assign(other); }
    DeviceTensor<T, Rank> &operator/=(DeviceTensor<T, Rank> const &other) { return this->div_assign(other); }

    /**
     * @brief Get the dimension for the given rank.
     */
    size_t dim(int d) const override {
        // Add support for negative indices.
        if (d < 0)
            d += Rank;
        return _dims[d];
    }

    /**
     * @brief Get all the dimensions.
     */
    Dim<Rank> dims() const override { return _dims; }

    /**
     * @brief Get the dimensions available to the GPU.
     */
    size_t *gpu_dims() { return _gpu_dims; }

    /**
     * @brief Get the dimensions available to the GPU.
     */
    size_t const *gpu_dims() const { return _gpu_dims; }

    /**
     * @brief Get the name of the tensor.
     */
    std::string const &name() const override { return _name; }

    /**
     * @brief Set the name of the tensor.
     */
    void set_name(std::string const &name) override { _name = name; }

    /**
     * @brief Get the stride of the given rank.
     */
    size_t stride(int d) const noexcept override {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    /**
     * @brief Get all the strides.
     */
    Stride<Rank> strides() const noexcept override { return _strides; }

    /**
     * @brief Get the strides available to the GPU.
     */
    size_t *gpu_strides() { return _gpu_strides; }

    /**
     * @brief Get the strides available to the GPU.
     */
    size_t const *gpu_strides() const { return _gpu_strides; }

    /**
     * Convert to a rank 1 tensor view.
     */
    DeviceTensorView<T, 1> to_rank_1_view() const {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return DeviceTensorView<T, 1>{*this, dim};
    }

    /// @brief Returns the linear size of the tensor
    size_t size() const { return _dims[0] * _strides[0]; }

    /**
     * @brief Whether this object is the full view.
     */
    bool full_view_of_underlying() const noexcept override { return true; }

    /**
     * Return the mode of the tensor.
     */
    detail::HostToDeviceMode mode() const { return _mode; }

    /**********************************************
     * Interface between device and host tensors. *
     **********************************************/

    /**
     * @brief Copy a host tensor to the device.
     */
    explicit DeviceTensor(Tensor<T, Rank> const &, detail::HostToDeviceMode mode = detail::MAPPED);

    /**
     * @brief Copy a device tensor to the host.
     */
    explicit operator Tensor<T, Rank>() const;

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
    size_t *_gpu_dims{nullptr};

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
    size_t *_gpu_strides{nullptr};

    /**
     * @property _data
     *
     * @brief A device pointer to the data on the device.
     */
    dev_datatype *_data{nullptr};

    /**
     * @property _host_data
     *
     * @brief If the tensor is mapped or pinned, this is the data on the host.
     */
    host_datatype *_host_data{nullptr};

    /**
     * @property _mode
     *
     * @brief The storage mode of the tensor.
     */
    detail::HostToDeviceMode _mode{detail::UNKNOWN};

    friend struct DeviceTensorView<T, Rank>;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensorView;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensor;
};

/**
 * @struct DeviceTensor<T, 0>
 *
 * Implementation for a zero-rank tensor.
 */
template <typename T>
struct DeviceTensor<T, 0> : public virtual tensor_base::DeviceTensor,
                            virtual tensor_base::BasicTensor<T, 0>,
                            virtual tensor_base::DeviceTypedTensor<T>,
                            virtual tensor_base::LockableTensor,
                            virtual tensor_base::AlgebraOptimizedTensor {
  public:
    using dev_datatype  = typename tensor_base::DeviceTypedTensor<T>::dev_datatype;
    using host_datatype = typename tensor_base::DeviceTypedTensor<T>::host_datatype;

    /**
     * @brief Construct a new tensor on the GPU.
     */
    DeviceTensor() : _mode(detail::DEV_ONLY) { hip_catch(hipMalloc((void **)&_data, sizeof(T))); }

    /**
     * @brief Copy construct a new GPU tensor.
     */
    DeviceTensor(DeviceTensor<T, 0> const &other, detail::HostToDeviceMode mode = detail::UNKNOWN) : _mode{mode} {
        if (mode == detail::DEV_ONLY) {
            hip_catch(hipMalloc((void **)&_data, sizeof(T)));
            hip_catch(hipMemcpyAsync((void *)_data, (void const *)other.gpu_data(), sizeof(T), hipMemcpyDeviceToDevice, gpu::get_stream()));
            gpu::stream_wait();
        } else if (mode == detail::MAPPED) {
            _host_data = new T((T)other);
            hip_catch(hipHostRegister((void *)_host_data, sizeof(T), hipHostRegisterDefault));
            hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        } else if (mode == detail::PINNED) {
            hip_catch(hipHostMalloc((void **)&_host_data, sizeof(T), 0));
            *_host_data = (T)other;
            hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        } else if (mode == detail::UNKNOWN) {
            _mode = other._mode;
            if (other._mode == detail::UNKNOWN) {
                EINSUMS_THROW_EXCEPTION(uninitialized_error, "Trying to copy uninitialized tensor!");
            }
        } else {
            EINSUMS_THROW_EXCEPTION(enum_error, "Unknown occupancy mode!");
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
            hip_catch(hipHostUnregister((void *)this->_host_data));
            delete this->_host_data;
        } else if (this->_mode == detail::PINNED) {
            hip_catch(hipHostFree((void *)this->_host_data));
        } else if (this->_mode == detail::DEV_ONLY) {
            hip_catch(hipFree((void *)this->_data));
        }
    }

    /**
     * @brief Construct a new tensor by dims.
     */
    DeviceTensor(Dim<0> dims, detail::HostToDeviceMode mode = detail::DEV_ONLY) : _mode{mode} {
        if (mode == detail::DEV_ONLY) {
            hip_catch(hipMalloc((void **)&_data, sizeof(T)));
        } else if (mode == detail::MAPPED) {
            _host_data = new T();
            hip_catch(hipHostRegister((void *)_host_data, sizeof(T), hipHostRegisterDefault));
            hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        } else if (mode == detail::PINNED) {
            hip_catch(hipHostMalloc((void **)&_host_data, sizeof(T), 0));
            hip_catch(hipHostGetDevicePointer((void **)&_data, (void *)_host_data, 0));
        } else {
            EINSUMS_THROW_EXCEPTION(enum_error, "Unknown occupancy mode!");
        }
    }

    /**
     * @brief Construct a new named zero-rank tensor on the GPU.
     */
    explicit DeviceTensor(std::string name, detail::HostToDeviceMode mode = detail::DEV_ONLY) : _name{std::move(name)}, _mode{mode} {
        switch (mode) {
        case detail::MAPPED:
            this->_host_data = new T();
            hip_catch(hipHostRegister((void *)this->_host_data, sizeof(T), hipHostRegisterDefault));
            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case detail::PINNED:
            hip_catch(hipHostMalloc((void **)&(this->_host_data), sizeof(T), 0));
            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case detail::DEV_ONLY:
            this->_host_data = nullptr;
            hip_catch(hipMalloc((void **)&(this->_data), sizeof(T)));
            break;
        default:
            EINSUMS_THROW_EXCEPTION(enum_error, "Unknown occupancy mode!");
        }
    }

    auto               gpu_data() -> dev_datatype               *{ return _data; }
    [[nodiscard]] auto gpu_data() const -> dev_datatype const * { return _data; }

    auto               data() -> host_datatype               *override { return _host_data; }
    [[nodiscard]] auto data() const -> host_datatype const * override { return _host_data; }

    auto operator=(DeviceTensor<T, 0> const &other) -> DeviceTensor<T, 0> & {
        hip_catch(hipMemcpyAsync((void *)_data, (void const *)other.gpu_data(), sizeof(T), hipMemcpyDeviceToDevice, gpu::get_stream()));
        gpu::stream_wait();
        return *this;
    }

    auto operator=(T const &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data = other;
        } else if (_mode == detail::DEV_ONLY) {
            hip_catch(hipMemcpyAsync((void *)_data, (void const *)&other, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            gpu::stream_wait();
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
        }
        return *this;
    }

    auto operator+=(T const &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data += other;
        } else if (_mode == detail::DEV_ONLY) {
            T temp;
            hip_catch(hipMemcpy((void *)&temp, (void const *)_data, sizeof(T), hipMemcpyDeviceToHost));
            // no sync
            temp += other;
            hip_catch(hipMemcpyAsync((void *)_data, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            gpu::stream_wait();
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
        }
        return *this;
    }

    auto operator-=(T const &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data -= other;
        } else if (_mode == detail::DEV_ONLY) {
            T temp;
            hip_catch(hipMemcpy((void *)&temp, (void const *)_data, sizeof(T), hipMemcpyDeviceToHost));
            // no sync
            temp -= other;
            hip_catch(hipMemcpyAsync((void *)_data, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            gpu::stream_wait();
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
        }
        return *this;
    }

    auto operator*=(T const &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data *= other;
        } else if (_mode == detail::DEV_ONLY) {
            T temp;
            hip_catch(hipMemcpy((void *)&temp, (void const *)_data, sizeof(T), hipMemcpyDeviceToHost));
            // no sync
            temp *= other;
            hip_catch(hipMemcpyAsync((void *)_data, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            gpu::stream_wait();
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
        }
        return *this;
    }

    auto operator/=(T const &other) -> DeviceTensor<T, 0> & {
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            *_host_data /= other;
        } else if (_mode == detail::DEV_ONLY) {
            T temp;
            hip_catch(hipMemcpy((void *)&temp, (void const *)_data, sizeof(T), hipMemcpyDeviceToHost));
            // no sync
            temp /= other;
            hip_catch(hipMemcpyAsync((void *)_data, (void const *)&temp, sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            gpu::stream_wait();
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
        }
        return *this;
    }

    operator T() const {
        T out;
        if (_mode == detail::MAPPED || _mode == detail::PINNED) {
            return *_host_data;
        } else if (_mode == detail::DEV_ONLY) {
            hip_catch(hipMemcpy((void *)&out, (void const *)_data, sizeof(T), hipMemcpyDeviceToHost));
            // no sync
            return out;
        } else {
            EINSUMS_THROW_EXCEPTION(uninitialized_error, "Tensor was not initialized!");
        }
    }

    [[nodiscard]] auto name() const -> std::string const & override { return _name; }
    void               set_name(std::string const &name) override { _name = name; }

    [[nodiscard]] auto dim(int) const -> size_t override { return 1; }

    [[nodiscard]] auto dims() const -> Dim<0> override { return Dim<0>{}; }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool override { return true; }

    /**
     * @brief Get the stride of the given rank.
     */
    [[nodiscard]] auto stride(int d) const noexcept -> size_t override { return 0; }

    /**
     * @brief Get all the strides.
     */
    auto strides() const noexcept -> Stride<0> override { return Stride<0>{}; }

    /**********************************************
     * Interface between device and host tensors. *
     **********************************************/

    /**
     * @brief Copy a host tensor to the device.
     */
    explicit DeviceTensor(Tensor<T, 0> const &other, detail::HostToDeviceMode mode = detail::MAPPED) : _mode{mode} {
        switch (mode) {
        case detail::MAPPED:
            this->_host_data = new T();
            *_host_data      = (T)other;
            hip_catch(hipHostRegister((void *)this->_host_data, sizeof(T), hipHostRegisterDefault));
            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case detail::PINNED:
            hip_catch(hipHostMalloc((void **)&(this->_host_data), sizeof(T), 0));
            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            *_host_data = (T)other;
            break;
        case detail::DEV_ONLY:
            this->_host_data = nullptr;
            hip_catch(hipMallocAsync((void **)&(this->_data), sizeof(T), gpu::get_stream()));
            hip_catch(hipMemcpyAsync((void *)this->_data, (void const *)other.data(), sizeof(T), hipMemcpyHostToDevice, gpu::get_stream()));
            gpu::stream_wait();
            break;
        default:
            EINSUMS_THROW_EXCEPTION(enum_error, "Could not understand occupancy mode!");
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
    dev_datatype *_data{nullptr};

    /**
     * @property _host_data
     *
     * @brief If the tensor is mapped or pinned, this is the data on the host.
     */
    host_datatype *_host_data{nullptr};

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
};

template <typename T, size_t Rank>
struct DeviceTensorView : public virtual tensor_base::BasicTensor<T, Rank>,
                          virtual tensor_base::DeviceTensor,
                          virtual tensor_base::TensorView<T, Rank, DeviceTensor<T, Rank>>,
                          virtual tensor_base::DeviceTypedTensor<T>,
                          virtual tensor_base::LockableTensor,
                          virtual tensor_base::AlgebraOptimizedTensor {
  public:
    using dev_datatype  = typename tensor_base::DeviceTypedTensor<T>::dev_datatype;
    using host_datatype = typename tensor_base::DeviceTypedTensor<T>::host_datatype;

    DeviceTensorView() = delete;

    /**
     * @brief Copy constructor.
     */
    DeviceTensorView(DeviceTensorView const &);

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
    DeviceTensorView(einsums::DeviceTensor<T, OtherRank> const &other, Dim<Rank> const &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(other, std::forward<Args>(args)...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensorView<T, OtherRank> &other, Dim<Rank> const &dim, Args &&...args)
        : _name{other.name()}, _dims{dim} {
        common_initialization(other, std::forward<Args>(args)...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensorView<T, OtherRank> const &other, Dim<Rank> const &dim, Args &&...args)
        : _name{other.name()}, _dims{dim} {
        common_initialization(const_cast<DeviceTensorView<T, OtherRank> &>(other), std::forward<Args>(args)...);
    }

    /**
     * Create a device tensor view that maps an in-core tensor to the GPU.
     */
    template <CoreBasicTensorConcept TensorType>
        requires(TensorType::Rank == Rank)
    explicit DeviceTensorView(TensorType &core_tensor) {
        _name                    = core_tensor.name();
        _dims                    = core_tensor.dims();
        _strides                 = core_tensor.strides();
        _full_view_of_underlying = true;
        _host_data               = core_tensor.data();

        _free_dev_data = true;

        dims_to_strides(_dims, _strides);

        hip_catch(hipMalloc((void **)&(_gpu_dims), 3 * sizeof(size_t) * Rank));
        this->_gpu_strides       = this->_gpu_dims + Rank;
        this->_gpu_index_strides = this->_gpu_dims + 2 * Rank;

        hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * Rank, hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * Rank, hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)this->_gpu_index_strides, (void const *)this->_index_strides.data(), sizeof(size_t) * Rank,
                            hipMemcpyHostToDevice));
        gpu::device_synchronize();

        hip_catch(hipHostRegister(_host_data, _strides[0] * _dims[0] * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer((void **)&_data, _host_data, 0));
    }

    DeviceTensorView<T, Rank> &assign(T const *other);

    template <template <typename, size_t> typename AType>
        requires DeviceRankTensor<AType<T, Rank>, Rank, T>
    DeviceTensorView<T, Rank> &assign(AType<T, Rank> const &other);

    void set_all(T const &value);

    void zero() { set_all(T{0.0}); }

    /**
     * @brief Copy as much data as is needed from the host pointer to the device.
     */
    auto operator=(T const *other) -> DeviceTensorView &;

    /**
     * @brief Copy data from another tensor.
     */
    template <template <typename, size_t> typename AType>
        requires DeviceRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(AType<T, Rank> const &other) -> DeviceTensorView &;

    /**
     * @brief Copy data from another tensor.
     */
    auto operator=(DeviceTensorView<T, Rank> const &other) -> DeviceTensorView &;

    /**
     * @brief Fill the view with a value.
     */
    auto operator=(T const &fill_value) -> DeviceTensorView & {
        this->set_all(fill_value);
        return *this;
    }

    DeviceTensorView<T, Rank> &mult_assign(T const &value);
    DeviceTensorView<T, Rank> &div_assign(T const &value);
    DeviceTensorView<T, Rank> &add_assign(T const &value);
    DeviceTensorView<T, Rank> &sub_assign(T const &value);

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator*=(T const &value) { return this->mult_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator/=(T const &value) { return this->div_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator+=(T const &value) { return this->add_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator-=(T const &value) { return this->sub_assign(value); }

    /**
     * @brief Returns a pointer to the host-readable data.
     *
     * Returns a pointer to the data as available to the host. If the tensor is in DEV_ONLY mode,
     * then this will return a null pointer. Otherwise, the pointer should be useable, but the data
     * may be outdated.
     *
     * @return T* A pointer to the data.
     */
    host_datatype *data() override { return _host_data; }

    /**
     * @brief Returns a pointer to the host-readable data.
     *
     * Returns a pointer to the data as available to the host. If the tensor is in DEV_ONLY mode,
     * then this will return a null pointer. Otherwise, the pointer should be useable, but the data
     * may be outdated.
     *
     * @return const T* An immutable pointer to the data.
     */
    host_datatype const *data() const override { return _host_data; }

    /**
     * @brief Get a pointer to the GPU data.
     */
    dev_datatype *gpu_data() { return _data; }

    /**
     * @brief Get a pointer to the GPU data.
     */
    dev_datatype const *gpu_data() const { return const_cast<dev_datatype const *>(_data); }

    /**
     * @brief Get a pointer to an element in the view.
     */
    template <typename... MultiIndex>
    dev_datatype *gpu_data(MultiIndex... index);

    /**
     * @brief Get a const pointer to an element in the view.
     */
    template <typename... MultiIndex>
    dev_datatype const *gpu_data(MultiIndex... index) const;

    /**
     * @brief Get a pointer to an element in the view.
     */
    dev_datatype *gpu_data_array(std::array<size_t, Rank> const &index_list);

    /**
     * @brief Get a const pointer to an element in the view.
     */
    dev_datatype const *gpu_data_array(std::array<size_t, Rank> const &index_list) const;

    /**
     * @brief Get a value from the view.
     */
    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> T;

    /**
     * @brief Get the dimension of the given rank.
     */
    size_t dim(int d) const override {
        if (d < 0)
            d += Rank;
        return _dims[d];
    }

    /**
     * @brief Get the dimensions of the view.
     */
    Dim<Rank> dims() const override { return _dims; }

    /**
     * @brief Get the dimensions of the view made available to the GPU.
     */
    size_t *gpu_dims() { return _gpu_dims; }

    /**
     * @brief Get the dimensions of the view made available to the GPU.
     */
    size_t const *gpu_dims() const { return _gpu_dims; }

    /**
     * @brief Get the name of the view.
     */
    std::string const &name() const override { return _name; }

    /**
     * @brief Set the name of the view.
     */
    void set_name(std::string const &name) override { _name = name; }

    /**
     * @brief Get the stride of the given rank.
     */
    size_t stride(int d) const noexcept override {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    /**
     * @brief Get the strides of the view.
     */
    Stride<Rank> strides() const noexcept override { return _strides; }

    /**
     * @brief Get the strides of the view made available to the GPU.
     */
    size_t *gpu_strides() { return _gpu_strides; }

    /**
     * @brief Get the strides of the view made available to the GPU.
     */
    size_t const *gpu_strides() const { return _gpu_strides; }

    /**
     * @brief Convert the view to a one-dimensional array.
     */
    auto to_rank_1_view() const -> DeviceTensorView<T, 1>;

    /**
     * @brief Whether the view wraps all the data.
     */
    bool full_view_of_underlying() const noexcept override { return _full_view_of_underlying; }

    /**
     * @brief Get the size of the view.
     */
    size_t size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

    operator Tensor<T, Rank>() const {
        DeviceTensor temp(*this);

        return (Tensor<T, Rank>)temp;
    }

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
    size_t *_gpu_dims{nullptr};

    /**
     * @property _strides
     *
     * @brief The strides of the view.
     */
    einsums::Stride<Rank> _strides, _index_strides;

    /**
     * @property _gpu_strides
     *
     * @brief The strides of the view made available to the GPU.
     */
    size_t *_gpu_strides{nullptr}, *_gpu_index_strides;
    // Offsets<Rank> _offsets;

    /**
     * @property _full_view_of_underlying
     *
     * @brief Whether the view captures all of the data.
     */
    bool _full_view_of_underlying{false};

    bool _free_dev_data{false};

    /**
     * @property _data
     *
     * @brief A pointer to the GPU data.
     */
    dev_datatype *_data{nullptr};

    /**
     * @property _host_data
     *
     * @brief A pointer to the host data.
     */
    host_datatype *_host_data{nullptr};
    /**
     * @brief Method for initializing the view.
     */
    template <template <typename, size_t> typename TensorType, size_t OtherRank, typename... Args>
    auto common_initialization(TensorType<T, OtherRank> const &other, Args &&...args)
        -> std::enable_if_t<std::is_base_of_v<::einsums::tensor_base::Tensor<T, OtherRank>, TensorType<T, OtherRank>>>;

    template <typename OtherT, size_t OtherRank>
    friend struct DeviceTensorView;

    template <typename OtherT, size_t OtherRank>
    friend struct DeviceTensor;
};

#ifdef __cpp_deduction_guides
template <typename... Args>
    requires(!std::is_same_v<Args, detail::HostToDeviceMode> && ...)
DeviceTensor(std::string const &, Args...) -> DeviceTensor<double, sizeof...(Args)>;
template <typename... Args>
    requires(!std::is_same_v<Args, detail::HostToDeviceMode> && ...)
DeviceTensor(std::string const &, detail::HostToDeviceMode, Args...) -> DeviceTensor<double, sizeof...(Args)>;
template <typename T, size_t OtherRank, typename... Dims>
explicit DeviceTensor(DeviceTensor<T, OtherRank> &&otherTensor, std::string name, Dims... dims) -> DeviceTensor<T, sizeof...(dims)>;
template <size_t Rank, typename... Args>
    requires(!std::is_same_v<Args, detail::HostToDeviceMode> && ...)
explicit DeviceTensor(Dim<Rank> const &, Args...) -> DeviceTensor<double, Rank>;
template <size_t Rank, typename... Args>
    requires(!std::is_same_v<Args, detail::HostToDeviceMode> && ...)
explicit DeviceTensor(Dim<Rank> const &, detail::HostToDeviceMode, Args...) -> DeviceTensor<double, Rank>;

template <typename T, size_t Rank, size_t OtherRank, typename... Args>
DeviceTensorView(DeviceTensor<T, OtherRank> &, Dim<Rank> const &, Args...) -> DeviceTensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
DeviceTensorView(DeviceTensor<T, OtherRank> const &, Dim<Rank> const &, Args...) -> DeviceTensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
DeviceTensorView(DeviceTensorView<T, OtherRank> &, Dim<Rank> const &, Args...) -> DeviceTensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
DeviceTensorView(DeviceTensorView<T, OtherRank> const &, Dim<Rank> const &, Args...) -> DeviceTensorView<T, Rank>;
template <typename T, size_t Rank, size_t OtherRank, typename... Args>
DeviceTensorView(std::string, DeviceTensor<T, OtherRank> &, Dim<Rank> const &, Args...) -> DeviceTensorView<T, Rank>;
#endif

} // namespace einsums

#include "Einsums/Tensor/Backends/DeviceTensor.hpp"
#include "Einsums/Tensor/Backends/DeviceTensorView.hpp"
#endif