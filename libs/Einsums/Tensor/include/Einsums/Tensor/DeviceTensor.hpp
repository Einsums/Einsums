//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#ifndef DEVICE_TENSOR_HPP
#define DEVICE_TENSOR_HPP
// We use this so that the implementation headers work on their own.

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/Tensor/TensorForward.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/TensorBase.hpp>
#include <Einsums/TypeSupport/AreAllConvertible.hpp>
#include <Einsums/TypeSupport/Arguments.hpp>
#include <Einsums/TypeSupport/CountOfType.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/TypeName.hpp>

#include <cstddef>
#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <numeric>
#include <vector>

namespace einsums {

#ifndef DOXYGEN
template <typename T, size_t Rank>
struct DeviceTensorView;

template <typename T, size_t Rank>
struct DeviceTensor;

template <typename T, size_t Rank>
struct BlockDeviceTensor;
#endif

/**
 * @class HostDevReference
 *
 * @brief Wraps some functionality of a reference to allow host-device communication.
 * This class provides some functionality of a reference, but the data may actually be stored on the device.
 * Data is copied back and forth with each call.
 *
 * @note It is best to avoid using this class in inner loops, as a whole bunch of small memory transfers is very slow.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
class HostDevReference {
  private:
    /**
     * @property _ptr
     *
     * @brief The pointer held by this object.
     *
     * @versionadded{1.0.0}
     */
    T *_ptr;

    /**
     * @property is_on_host
     *
     * @brief True if the pointer is a host pointer. False if it is a device pointer.
     *
     * @versionadded{1.0.0}
     */
    bool is_on_host;

  public:
    /**
     * Construct an empty reference.
     *
     * @versionadded{1.0.0}
     */
    HostDevReference() : _ptr{nullptr}, is_on_host{true} {}

    /**
     * Construct a reference wrapping the specified pointer.
     *
     * @versionadded{1.0.0}
     */
    HostDevReference(T *ptr, bool is_host) : _ptr{ptr}, is_on_host{is_host} {}

    /**
     * Delete the reference. Because the data is managed by something else, don't acutally free the pointer.
     *
     * @versionadded{1.0.0}
     */
    ~HostDevReference() { _ptr = nullptr; }

    /**
     * Get the value of the reference.
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
     */
    T *operator&() { return _ptr; }

    /**
     * Convert to the underlying type.
     *
     * @versionadded{1.0.0}
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
 *
 * @versionadded{1.0.0}
 */
template <typename T, size_t rank>
struct DeviceTensor : public einsums::tensor_base::DeviceTensorBase,
                      einsums::tensor_base::AlgebraOptimizedTensor,
                      design_pats::Lockable<std::recursive_mutex> {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The type of data used by the GPU.
     *
     * For real data types, this is the same as on the host. For complex data types, the appropriate
     * HIP data type needs to be used.
     *
     * @versionadded{1.0.0}
     */
    using dev_datatype = typename einsums::tensor_base::DeviceTypedTensor<T>::dev_datatype;

    /**
     * @typedef host_datatype
     *
     * @brief The type of data used by the CPU.
     *
     * This is the same as the ValueType.
     *
     * @versionadded{1.0.0}
     */
    using host_datatype = typename einsums::tensor_base::DeviceTypedTensor<T>::host_datatype;

    /**
     * @typedef ValueType
     *
     * @brief The type of data stored by the tensor.
     *
     * @versionadded{1.0.0}
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     *
     * @versionadded{1.0.0}
     */
    constexpr static size_t Rank = rank;

    /**
     * @brief Construct a new tensor on the GPU.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor() = default;

    /**
     * @brief Copy construct a new GPU tensor.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor(DeviceTensor<T, rank> const &other, detail::HostToDeviceMode mode = detail::UNKNOWN);

    /**
     * @brief Destructor.
     *
     * @versionadded{1.0.0}
     */
    ~DeviceTensor();

    /**
     * @brief Construct a new DeviceTensor object with the given name and dimensions.
     *
     * Constructs a new DeviceTensor object using the information provided in \p name and \p dims .
     *
     * @code
     * auto A = DeviceTensor("A", detail::DEV_ONLY, 3, 3);
     * @endcode
     *
     * The newly constructed DeviceTensor is NOT zeroed out for you. If you start having NaN issues
     * in your code try calling DeviceTensor.zero() or zero(DeviceTensor) to see if that resolves it.
     *
     * @tparam Dims Variadic template arguments for the dimensions. Must be castable to size_t.
     * @param name Name of the new tensor.
     * @param mode The storage mode of the tensor.
     * @param dims The dimensions of each rank of the tensor.
     *
     * @versionadded{1.0.0}
     */
    template <typename... Dims>
        requires requires {
            requires(sizeof...(Dims) == rank);
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
     *
     * @versionadded{1.0.0}
     */
    template <typename... Dims>
        requires requires {
            requires(sizeof...(Dims) == rank);
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
     *
     * @versionadded{1.0.0}
     */
    template <size_t OtherRank, typename... Dims>
    explicit DeviceTensor(DeviceTensor<T, OtherRank> &&existingTensor, std::string name, Dims... dims);

    /**
     * @brief Construct a new DeviceTensor object using the dimensions given by Dim object.
     *
     * @param dims The dimensions of the new tensor in Dim form.
     * @param mode The storage mode of the tensor.
     *
     * @versionadded{1.0.0}
     */
    explicit DeviceTensor(Dim<rank> dims, detail::HostToDeviceMode mode = detail::DEV_ONLY);

    /**
     * @brief Construct a new Tensor object from a TensorView.
     *
     * Data is explicitly copied from the view to the new tensor.
     *
     * @param other The tensor view to copy.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor(DeviceTensorView<T, rank> const &other);

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     *
     * @versionadded{1.0.0}
     */
    void resize(Dim<rank> dims);

    /**
     * @brief Resize a tensor.
     *
     * @param dims The new dimensions of a tensor.
     *
     * @versionadded{1.0.0}
     */
    template <typename... Dims>
    auto resize(Dims... dims) -> std::enable_if_t<(std::is_integral_v<Dims> && ... && (sizeof...(Dims) == rank)), void> {
        resize(Dim<rank>{static_cast<size_t>(dims)...});
    }

    /**
     * @brief Zeroes out the tensor data.
     *
     * @versionadded{1.0.0}
     */
    void zero();

    /**
     * @brief Set the all entries to the given value.
     *
     * @param value Value to set the elements to.
     *
     * @versionadded{1.0.0}
     */
    void set_all(T value);

    /**
     * @brief Returns a pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return A pointer to the data.
     *
     * @versionadded{1.0.0}
     */
    auto gpu_data() -> dev_datatype * { return _data; }

    /**
     * @brief Returns a constant pointer to the data.
     *
     * Try very hard to not use this function. Current data may or may not exist
     * on the host device at the time of the call if using GPU backend.
     *
     * @return An immutable pointer to the data.
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto gpu_data(MultiIndex... index) -> dev_datatype *;

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
     *
     * @versionadded{1.0.0}
     */
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
     *
     * @versionadded{1.0.0}
     */
    host_datatype *data() { return _host_data; }

    /**
     * @brief Returns a pointer to the host-readable data.
     *
     * Returns a pointer to the data as available to the host. If the tensor is in DEV_ONLY mode,
     * then this will return a null pointer. Otherwise, the pointer should be useable, but the data
     * may be outdated.
     *
     * @return const T* An immutable pointer to the data.
     *
     * @versionadded{1.0.0}
     */
    host_datatype const *data() const { return _host_data; }

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
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
        requires requires {
            requires NoneOfType<AllT, MultiIndex...>;
            requires NoneOfType<Range, MultiIndex...>;
        }
    auto data(MultiIndex... index) -> host_datatype *;

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
     *
     * @versionadded{1.0.0}
     */
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
     *
     * @versionadded{1.0.0}
     */
    void read(std::vector<T> const &data);

    /**
     * Sends data from the device to the host.
     *
     * @param data The vector that will be filled.
     *
     * @versionadded{1.0.0}
     */
    void write(std::vector<T> &data);

    /**
     * Sends data from the host to the device.
     *
     * @param data The vector data.
     *
     * @versionadded{1.0.0}
     */
    void read(T const *data);

    /**
     * Sends data from the device to the host.
     *
     * @param data The vector that will be filled.
     *
     * @versionadded{1.0.0}
     */
    void write(T *data);

    /**
     * @brief Copy the data from one tensor to this one.
     *
     * @param other The tensor with the data to copy.
     *
     * @return A reference to the calling tensor.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor<T, rank> &assign(DeviceTensor<T, rank> const &other);

    /**
     * @copydoc DeviceTensor::assign()
     */
    DeviceTensor<T, rank> &assign(Tensor<T, rank> const &other);

    /**
     * @brief Initialize the tensor, copying properties and data from another to this one.
     *
     * @param other The tensor to copy and to use as a template for the dimensions and other things.
     * @param mode The storage mode of the tensor.
     *
     * @return A reference to the calling tensor.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor<T, rank> &init(DeviceTensor<T, rank> const &other, einsums::detail::HostToDeviceMode mode = einsums::detail::UNKNOWN);

    /**
     * @copydoc DeviceTensor::assign()
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    DeviceTensor<T, rank> &assign(DeviceTensor<TOther, rank> const &other);

    /**
     * @copydoc DeviceTensor::assign()
     */
    template <typename TOther>
    DeviceTensor<T, rank> &assign(DeviceTensorView<TOther, rank> const &other);

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return The value at that index.
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    auto operator()(MultiIndex &&...index) const -> T;

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return A wrapper around the value at that index that manages data transfer to the GPU.
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
        requires(std::is_integral_v<std::remove_cvref_t<MultiIndex>> && ... && true)
    auto operator()(MultiIndex &&...index) -> HostDevReference<T>;

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam int_type Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return The value at that index.
     *
     * @versionadded{1.0.0}
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) const -> T {
        return std::apply(*this, index);
    }

    /**
     * @brief Subscripts into the tensor.
     *
     * This version works when all elements are explicit values into the tensor.
     * It does not work with the All or Range tags.
     *
     * @tparam MultiIndex Datatype of the indices. Must be castable to std::int64_t.
     * @param index The explicit desired index into the tensor. Elements must be castable to std::int64_t.
     * @return A wrapper around the value at that index that manages data transfer to the GPU.
     *
     * @versionadded{1.0.0}
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) -> HostDevReference<T> {
        return std::apply(*this, index);
    }

    // WARNING: Chances are this function will not work if you mix All{}, Range{} and explicit indexes.
    /**
     * @brief Subscripts into the tensor and creates a view.
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
        requires AtLeastOneOfType<AllT, MultiIndex...>
    auto operator()(MultiIndex... index)
        -> DeviceTensorView<T, count_of_type<einsums::AllT, MultiIndex...>() + count_of_type<einsums::Range, MultiIndex...>()>;

    /**
     * @brief Subscripts into the tensor and creates a view.
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
        requires NumOfType<einsums::Range, rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> DeviceTensorView<T, rank>;

    /**
     * @brief Copy data from one tensor to another.
     *
     * @versionadded{1.0.0}
     */
    auto operator=(DeviceTensor<T, rank> const &other) -> DeviceTensor<T, rank> &;

    /**
     * @brief Copy data from one tensor to another, and convert types.
     *
     * @versionadded{1.0.0}
     */
    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(DeviceTensor<TOther, rank> const &other) -> DeviceTensor<T, rank> &;

    /**
     * @brief Copy data from a tensor view into a tensor.
     *
     * @versionadded{1.0.0}
     */
    template <typename TOther>
    auto operator=(DeviceTensorView<TOther, rank> const &other) -> DeviceTensor<T, rank> &;

    /**
     * @brief Copy data from one tensor to another.
     *
     * @versionadded{1.0.0}
     */
    auto operator=(Tensor<T, rank> const &other) -> DeviceTensor<T, rank> &;

    /**
     * Fill a tensor with a value.
     *
     * @versionadded{1.0.0}
     */
    auto operator=(T const &fill_value) -> DeviceTensor<T, rank> &;

    /**
     * @brief Operate and assign every element with a scalar.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor<T, rank> &add_assign(T const &other);

    /**
     * @copydoc DeviceTensor::add_assign(T const &)
     */
    DeviceTensor<T, rank> &mult_assign(T const &other);

    /**
     * @copydoc DeviceTensor::add_assign(T const &)
     */
    DeviceTensor<T, rank> &sub_assign(T const &other);

    /**
     * @copydoc DeviceTensor::add_assign(T const &)
     */
    DeviceTensor<T, rank> &div_assign(T const &other);

    /**
     * @copydoc DeviceTensor::add_assign(T const &)
     */
    DeviceTensor<T, rank> &operator*=(T const &other) { return this->mult_assign(other); }

    /**
     * @copydoc DeviceTensor::add_assign(T const &)
     */
    DeviceTensor<T, rank> &operator+=(T const &other) { return this->add_assign(other); }

    /**
     * @copydoc DeviceTensor::add_assign(T const &)
     */
    DeviceTensor<T, rank> &operator-=(T const &other) { return this->sub_assign(other); }

    /**
     * @copydoc DeviceTensor::add_assign(T const &)
     */
    DeviceTensor<T, rank> &operator/=(T const &other) { return this->div_assign(other); }

    /**
     * @brief Operate and assign two tensors element-wise.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor<T, rank> &add_assign(DeviceTensor<T, rank> const &other);

    /**
     * @copydoc DeviceTensor::add_assign(DeviceTensor<T, rank> const &)
     */
    DeviceTensor<T, rank> &mult_assign(DeviceTensor<T, rank> const &other);

    /**
     * @copydoc DeviceTensor::add_assign(DeviceTensor<T, rank> const &)
     */
    DeviceTensor<T, rank> &sub_assign(DeviceTensor<T, rank> const &other);

    /**
     * @copydoc DeviceTensor::add_assign(DeviceTensor<T, rank> const &)
     */
    DeviceTensor<T, rank> &div_assign(DeviceTensor<T, rank> const &other);

    /**
     * @copydoc DeviceTensor::add_assign(DeviceTensor<T, rank> const &)
     */
    DeviceTensor<T, rank> &operator*=(DeviceTensor<T, rank> const &other) { return this->mult_assign(other); }

    /**
     * @copydoc DeviceTensor::add_assign(DeviceTensor<T, rank> const &)
     */
    DeviceTensor<T, rank> &operator+=(DeviceTensor<T, rank> const &other) { return this->add_assign(other); }

    /**
     * @copydoc DeviceTensor::add_assign(DeviceTensor<T, rank> const &)
     */
    DeviceTensor<T, rank> &operator-=(DeviceTensor<T, rank> const &other) { return this->sub_assign(other); }

    /**
     * @copydoc DeviceTensor::add_assign(DeviceTensor<T, rank> const &)
     */
    DeviceTensor<T, rank> &operator/=(DeviceTensor<T, rank> const &other) { return this->div_assign(other); }

    /**
     * Create a view of the current tensor.
     *
     * @versionadded{1.0.0}
     */
    operator DeviceTensorView<T, rank>();

    /**
     * Create a view of the current tensor.
     *
     * @versionadded{1.0.0}
     */
    operator DeviceTensorView<T, rank> const() const;

    /**
     * @brief Get the dimension for the given rank.
     *
     * @versionadded{1.0.0}
     */
    size_t dim(int d) const {
        // Add support for negative indices.
        if (d < 0)
            d += rank;
        return _dims[d];
    }

    /**
     * @brief Get all the dimensions.
     *
     * @versionadded{1.0.0}
     */
    Dim<rank> dims() const { return _dims; }

    /**
     * @brief Get the dimension list that is visible to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *gpu_dims() { return _gpu_dims; }

    /**
     * @brief Get the dimension list that is visible to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t const *gpu_dims() const { return _gpu_dims; }

    /**
     * @brief Get the name of the tensor.
     *
     * @versionadded{1.0.0}
     */
    std::string const &name() const { return _name; }

    /**
     * @brief Set the name of the tensor.
     *
     * @versionadded{1.0.0}
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * @brief Get the stride of the given rank.
     *
     * @versionadded{1.0.0}
     */
    size_t stride(int d) const noexcept {
        if (d < 0)
            d += rank;
        return _strides[d];
    }

    /**
     * @brief Get all the strides.
     *
     * @versionadded{1.0.0}
     */
    Stride<rank> strides() const noexcept { return _strides; }

    /**
     * @brief Get the stride list that is visible to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *gpu_strides() { return _gpu_strides; }

    /**
     * @brief Get the stride list that is visible to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t const *gpu_strides() const { return _gpu_strides; }

    /**
     * Convert to a rank 1 tensor view.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView<T, 1> to_rank_1_view() const {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return DeviceTensorView<T, 1>{*this, dim};
    }

    /**
     * @brief Returns the linear size of the tensor.
     *
     * @versionadded{1.0.0}
     */
    size_t size() const { return _dims[0] * _strides[0]; }

    /**
     * @brief Whether this object is the full view.
     *
     * @versionadded{1.0.0}
     */
    bool full_view_of_underlying() const noexcept { return true; }

    /**
     * Return the mode of the tensor.
     *
     * @versionadded{1.0.0}
     */
    detail::HostToDeviceMode mode() const { return _mode; }

    /**********************************************
     * Interface between device and host tensors. *
     **********************************************/

    /**
     * @brief Copy a host tensor to the device.
     *
     * @versionadded{1.0.0}
     */
    explicit DeviceTensor(Tensor<T, rank> const &, detail::HostToDeviceMode mode = detail::MAPPED);

    /**
     * @brief Copy a device tensor to the host.
     *
     * @versionadded{1.0.0}
     */
    operator einsums::Tensor<T, rank>() const;

  private:
    /**
     * @property _name
     *
     * @brief The name of the tensor.
     *
     * @versionadded{1.0.0}
     */
    std::string _name{"(Unnamed)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the tensor.
     *
     * @versionadded{1.0.0}
     */
    einsums::Dim<rank> _dims;

    /**
     * @property _gpu_dims
     *
     * @brief The dimensions of the tensor made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *_gpu_dims{nullptr};

    /**
     * @property _strides
     *
     * @brief The strides of the tensor.
     *
     * @versionadded{1.0.0}
     */
    einsums::Stride<rank> _strides;

    /**
     * @property _gpu_strides
     *
     * @brief The strides of the tensor made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *_gpu_strides{nullptr};

    /**
     * @property _data
     *
     * @brief A device pointer to the data on the device.
     *
     * @versionadded{1.0.0}
     */
    dev_datatype *_data{nullptr};

    /**
     * @property _host_data
     *
     * @brief If the tensor is mapped or pinned, this is the data on the host.
     *
     * @versionadded{1.0.0}
     */
    host_datatype *_host_data{nullptr};

    /**
     * @property _mode
     *
     * @brief The storage mode of the tensor.
     *
     * @versionadded{1.0.0}
     */
    detail::HostToDeviceMode _mode{detail::UNKNOWN};

#ifndef DOXYGEN
    friend struct DeviceTensorView<T, rank>;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensorView;

    template <typename TOther, size_t RankOther>
    friend struct einsums::DeviceTensor;
#endif
};

/**
 * @struct DeviceTensor<T, 0>
 *
 * Implementation for a zero-rank tensor.
 *
 * @versionadded{1.0.0}
 */
template <typename T>
struct DeviceTensor<T, 0>
    : public einsums::tensor_base::DeviceTensorBase, design_pats::Lockable<std::recursive_mutex>, tensor_base::AlgebraOptimizedTensor {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The type of data used by the GPU.
     *
     * For real data types, this is the same as on the host. For complex data types, the appropriate
     * HIP data type needs to be used.
     *
     * @versionadded{1.0.0}
     */
    using dev_datatype = typename einsums::tensor_base::DeviceTypedTensor<T>::dev_datatype;

    /**
     * @typedef host_datatype
     *
     * @brief The type of data used by the CPU.
     *
     * This is the same as the ValueType.
     *
     * @versionadded{1.0.0}
     */
    using host_datatype = typename einsums::tensor_base::DeviceTypedTensor<T>::host_datatype;

    /**
     * @typedef ValueType
     *
     * @brief The type of data stored by the tensor.
     *
     * @versionadded{1.0.0}
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     *
     * @versionadded{1.0.0}
     */
    constexpr static size_t Rank = 0;

    /**
     * @brief Construct a new tensor on the GPU.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensor() : _mode(detail::DEV_ONLY) { hip_catch(hipMalloc((void **)&_data, sizeof(T))); }

    /**
     * @brief Copy construct a new GPU tensor.
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
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

    /**
     * @brief Get the pointer to the data stored on the GPU.
     *
     * @return The pointer to the data on the GPU.
     *
     * @versionadded{1.0.0}
     */
    auto gpu_data() -> dev_datatype * { return _data; }

    /**
     * @copydoc DeviceTensor::gpu_data
     */
    [[nodiscard]] auto gpu_data() const -> dev_datatype const * { return _data; }

    /**
     * @brief Get the pointer to the data available to the CPU.
     *
     * @return The pointer to the data on the CPU.
     *
     * @versionadded{1.0.0}
     */
    auto data() -> host_datatype * { return _host_data; }

    /**
     * @copydoc DeviceTensor::data()
     */
    [[nodiscard]] auto data() const -> host_datatype const * { return _host_data; }

    /**
     * @brief Copy the data from one zero-rank tensor to another.
     *
     * @param other The value to copy.
     *
     * @return A reference to the current tensor.
     *
     * @versionadded{1.0.0}
     */
    auto operator=(DeviceTensor<T, 0> const &other) -> DeviceTensor<T, 0> & {
        hip_catch(hipMemcpyAsync((void *)_data, (void const *)other.gpu_data(), sizeof(T), hipMemcpyDeviceToDevice, gpu::get_stream()));
        gpu::stream_wait();
        return *this;
    }

    /**
     * @brief Copy a scalar to the tensor.
     *
     * @param other The value to copy.
     *
     * @return A reference to the current tensor.
     *
     * @versionadded{1.0.0}
     */
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

    /**
     * @brief Operate and assign with a value.
     *
     * @param other The value to operate with.
     *
     * @return A reference to the current tensor.
     *
     * @versionadded{1.0.0}
     */
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

    /**
     * @copydoc DeviceTensor::operator+=()
     */
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

    /**
     * @copydoc DeviceTensor::operator+=()
     */
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

    /**
     * @copydoc DeviceTensor::operator+=()
     */
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

    /**
     * @brief Get the scalar value of the tensor.
     *
     * @return The scalar value stored in the tensor.
     *
     * @versionadded{1.0.0}
     */
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

    /**
     * @brief Get the name of the tensor.
     *
     * @return The name of the tensor.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] auto name() const -> std::string const & { return _name; }

    /**
     * @brief Set the name of the tensor.
     *
     * @param name The new name of the tensor.
     *
     * @versionadded{1.0.0}
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * @brief Get the dimension along a given axis. The argument is ignored.
     *
     * @param axis Ignored. The dimension is always 1.
     *
     * @return The value 1.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] auto dim(int axis) const -> size_t { return 1; }

    /**
     * @brief Get the dimensions of the tensor. This is always an empty array.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] auto dims() const -> Dim<0> { return Dim<0>{}; }

    /**
     * @brief Check if the tensor contains contiguous data.
     *
     * @return True, since this tensor stores contiguous data.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }

    /**
     * @brief Get the stride of the given rank.
     *
     * @versionadded{1.0.0}
     */
    [[nodiscard]] auto stride(int d) const noexcept -> size_t { return 0; }

    /**
     * @brief Get all the strides.
     *
     * @versionadded{1.0.0}
     */
    auto strides() const noexcept -> Stride<0> { return Stride<0>{}; }

    /**********************************************
     * Interface between device and host tensors. *
     **********************************************/

    /**
     * @brief Copy a host tensor to the device.
     *
     * @versionadded{1.0.0}
     */
    explicit DeviceTensor(Tensor<T, 0> const &other, detail::HostToDeviceMode mode = detail::MAPPED) : _mode{mode} {
        switch (mode) {
        case detail::MAPPED:
            this->_host_data = new T();
            *_host_data      = other.operator T();
            hip_catch(hipHostRegister((void *)this->_host_data, sizeof(T), hipHostRegisterDefault));
            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            break;
        case detail::PINNED:
            hip_catch(hipHostMalloc((void **)&(this->_host_data), sizeof(T), 0));
            hip_catch(hipHostGetDevicePointer((void **)&(this->_data), (void *)this->_host_data, 0));
            *_host_data = other.operator T();
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
     *
     * @versionadded{1.0.0}
     */
    explicit operator einsums::Tensor<T, 0>() {
        einsums::Tensor<T, 0> out(std::move(_name));

        out = (T) * this;
        return out;
    }

  private:
    /**
     * @property _name
     *
     * @brief The name of the tensor.
     *
     * @versionadded{1.0.0}
     */
    std::string _name{"(Unnamed)"};

    /**
     * @property _data
     *
     * @brief A device pointer to the data on the device.
     *
     * @versionadded{1.0.0}
     */
    dev_datatype *_data{nullptr};

    /**
     * @property _host_data
     *
     * @brief If the tensor is mapped or pinned, this is the data on the host.
     *
     * @versionadded{1.0.0}
     */
    host_datatype *_host_data{nullptr};

    /**
     * @property _mode
     *
     * @brief The storage mode of the tensor.
     *
     * @versionadded{1.0.0}
     */
    detail::HostToDeviceMode _mode;

    template <typename TOther, size_t RankOther>
    friend struct DeviceTensorView;

    template <typename TOther, size_t RankOther>
    friend struct einsums::DeviceTensor;
};

/**
 * @struct DeviceTensorView
 *
 * @brief Holds a view of a DeviceTensor.
 *
 * This class allows for a view based at an offset and with different dimensions to be mapped
 * back onto a DeviceTensor.
 *
 * @versionadded{1.0.0}
 */
template <typename T, size_t rank>
struct DeviceTensorView : public einsums::tensor_base::DeviceTensorBase,
                          design_pats::Lockable<std::recursive_mutex>,
                          tensor_base::AlgebraOptimizedTensor {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The type of data used by the GPU.
     *
     * For real data types, this is the same as on the host. For complex data types, the appropriate
     * HIP data type needs to be used.
     *
     * @versionadded{1.0.0}
     */
    using dev_datatype = typename einsums::tensor_base::DeviceTypedTensor<T>::dev_datatype;

    /**
     * @typedef host_datatype
     *
     * @brief The type of data used by the CPU.
     *
     * This is the same as the ValueType.
     *
     * @versionadded{1.0.0}
     */
    using host_datatype = typename einsums::tensor_base::DeviceTypedTensor<T>::host_datatype;

    /**
     * @typedef ValueType
     *
     * @brief The type of data stored by the tensor.
     *
     * @versionadded{1.0.0}
     */
    using ValueType = T;

    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     *
     * @versionadded{1.0.0}
     */
    constexpr static size_t Rank = rank;

    /**
     * @typedef underlying_type
     *
     * @brief The type of tensor that this view views.
     *
     * @versionadded{1.0.0}
     */
    using underlying_type = einsums::DeviceTensor<T, rank>;

    /**
     * Deleted default constructor.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView() = delete;

    /**
     * @brief Copy constructor.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView(DeviceTensorView const &);

    /**
     * @brief Destructor.
     *
     * @versionadded{1.0.0}
     */
    ~DeviceTensorView();

    // std::enable_if doesn't work with constructors.  So we explicitly create individual
    // constructors for the types of tensors we support (Tensor and TensorView).  The
    // call to common_initialization is able to perform an enable_if check.
    /**
     * @brief Create a tensor view around the given tensor.
     *
     * @versionadded{1.0.0}
     */
    template <size_t OtherRank, typename... Args>
    DeviceTensorView(einsums::DeviceTensor<T, OtherRank> const &other, Dim<rank> const &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(other, std::forward<Args>(args)...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     *
     * @versionadded{1.0.0}
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensorView<T, OtherRank> &other, Dim<rank> const &dim, Args &&...args)
        : _name{other.name()}, _dims{dim} {
        common_initialization(other, std::forward<Args>(args)...);
    }

    /**
     * @brief Create a tensor view around the given tensor.
     *
     * @versionadded{1.0.0}
     */
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensorView<T, OtherRank> const &other, Dim<rank> const &dim, Args &&...args)
        : _name{other.name()}, _dims{dim} {
        common_initialization(const_cast<DeviceTensorView<T, OtherRank> &>(other), std::forward<Args>(args)...);
    }

    /**
     * Create a device tensor view that maps an in-core tensor to the GPU.
     *
     * @versionadded{1.0.0}
     */
    template <CoreBasicTensorConcept TensorType>
        requires(TensorType::Rank == rank)
    explicit DeviceTensorView(TensorType &core_tensor) {
        _name                    = core_tensor.name();
        _dims                    = core_tensor.dims();
        _strides                 = core_tensor.strides();
        _full_view_of_underlying = true;
        _host_data               = core_tensor.data();

        _free_dev_data = true;

        dims_to_strides(_dims, _strides);

        hip_catch(hipMalloc((void **)&(_gpu_dims), 3 * sizeof(size_t) * rank));
        this->_gpu_strides       = this->_gpu_dims + rank;
        this->_gpu_index_strides = this->_gpu_dims + 2 * rank;

        hip_catch(hipMemcpy((void *)this->_gpu_dims, (void const *)this->_dims.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)this->_gpu_strides, (void const *)this->_strides.data(), sizeof(size_t) * rank, hipMemcpyHostToDevice));
        hip_catch(hipMemcpy((void *)this->_gpu_index_strides, (void const *)this->_index_strides.data(), sizeof(size_t) * rank,
                            hipMemcpyHostToDevice));
        gpu::device_synchronize();

        hip_catch(hipHostRegister(_host_data, _strides[0] * _dims[0] * sizeof(T), hipHostRegisterDefault));
        hip_catch(hipHostGetDevicePointer((void **)&_data, _host_data, 0));
    }

    /**
     * @brief Copy the data from a pointer to memory into this view.
     *
     * This is an advanced method. Only use if you know what you are doing.
     *
     * @param other A pointer to the data to copy. The data needs to have the same size as the view.
     *
     * @return A reference to this tensor.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView<T, rank> &assign(T const *other);

    /**
     * @brief Copy the data from some tensor into this view.
     *
     * @param other The tensor to copy.
     *
     * @return A reference to this tensor.
     *
     * @versionadded{1.0.0}
     */
    template <template <typename, size_t> typename AType>
        requires DeviceRankTensor<AType<T, rank>, rank, T>
    DeviceTensorView<T, rank> &assign(AType<T, rank> const &other);

    /**
     * @brief Set every value in the tensor to the one passed to this method.
     *
     * @param value The value to fill the tensor with.
     *
     * @versionadded{1.0.0}
     */
    void set_all(T const &value);

    /**
     * @brief Set every value in the tensor to zero.
     *
     * @versionadded{1.0.0}
     */
    void zero() { set_all(T{0.0}); }

    /**
     * @brief Copy as much data as is needed from the host pointer to the device.
     *
     * @param other A pointer to the data to copy.
     *
     * @return A reference to the current tensor.
     *
     * @versionadded{1.0.0}
     */
    auto operator=(T const *other) -> DeviceTensorView &;

    /**
     * @brief Copy data from another tensor.
     *
     * @param other The tensor to copy.
     *
     * @return A reference to the current tensor.
     *
     * @versionadded{1.0.0}
     */
    template <template <typename, size_t> typename AType>
        requires DeviceRankTensor<AType<T, rank>, rank, T>
    auto operator=(AType<T, rank> const &other) -> DeviceTensorView &;

    /**
     * @brief Copy data from another tensor.
     *
     * @param other The other tensor to copy.
     *
     * @return A reference to the current tensor.
     *
     * @versionadded{1.0.0}
     */
    auto operator=(DeviceTensorView<T, rank> const &other) -> DeviceTensorView &;

    /**
     * @brief Fill the view with a value.
     *
     * @param fill_value The value to fill the view with.
     *
     * @return A reference to the current tensor.
     *
     * @versionadded{1.0.0}
     */
    auto operator=(T const &fill_value) -> DeviceTensorView & {
        this->set_all(fill_value);
        return *this;
    }

    /**
     * @brief Operate each element in the view with a scalar.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView<T, rank> &mult_assign(T const &value);

    /**
     * @copydoc DeviceTensorView::mult_assign()
     */
    DeviceTensorView<T, rank> &div_assign(T const &value);

    /**
     * @copydoc DeviceTensorView::mult_assign()
     */
    DeviceTensorView<T, rank> &add_assign(T const &value);

    /**
     * @copydoc DeviceTensorView::mult_assign()
     */
    DeviceTensorView<T, rank> &sub_assign(T const &value);

    /**
     * @brief Operate each element in the view with a scalar.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView &operator*=(T const &value) { return this->mult_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView &operator/=(T const &value) { return this->div_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     *
     * @versionadded{1.0.0}
     */
    DeviceTensorView &operator+=(T const &value) { return this->add_assign(value); }

    /**
     * @brief Operate each element in the view with a scalar.
     *
     * @versionadded{1.0.0}
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
     *
     * @versionadded{1.0.0}
     */
    host_datatype *data() { return _host_data; }

    /**
     * @brief Returns a pointer to the host-readable data.
     *
     * Returns a pointer to the data as available to the host. If the tensor is in DEV_ONLY mode,
     * then this will return a null pointer. Otherwise, the pointer should be useable, but the data
     * may be outdated.
     *
     * @return const T* An immutable pointer to the data.
     *
     * @versionadded{1.0.0}
     */
    host_datatype const *data() const { return _host_data; }

    /**
     * @brief Get a pointer to the GPU data.
     *
     * @versionadded{1.0.0}
     */
    dev_datatype *gpu_data() { return _data; }

    /**
     * @brief Get a pointer to the GPU data.
     *
     * @versionadded{1.0.0}
     */
    dev_datatype const *gpu_data() const { return const_cast<dev_datatype const *>(_data); }

    /**
     * @brief Get a pointer to an element in the view.
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
    dev_datatype *gpu_data(MultiIndex... index);

    /**
     * @brief Get a const pointer to an element in the view.
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
    dev_datatype const *gpu_data(MultiIndex... index) const;

    /**
     * @brief Get a pointer to an element in the view.
     *
     * @versionadded{1.0.0}
     */
    dev_datatype *gpu_data_array(std::array<size_t, rank> const &index_list);

    /**
     * @brief Get a const pointer to an element in the view.
     *
     * @versionadded{1.0.0}
     */
    dev_datatype const *gpu_data_array(std::array<size_t, rank> const &index_list) const;

    /**
     * @brief Get a value from the view.
     *
     * @param index The index to pull from.
     *
     * @return The value at the requested index.
     *
     * @versionadded{1.0.0}
     */
    template <typename... MultiIndex>
    auto operator()(MultiIndex &&...index) const -> T;

    /**
     * @brief Get a value from the view.
     *
     * @param index The index to pull from.
     *
     * @return The value at the requested index.
     *
     * @versionadded{1.0.0}
     */
    template <typename int_type>
        requires(std::is_integral_v<int_type>)
    auto operator()(std::array<int_type, Rank> const &index) const -> T {
        return std::apply(*this, index);
    }

    /**
     * @brief Get the dimension of the given rank.
     *
     * @versionadded{1.0.0}
     */
    size_t dim(int d) const {
        if (d < 0)
            d += rank;
        return _dims[d];
    }

    /**
     * @brief Get the dimensions of the view.
     *
     * @versionadded{1.0.0}
     */
    Dim<rank> dims() const { return _dims; }

    /**
     * @brief Get the dimensions of the view made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *gpu_dims() { return _gpu_dims; }

    /**
     * @brief Get the dimensions of the view made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t const *gpu_dims() const { return _gpu_dims; }

    /**
     * @brief Get the name of the view.
     *
     * @versionadded{1.0.0}
     */
    std::string const &name() const { return _name; }

    /**
     * @brief Set the name of the view.
     *
     * @versionadded{1.0.0}
     */
    void set_name(std::string const &name) { _name = name; }

    /**
     * @brief Get the stride of the given rank.
     *
     * @versionadded{1.0.0}
     */
    size_t stride(int d) const noexcept {
        if (d < 0)
            d += rank;
        return _strides[d];
    }

    /**
     * @brief Get the strides of the view.
     *
     * @versionadded{1.0.0}
     */
    Stride<rank> strides() const noexcept { return _strides; }

    /**
     * @brief Get the strides of the view made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *gpu_strides() { return _gpu_strides; }

    /**
     * @brief Get the strides of the view made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t const *gpu_strides() const { return _gpu_strides; }

    /**
     * @brief Convert the view to a one-dimensional array.
     *
     * @versionadded{1.0.0}
     */
    auto to_rank_1_view() const -> DeviceTensorView<T, 1>;

    /**
     * @brief Whether the view wraps all the data.
     *
     * @versionadded{1.0.0}
     */
    bool full_view_of_underlying() const noexcept { return _full_view_of_underlying; }

    /**
     * @brief Get the size of the view.
     *
     * @versionadded{1.0.0}
     */
    size_t size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + rank, 1, std::multiplies<>{}); }

    /**
     * @brief Copy the data into an in-core tensor.
     *
     * @versionadded{1.0.0}
     */
    operator einsums::Tensor<T, rank>() const {
        einsums::DeviceTensor<T, rank> temp(*this);

        return (einsums::Tensor<T, rank>)temp;
    }

  private:
    /**
     * @property _name
     *
     * @brief The name of the view.
     *
     * @versionadded{1.0.0}
     */
    std::string _name{"(Unnamed View)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the view.
     *
     * @versionadded{1.0.0}
     */
    einsums::Dim<rank> _dims;

    /**
     * @property _gpu_dims
     *
     * @brief The dimensions of the view made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *_gpu_dims{nullptr};

    /**
     * @property _strides
     *
     * @brief The strides of the view.
     *
     * @versionadded{1.0.0}
     */
    einsums::Stride<rank> _strides, _index_strides;

    /**
     * @property _gpu_strides
     *
     * @brief The strides of the view made available to the GPU.
     *
     * @versionadded{1.0.0}
     */
    size_t *_gpu_strides{nullptr}, *_gpu_index_strides;
    // Offsets<rank> _offsets;

    /**
     * @property _full_view_of_underlying
     *
     * @brief Whether the view captures all of the data.
     *
     * @versionadded{1.0.0}
     */
    bool _full_view_of_underlying{false};

    /**
     * @property _free_dev_data
     *
     * @brief Flag telling whether this view needs to free its data upon deletion.
     *
     * This is normally false, but when a device view is made of a core tensor, then this
     * will be true, so that the buffers for the core tensor are freed.
     *
     * @versionadded{1.0.0}
     */
    bool _free_dev_data{false};

    /**
     * @property _data
     *
     * @brief A pointer to the GPU data.
     *
     * @versionadded{1.0.0}
     */
    dev_datatype *_data{nullptr};

    /**
     * @property _host_data
     *
     * @brief A pointer to the host data.
     *
     * @versionadded{1.0.0}
     */
    host_datatype *_host_data{nullptr};
    /**
     * @brief Method for initializing the view.
     *
     * @versionadded{1.0.0}
     */
#ifndef DOXYGEN
    template <template <typename, size_t> typename TensorType, size_t OtherRank, typename... Args>
        requires(TRTensorConcept<TensorType<T, OtherRank>, OtherRank, T>)
    void common_initialization(TensorType<T, OtherRank> const &other, Args &&...args);

    template <typename OtherT, size_t OtherRank>
    friend struct DeviceTensorView;

    template <typename OtherT, size_t OtherRank>
    friend struct ::einsums::DeviceTensor;
#endif
};

#if !defined(DOXYGEN) && defined(__cpp_deduction_guides)
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

#include <Einsums/Tensor/Backends/DeviceTensor.hpp>
#include <Einsums/Tensor/Backends/DeviceTensorView.hpp>

#ifndef DOXYGEN
namespace einsums {
TENSOR_EXPORT_RANK(DeviceTensor, 0)
TENSOR_EXPORT(DeviceTensor)
TENSOR_EXPORT(DeviceTensorView)
} // namespace einsums
#endif
#endif