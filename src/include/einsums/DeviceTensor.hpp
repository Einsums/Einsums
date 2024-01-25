#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/Tensor.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::gpu)

template <typename T, size_t Rank>
struct DeviceTensorView;

template <typename T, size_t Rank>
struct DeviceTensor;

namespace detail {

/**
 * @enum HostToDeviceMode
 *
 * @brief Enum that specifies how device tensors store data and make it available to the GPU.
 */
enum HostToDeviceMode { DEV_ONLY, MAPPED, PINNED };

/**
 * @struct IsDeviceRankTensor
 *
 * @brief Struct for specifying that a tensor is device compatible.
 */
template <typename D, size_t Rank, typename T>
struct IsDeviceRankTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, gpu::DeviceTensor<T, Rank>> ||
                                                      std::is_same_v<std::decay_t<D>, gpu::DeviceTensorView<T, Rank>>> {};

/**
 * @property IsDeviceRankTensorV
 *
 * @brief True if the tensor is device compatible.
 */
template <typename D, size_t Rank, typename T>
inline constexpr bool IsDeviceRankTensorV = IsDeviceRankTensor<D, Rank, T>::value;

/**
 * @concept DeviceRankTensor
 *
 * @brief Concept for testing whether a tensor parameter is available to the GPU.
 */
template <typename Input, size_t Rank, typename DataType = double>
concept DeviceRankTensor = detail::IsDeviceRankTensorV<Input, Rank, DataType>;

/**
 * Turns a single sentinel value into an index combination.
 *
 * @param index The sentinel value.
 * @param dims The dimensions of the tensor along each axis.
 * @param out The output array.
 */
template <typename T, size_t Rank>
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
template <typename T, size_t Rank>
__host__ __device__ size_t combination_to_index(const std::array<int, Rank> &inds, const einsums::Dim<Rank> &dims,
                                                const einsums::Stride<Rank> &strides);

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
    T   *_ptr;

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
    HostDevReference() : _ptr(nullptr), is_on_host(true) {}

    /**
     * Construct a reference wrapping the specified pointer.
     */
    HostDevReference(T *ptr, bool is_host) : _ptr(*ptr), is_on_host(is_host) {}

    /**
     * Delete the reference. Because the data is managed by something else, don't acutally free the pointer.
     */
    ~HostDevReference() {
        _ptr = nullptr;
    }

    /**
     * Get the value of the reference.
     */
    T get() {
        if (is_on_host) {
            return *_ptr;
        } else {
            T out;
            hip_catch(hipMemcpy((void *)&out, (const void *)_ptr, sizeof(T), hipMemcpyDeviceToHost));
            return out;
        }
    }

    /**
     * Copy some data to the reference.
     */
    HostDevReference<T> &operator=(const T &other) {
        if (is_on_host) {
            *_ptr = other;
        } else {
            hip_catch(hipMemcpy((void *)_ptr, (const void *)&other, sizeof(T), hipMemcpyHostToDevice));
        }
    }

    /**
     * Copy some data to the reference.
     */
    HostDevReference<T> &operator=(const HostDevReference<T> &other) {
        if (is_on_host) {
            *_ptr = other.get();
        } else {
            hip_catch(hipMemcpy((void *)_ptr, (const void *)&(other.get()), sizeof(T), hipMemcpyHostToDevice));
        }
    }

    /**
     * Get the address handled by the reference.
     */
    T *operator &() {
        return _ptr;
    }
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
    using dev_datatype  = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
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
    std::string             _name{"(Unnamed)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the tensor.
     */
    einsums::Dim<Rank>    _dims;

    /**
     * @property _gpu_dims
     *
     * @brief The dimensions of the tensor made available to the GPU.
     */
    size_t                 *_gpu_dims;

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
    size_t                 *_gpu_strides;

    /**
     * @property _data
     *
     * @brief A device pointer to the data on the device.
     */
    device_ptr dev_datatype *_data;

    /**
     * @property _host_data
     *
     * @brief If the tensor is mapped or pinned, this is the data on the host.
     */
    host_ptr host_datatype  *_host_data;

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
    DeviceTensor(Dim<Rank> dims, detail::HostToDeviceMode mode);

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
     * Fill a tensor with a value.
     */
    auto operator=(const T &fill_value) -> DeviceTensor<T, Rank> &;

    /**
     * @brief Operate and assign every element with a scalar.
     */
    DeviceTensor<T, Rank> &operator*=(const T &other);
    DeviceTensor<T, Rank> &operator+=(const T &other);
    DeviceTensor<T, Rank> &operator-=(const T &other);
    DeviceTensor<T, Rank> &operator/=(const T &other);

    /**
     * @brief Operate and assign two tensors element-wise.
     */
    DeviceTensor<T, Rank> &operator*=(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &operator+=(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &operator-=(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &operator/=(const DeviceTensor<T, Rank> &other);

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
    device_ptr size_t *gpu_dims() { return _gpu_dims; }

    ALIAS_TEMPLATE_FUNCTION(shape, dims);

    /**
     * @brief Get the name of the tensor.
     */
    [[nodiscard]] auto name() const -> const std::string & { return _name; }

    /**
     * @brief Set the name of the tensor.
     */
    void               set_name(const std::string &name) { _name = name; }

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
    device_ptr size_t *gpu_strides() { return _gpu_strides; }

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
    explicit operator Tensor<T, Rank>();
};

template <typename T, size_t Rank>
struct DeviceTensorView : public ::einsums::detail::TensorBase<T, Rank> {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype  = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
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
    std::string           _name{"(Unnamed View)"};

    /**
     * @property _dims
     *
     * @brief The dimensions of the view.
     */
    einsums::Dim<Rank>    _dims;

    /**
     * @property _gpu_dims
     *
     * @brief The dimensions of the view made available to the GPU.
     */
    size_t               *_gpu_dims;

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
    size_t               *_gpu_strides;
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

    /**
     * @brief Copy as much data as is needed from the host pointer to the device.
     */
    auto operator=(const host_ptr T *other) -> DeviceTensorView &;

    /**
     * @brief Copy data from another tensor.
     */
    template <template <typename, size_t> typename AType>
        requires detail::DeviceRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(const AType<T, Rank> &other) -> DeviceTensorView &;

    /**
     * @brief Copy data from a tensor. 
     */
    template <template <typename, size_t> typename AType>
        requires detail::DeviceRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(const AType<T, Rank> &&other) -> DeviceTensorView &;

    /**
     * @brief Fill the view with a value.
     */
    auto operator=(const T &fill_value) -> DeviceTensorView &;

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator*=(const T &value);

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator/=(const T &value);

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator+=(const T &value);

    /**
     * @brief Operate each element in the view with a scalar.
     */
    DeviceTensorView &operator-=(const T &value);

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
    auto data_array(const std::array<size_t, Rank> &index_list) const -> device_ptr T *;

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
    device_ptr size_t *gpu_dims() const { return _gpu_dims; }

    /**
     * @brief Get the name of the view.
     */
    [[nodiscard]] auto name() const -> const std::string & { return _name; }

    /**
     * @brief Set the name of the view.
     */
    void               set_name(const std::string &name) { _name = name; }

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
    device_ptr size_t *gpu_strides() const { return _gpu_strides; }

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

END_EINSUMS_NAMESPACE_HPP(einsums::gpu)

#include "einsums/gpu/DeviceTensor.imp.hip"
#include "einsums/gpu/DeviceTensorView.imp.hip"