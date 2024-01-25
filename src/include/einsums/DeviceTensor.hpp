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

enum HostToDeviceMode { DEV_ONLY, MAPPED, PINNED };

template <typename D, size_t Rank, typename T>
struct IsDeviceRankTensor : public std::bool_constant<std::is_same_v<std::decay_t<D>, gpu::DeviceTensor<T, Rank>> ||
                                                      std::is_same_v<std::decay_t<D>, gpu::DeviceTensorView<T, Rank>>> {};

template <typename D, size_t Rank, typename T>
inline constexpr bool IsDeviceRankTensorV = IsDeviceRankTensor<D, Rank, T>::value;

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
 * Turns a combination of indices into a single sentinel value that can be used to index into an array.
 */
template <typename T, size_t Rank>
__host__ __device__ size_t combination_to_index(const std::array<int, Rank> &inds, const einsums::Dim<Rank> &dims,
                                                const einsums::Stride<Rank> &strides);

} // namespace detail

template <typename T>
class HostDevReference {
  private:
    T   *_ptr;
    bool is_on_host;

  public:
    HostDevReference() : _ptr(nullptr), is_on_host(true) {}

    HostDevReference(T *ptr, bool is_host) : _ptr(*ptr), is_on_host(is_host) {}

    ~HostDevReference() {
        if (is_on_host && _ptr != nullptr) {
            delete _ptr;
        }
    }

    T get() {
        if (is_on_host) {
            return *_ptr;
        } else {
            T out;
            hip_catch(hipMemcpy((void *)&out, (const void *)_ptr, sizeof(T), hipMemcpyDeviceToHost));
            return out;
        }
    }

    HostDevReference<T> &operator=(const T &other) {
        if (is_on_host) {
            *_ptr = other;
        } else {
            hip_catch(hipMemcpy((void *)_ptr, (const void *)&other, sizeof(T), hipMemcpyHostToDevice));
        }
    }

    HostDevReference<T> &operator=(const HostDevReference &other) {
        if (is_on_host) {
            *_ptr = other.get();
        } else {
            hip_catch(hipMemcpy((void *)_ptr, (const void *)&(other.get()), sizeof(T), hipMemcpyHostToDevice));
        }
    }
};

template <typename T, size_t Rank>
struct DeviceTensor : public ::einsums::detail::TensorBase<T, Rank> {
  public:
    using dev_datatype  = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;
    using host_datatype = T;

  private:
    std::string             _name{"(Unnamed)"};
    ::einsums::Dim<Rank>    _dims;
    size_t                 *_gpu_dims;
    ::einsums::Stride<Rank> _strides;
    size_t                 *_gpu_strides;

    device_ptr dev_datatype *_data;
    host_ptr host_datatype  *_host_data;
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
    template <typename... MultiIndex>
        requires requires { requires AtLeastOneOfType<AllT, MultiIndex...>; }
    auto operator()(MultiIndex... index)
        -> DeviceTensorView<T, count_of_type<AllT, MultiIndex...>() + count_of_type<Range, MultiIndex...>()>;

    template <typename... MultiIndex>
        requires NumOfType<Range, Rank, MultiIndex...>
    auto operator()(MultiIndex... index) const -> DeviceTensorView<T, Rank>;

    auto operator=(const DeviceTensor<T, Rank> &other) -> DeviceTensor<T, Rank> &;

    template <typename TOther>
        requires(!std::same_as<T, TOther>)
    auto operator=(const DeviceTensor<TOther, Rank> &other) -> DeviceTensor<T, Rank> &;

    template <typename TOther>
    auto operator=(const DeviceTensorView<TOther, Rank> &other) -> DeviceTensor<T, Rank> &;

    auto operator=(const T &fill_value) -> DeviceTensor<T, Rank> &;

    DeviceTensor<T, Rank> &operator*=(const T &other);
    DeviceTensor<T, Rank> &operator+=(const T &other);
    DeviceTensor<T, Rank> &operator-=(const T &other);
    DeviceTensor<T, Rank> &operator/=(const T &other);

    DeviceTensor<T, Rank> &operator*=(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &operator+=(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &operator-=(const DeviceTensor<T, Rank> &other);
    DeviceTensor<T, Rank> &operator/=(const DeviceTensor<T, Rank> &other);

    [[nodiscard]] auto dim(int d) const -> size_t {
        // Add support for negative indices.
        if (d < 0)
            d += Rank;
        return _dims[d];
    }

    auto dims() const -> Dim<Rank> { return _dims; }

    device_ptr size_t *gpu_dims() { return _gpu_dims; }

    ALIAS_TEMPLATE_FUNCTION(shape, dims);

    [[nodiscard]] auto name() const -> const std::string & { return _name; }
    void               set_name(const std::string &name) { _name = name; }

    [[nodiscard]] auto stride(int d) const noexcept -> size_t {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    auto strides() const noexcept -> const auto & { return _strides; }

    device_ptr size_t *gpu_strides() { return _gpu_strides; }

    auto to_rank_1_view() const -> DeviceTensorView<T, 1> {
        size_t size = _strides.size() == 0 ? 0 : _strides[0] * _dims[0];
        Dim<1> dim{size};

        return DeviceTensorView<T, 1>{*this, dim};
    }

    // Returns the linear size of the tensor
    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return true; }

    /**********************************************
     * Interface between device and host tensors. *
     **********************************************/

    /**
     * Copy a host tensor to the device.
     */
    explicit DeviceTensor(const Tensor<T, Rank> &, detail::HostToDeviceMode mode = detail::MAPPED);

    /**
     * Copy a device tensor to the host.
     */
    explicit operator Tensor<T, Rank>();
};

template <typename T, size_t Rank>
struct DeviceTensorView : public ::einsums::detail::TensorBase<T, Rank> {
  public:
    using dev_datatype  = std::conditional_t<std::is_same_v<T, std::complex<float>>, hipComplex,
                                            std::conditional_t<std::is_same_v<T, std::complex<double>>, hipDoubleComplex, T>>;
    using host_datatype = T;

  private:
    std::string           _name{"(Unnamed View)"};
    einsums::Dim<Rank>    _dims;
    size_t               *_gpu_dims;
    einsums::Stride<Rank> _strides;
    size_t               *_gpu_strides;
    // Offsets<Rank> _offsets;

    bool _full_view_of_underlying{false};

    dev_datatype *_data;

  public:
    DeviceTensorView() = delete;
    DeviceTensorView(const DeviceTensorView &);
    ~DeviceTensorView();

    // std::enable_if doesn't work with constructors.  So we explicitly create individual
    // constructors for the types of tensors we support (Tensor and TensorView).  The
    // call to common_initialization is able to perform an enable_if check.
    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(const DeviceTensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(const_cast<DeviceTensor<T, OtherRank> &>(other), args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args) : _name{other._name}, _dims{dim} {
        common_initialization(other, args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(DeviceTensorView<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(other, args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(const DeviceTensorView<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{other._name}, _dims{dim} {
        common_initialization(const_cast<DeviceTensorView<T, OtherRank> &>(other), args...);
    }

    template <size_t OtherRank, typename... Args>
    explicit DeviceTensorView(std::string name, DeviceTensor<T, OtherRank> &other, const Dim<Rank> &dim, Args &&...args)
        : _name{std::move(name)}, _dims{dim} {
        common_initialization(other, args...);
    }

    auto operator=(const host_ptr T *other) -> DeviceTensorView &;

    template <template <typename, size_t> typename AType>
        requires detail::DeviceRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(const AType<T, Rank> &other) -> DeviceTensorView &;

    template <template <typename, size_t> typename AType>
        requires detail::DeviceRankTensor<AType<T, Rank>, Rank, T>
    auto operator=(const AType<T, Rank> &&other) -> DeviceTensorView &;

    auto operator=(const T &fill_value) -> DeviceTensorView &;

    DeviceTensorView &operator*=(const T &value);
    DeviceTensorView &operator/=(const T &value);
    DeviceTensorView &operator+=(const T &value);
    DeviceTensorView &operator-=(const T &value);

    auto data() -> dev_datatype * { return _data; }
    auto data() const -> const dev_datatype * { return static_cast<const T *>(_data); }
    template <typename... MultiIndex>
    auto data(MultiIndex... index) const -> dev_datatype *;

    auto data_array(const std::array<size_t, Rank> &index_list) const -> device_ptr T *;

    template <typename... MultiIndex>
    auto operator()(MultiIndex... index) const -> T;

    [[nodiscard]] auto dim(int d) const -> size_t {
        if (d < 0)
            d += Rank;
        return _dims[d];
    }

    auto dims() const -> Dim<Rank> { return _dims; }

    device_ptr size_t *gpu_dims() const { return _gpu_dims; }

    [[nodiscard]] auto name() const -> const std::string & { return _name; }
    void               set_name(const std::string &name) { _name = name; }

    [[nodiscard]] auto stride(int d) const noexcept -> size_t {
        if (d < 0)
            d += Rank;
        return _strides[d];
    }

    auto strides() const noexcept -> const auto & { return _strides; }

    device_ptr size_t *gpu_strides() const { return _gpu_strides; }

    auto to_rank_1_view() const -> DeviceTensorView<T, 1>;

    [[nodiscard]] auto full_view_of_underlying() const noexcept -> bool { return _full_view_of_underlying; }

    [[nodiscard]] auto size() const { return std::accumulate(std::begin(_dims), std::begin(_dims) + Rank, 1, std::multiplies<>{}); }

  private:
    template <template <typename, size_t> typename TensorType, size_t OtherRank, typename... Args>
    auto common_initialization(TensorType<T, OtherRank> &other, Args &&...args)
        -> std::enable_if_t<std::is_base_of_v<::einsums::detail::TensorBase<T, OtherRank>, TensorType<T, OtherRank>>>;
};

END_EINSUMS_NAMESPACE_HPP(einsums::gpu)

#include "einsums/gpu/DeviceTensor.imp.hip"
#include "einsums/gpu/DeviceTensorView.imp.hip"