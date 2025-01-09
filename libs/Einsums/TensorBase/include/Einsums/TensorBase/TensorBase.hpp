//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/Config.hpp>

#include <complex>
#include <memory>
#include <mutex>

#if defined(EINSUMS_COMPUTE_CODE)
#    include <hip/hip_complex.h>
#endif

namespace einsums::tensor_base {

/**
 * @struct TensorBase
 *
 * @brief Base type for all tensors in Einsums.
 */
struct TensorBase {
  public:
    /**
     * Default constructor.
     */
    TensorBase() = default;

    /**
     * Default copy constructor.
     */
    TensorBase(TensorBase const &) = default;

    /**
     * Default destructor.
     */
    virtual ~TensorBase() = default;

    /**
     * Indicates that the tensor contains all elements.
     */
    virtual bool full_view_of_underlying() const { return true; }

    /**
     * Gets the name of the tensor.
     */
    virtual std::string const &name() const = 0;

    /**
     * Sets the name of the tensor.
     */
    virtual void set_name(std::string const &new_name) = 0;
};

/**
 * @struct TypedTensor
 *
 * @brief Represents a tensor that stores a data type.
 *
 * @tparam T The data type the tensor stores.
 */
template <typename T>
struct TypedTensor : virtual TensorBase {
    /**
     * @typedef ValueType
     *
     * @brief Gets the stored data type.
     */
    using ValueType = T;

    /**
     * Default constructor.
     */
    TypedTensor() = default;

    /**
     * Default copy constructor.
     */
    TypedTensor(TypedTensor const &) = default;

    /**
     * Default destructor.
     */
    virtual ~TypedTensor() = default;
};

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @struct DeviceTypedTensor
 *
 * Represents a tensor that stores different data types on the host and the device.
 * By default, if the host stores complex<float> or complex<double>, then it converts
 * it to hipComplex or hipDoubleComplex. The two types should have the same storage size.
 * @tparam HostT The host data type.
 * @tparam DevT The device type.
 */
template <typename HostT, typename DevT = void>
    requires(std::is_void_v<DevT> || sizeof(HostT) == sizeof(DevT))
struct DeviceTypedTensor : virtual TypedTensor<HostT> {
  public:
    /**
     * @typedef dev_datatype
     *
     * @brief The data type stored on the device. This is only different if T is complex.
     */
    using dev_datatype =
        std::conditional_t<std::is_void_v<DevT>,
                           std::conditional_t<std::is_same_v<HostT, std::complex<float>>, hipComplex,
                                              std::conditional_t<std::is_same_v<HostT, std::complex<double>>, hipDoubleComplex, HostT>>,
                           DevT>;

    /**
     * @typedef host_datatype
     *
     * @brief The datatype that the host sees.
     */
    using host_datatype = HostT;

    /**
     * Default constructor.
     */
    DeviceTypedTensor() = default;

    /**
     * Default copy constructor.
     */
    DeviceTypedTensor(DeviceTypedTensor const &) = default;

    /**
     * Default destructor.
     */
    ~DeviceTypedTensor() override = default;
};
#endif

/**
 * @struct RankTensorNoRank
 *
 * @brief Base class for RankTensorBase that does not store the rank.
 *
 * This is used to indicate that the tensor has a rank that is known at
 * compile time without needing to provide the rank. Internal use only.
 */
struct RankTensorNoRank {
    /**
     * Default constructor.
     */
    RankTensorNoRank() = default;

    /**
     * Default copy constructor.
     */
    RankTensorNoRank(RankTensorNoRank const &) = default;

    /**
     * Default destructor.
     */
    virtual ~RankTensorNoRank() = default;
};

/**
 * @struct RankTensor
 *
 * @brief Base class for tensors with a rank. Used for querying the rank.
 *
 * @tparam R The rank of the tensor.
 */
template <size_t R>
struct RankTensor : virtual TensorBase, virtual RankTensorNoRank {
    /**
     * @property Rank
     *
     * @brief The rank of the tensor.
     */
    static constexpr size_t Rank = R;

    /**
     * Default constructor.
     */
    RankTensor() = default;

    /**
     * Default copy constructor.
     */
    RankTensor(RankTensor const &) = default;

    /**
     * Default destructor.
     */
    ~RankTensor() override = default;

    /**
     * Gets the dimensions of the tensor.
     */
    virtual Dim<R> dims() const = 0;

    /**
     * Gets the dimension along a given axis.
     */
    virtual size_t dim(int d) const = 0;
};

/**
 * @struct TensorNoExtra
 *
 * @brief Specifies that a class is a tensor, just without any template parameters. Internal use only. Use TensorBase instead.
 *
 * Used for checking that a type is a tensor without regard for storage type.
 */
struct TensorNoExtra {
    /**
     * Default constructor.
     */
    TensorNoExtra() = default;

    /**
     * Default copy constructor.
     */
    TensorNoExtra(TensorNoExtra const &) = default;

    /**
     * Default destructor.
     */
    virtual ~TensorNoExtra() = default;
};

/**
 * @struct Tensor
 *
 * @brief Base class for all tensors with a type and rank. Just says that a class is a tensor.
 *
 * Indicates a tensor with a rank and type. Some virtual methods need to be defined.
 *
 * @tparam T The type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct Tensor : public virtual TensorNoExtra, virtual TypedTensor<T>, virtual RankTensor<Rank> {
    /**
     * Default constructor.
     */
    Tensor() = default;

    /**
     * Default copy constructor.
     */
    Tensor(Tensor const &) = default;

    /**
     * Default destructor.
     */
    ~Tensor() override = default;

    /// @copydoc TensorBase::full_view_of_underlying()
    bool full_view_of_underlying() const override { return true; }

    /// @copydoc TensorBase::name()
    std::string const &name() const override = 0;

    /// @copydoc TensorBase::set_name()
    void set_name(std::string const &new_name) override = 0;
};

/**
 * @struct LockableTensor
 *
 * @brief Base class for lockable tensors. Works with all of std:: locking functions. Uses a recursive mutex.
 */
struct LockableTensor {
  protected:
    /**
     * @property _lock
     *
     * @brief The base mutex for locking the tensor.
     */
    mutable std::shared_ptr<std::recursive_mutex> _lock; // Make it mutable so that it can be modified even in const methods.

  public:
    /**
     * Default destructor.
     */
    virtual ~LockableTensor() = default;

    /**
     * Default constructor.
     */
    LockableTensor() { _lock = std::make_shared<std::recursive_mutex>(); }

    /**
     * Default copy constructor.
     */
    LockableTensor(LockableTensor const &) { _lock = std::make_shared<std::recursive_mutex>(); }

    /**
     * @brief Lock the tensor.
     */
    virtual void lock() const { _lock->lock(); }

    /**
     * @brief Try to lock the tensor. Returns false if a lock could not be obtained, or true if it could.
     */
    virtual bool try_lock() const { return _lock->try_lock(); }

    /**
     * @brief Unlock the tensor.
     */
    virtual void unlock() const { _lock->unlock(); }

    /**
     * @brief Get the mutex.
     */
    virtual std::shared_ptr<std::recursive_mutex> get_mutex() const { return _lock; }

    /**
     * @brief Set the mutex.
     */
    void set_mutex(std::shared_ptr<std::recursive_mutex> mutex) { _lock = mutex; }
};

/*==================
 * Location-based.
 *==================*/

/**
 * @struct CoreTensor
 *
 * @brief Represents a tensor only available to the core.
 */
struct CoreTensor {
    /**
     * Default constructor.
     */
    CoreTensor() = default;

    /**
     * Default copy constructor.
     */
    CoreTensor(CoreTensor const &) = default;

    /**
     * Default destructor.
     */
    virtual ~CoreTensor() = default;
};

#if defined(EINSUMS_COMPUTE_CODE)
/**
 * @struct DeviceTensor
 *
 * @brief Represents a tensor available to graphics hardware.
 */
struct DeviceTensor {
  public:
    /**
     * Default constructor.
     */
    DeviceTensor() = default;

    /**
     * Default copy constructor.
     */
    DeviceTensor(DeviceTensor const &) = default;

    /**
     * Default destructor.
     */
    virtual ~DeviceTensor() = default;
};
#endif

/**
 * @struct DiskTensor
 *
 * @brief Represents a tensor stored on disk.
 */
struct DiskTensor {
    /**
     * Default constructor.
     */
    DiskTensor() = default;

    /**
     * Default copy constructor.
     */
    DiskTensor(DiskTensor const &) = default;

    /**
     * Default destructor.
     */
    virtual ~DiskTensor() = default;
};

/*===================
 * Other properties.
 *===================*/

/**
 * @struct TensorViewNoExtra
 *
 * @brief Internal property that specifies that a tensor is a view.
 *
 * This specifies that a tensor is a view without needing to specify template parameters.
 * This struct is not intended to be used directly by the user, and is used as a base class
 * for TensorViewBase.
 */
struct TensorViewNoExtra {
    /**
     * Default constructor.
     */
    TensorViewNoExtra() = default;

    /**
     * Default copy constructor.
     */
    TensorViewNoExtra(TensorViewNoExtra const &) = default;

    /**
     * Default destructor.
     */
    virtual ~TensorViewNoExtra() = default;
};

/**
 * @struct TensorView
 *
 * @brief Represents a view of a different tensor.
 *
 * @tparam T The type stored by the underlying tensor.
 * @tparam Rank The rank of the view.
 * @tparam UnderlyingType The tensor type viewed by this view.
 */
template <typename UnderlyingType>
struct TensorView : public virtual TensorViewNoExtra, virtual TensorBase {
  public:
    /**
     * @typedef underlying_type
     *
     * This is the type of tensor being viewed.
     */
    using underlying_type = UnderlyingType;

    /**
     * Default constructor.
     */
    TensorView() = default;

    /**
     * Default copy constructor.
     */
    TensorView(TensorView const &) = default;

    /**
     * Default destructor.
     */
    ~TensorView() override = default;

    /**
     * @copydoc TensorBase::full_view_of_underlying()
     */
    bool full_view_of_underlying() const override = 0;
};

/**
 * @struct BasicTensorNoExtra
 *
 * @brief Represents a regular tensor. Internal use only. Use BasicTensorBase instead.
 *
 * Represents a regular tensor, but without template parameters. See BasicTensorBase for more details.
 */
struct BasicTensorNoExtra {
    /**
     * Default constructor.
     */
    BasicTensorNoExtra() = default;

    /**
     * Default copy constructor.
     */
    BasicTensorNoExtra(BasicTensorNoExtra const &) = default;

    /**
     * Default destructor.
     */
    virtual ~BasicTensorNoExtra() = default;
};

/**
 * @struct BasicTensor
 *
 * @brief Represents a regular tensor. It has no special layouts.
 *
 * Represents a tensor that is not block diagonal, tiled, or otherwise.
 * Just a plain vanilla tensor. These store memory in such a way that it
 * is possible to pass to BLAS and LAPACK.
 *
 * @tparam T The type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct BasicTensor : virtual Tensor<T, Rank>, virtual BasicTensorNoExtra {
    /**
     * Default constructor.
     */
    BasicTensor() = default;

    /**
     * Default copy constructor.
     */
    BasicTensor(BasicTensor const &) = default;

    /**
     * Default destructor.
     */
    ~BasicTensor() override = default;

    /**
     * Get a pointer to the data stored in this tensor.
     * @pure
     */
    virtual T *data() = 0;

    /**
     * @copydoc BasicTensor::data()
     * @pure
     */
    virtual T const *data() const = 0;

    /**
     * Get the stride of the tensor along a given axis.
     *
     * @param d The axis to query.
     * @pure
     */
    virtual size_t stride(int d) const = 0;

    /**
     * Get the strides of the tensor.
     * @pure
     */
    virtual Stride<Rank> strides() const = 0;
};

/**
 * @struct CollectedTensorNoExtra
 *
 * @brief Represents a tensor that stores things in a collection. Internal use only.
 *
 * Specifies that a tensor is actually a collection of tensors without needing to specify
 * template parameters. Only used internally. Use CollectedTensorBase for your code.
 */
struct CollectedTensorNoExtra {
    /**
     * Default constructor.
     */
    CollectedTensorNoExtra() = default;

    /**
     * Default copy constructor.
     */
    CollectedTensorNoExtra(CollectedTensorNoExtra const &) = default;

    /**
     * Default destructor.
     */
    virtual ~CollectedTensorNoExtra() = default;
};

/**
 * @struct CollectedTensorOnlyStored
 *
 * @brief Represents a tensor that stores things in a collection. Internal use only.
 *
 * Specifies that a tensor is actually a collection of tensors without needing to specify
 * template parameters. Only used internally. Use CollectedTensorBase for your code.
 *
 * @tparam Stored The type of tensor stored.
 */
template <typename Stored>
struct CollectedTensorOnlyStored {
    /**
     * Default constructor.
     */
    CollectedTensorOnlyStored() = default;

    /**
     * Default copy constructor.
     */
    CollectedTensorOnlyStored(CollectedTensorOnlyStored const &) = default;

    /**
     * Default destructor.
     */
    virtual ~CollectedTensorOnlyStored() = default;
};

/**
 * @struct CollectedTensor
 *
 * @brief Specifies that a tensor is a collection of other tensors.
 *
 * Examples of tensor collections include BlockTensors and TiledTensors, which store
 * lists of tensors to save on memory.
 *
 * @tparam TensorType The type of tensor stored by the collection.
 */
template <typename TensorType>
struct CollectedTensor : virtual CollectedTensorNoExtra, virtual TensorNoExtra {
    /**
     * @typedef tensor_type
     *
     * This is the type of tensor being stored.
     */
    using tensor_type = TensorType;

    /**
     * Default constructor.
     */
    CollectedTensor() = default;

    /**
     * Default copy constructor.
     */
    CollectedTensor(CollectedTensor const &) = default;

    /**
     * Default destructor.
     */
    ~CollectedTensor() override = default;
};

/**
 * @struct TiledTensorNoExtra
 *
 * @brief Specifies that a tensor is a tiled tensor without needing to specify type parameters.
 *
 * Only used internally. Use TiledTensorBase in your code.
 */
struct TiledTensorNoExtra {
    /**
     * Default constructor.
     */
    TiledTensorNoExtra() = default;

    /**
     * Default copy constructor.
     */
    TiledTensorNoExtra(TiledTensorNoExtra const &) = default;

    /**
     * Default destructor.
     */
    virtual ~TiledTensorNoExtra() = default;
};

// Large class. See TiledTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct TiledTensor;

/**
 * @struct BlockTensorNoExtra
 *
 * @brief Specifies that a tensor is a block tensor. Internal use only. Use BlockTensorBase instead.
 *
 * Specifies that a tensor is a block tensor without needing template parameters. Internal use only.
 * Use BlockTensorBase in your code.
 */
struct BlockTensorNoExtra {
    /**
     * Default constructor.
     */
    BlockTensorNoExtra() = default;

    /**
     * Default copy constructor.
     */
    BlockTensorNoExtra(BlockTensorNoExtra const &) = default;

    /**
     * Default destructor.
     */
    virtual ~BlockTensorNoExtra() = default;
};

// Large class. See BlockTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct BlockTensor;

/**
 * @struct FunctionTensorNoExtra
 *
 * @brief Specifies that a tensor is a function tensor, but with no template parameters. Internal use only. Use FunctionTensorBase
 * instead.
 *
 * Used for checking to see if a tensor is a function tensor. Internal use only. Use FunctionTensorBase instead.
 */
struct FunctionTensorNoExtra {
    /**
     * Default constructor.
     */
    FunctionTensorNoExtra() = default;

    /**
     * Default copy constructor.
     */
    FunctionTensorNoExtra(FunctionTensorNoExtra const &) = default;

    /**
     * Default destructor.
     */
    virtual ~FunctionTensorNoExtra() = default;
};

// Large class. See FunctionTensor.hpp for code.
template <typename T, size_t Rank>
struct FunctionTensor;

/**
 * @struct AlgebraOptimizedTensor
 *
 * @brief Specifies that the tensor type can be used by einsum to select different routines other than the generic algorithm.
 */
struct AlgebraOptimizedTensor {
    /**
     * Default constructor.
     */
    AlgebraOptimizedTensor() = default;

    /**
     * Default copy constructor.
     */
    AlgebraOptimizedTensor(AlgebraOptimizedTensor const &) = default;

    /**
     * Default destructor.
     */
    virtual ~AlgebraOptimizedTensor() = default;
};

class RuntimeTensorNoType {};

class RuntimeTensorViewNoType {};

} // namespace einsums::tensor_base