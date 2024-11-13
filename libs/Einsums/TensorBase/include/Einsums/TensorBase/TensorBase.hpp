//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/TensorBase/Common.hpp>

#include <complex>
#include <mutex>

#if defined(EINSUMS_COMPUTE_CODE)
#    include <hip/hip_complex.h>
#endif

namespace einsums::tensor_base {

/**
 * @struct TypedTensor
 *
 * @brief Represents a tensor that stores a data type.
 *
 * @tparam T The data type the tensor stores.
 */
template <typename T>
struct TypedTensor {
    /**
     * @typedef value_type
     *
     * @brief Gets the stored data type.
     */
    using value_type = T;

    TypedTensor()                    = default;
    TypedTensor(TypedTensor const &) = default;

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
struct DeviceTypedTensor : virtual typed_tensor<HostT> {
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

    using host_datatype = HostT;

    DeviceTypedTensor()                          = default;
    DeviceTypedTensor(DeviceTypedTensor const &) = default;

    ~DeviceTypedTensor() override = default;
};
#endif

/**
 * @struct RankTensor
 *
 * @brief Base class for tensors with a rank. Used for querying the rank.
 *
 * @tparam Rank The rank of the tensor.
 */
template <size_t Rank>
struct RankTensor {
    /**
     * @property rank
     *
     * @brief The rank of the tensor.
     */
    static constexpr size_t rank = Rank;

    RankTensor()                   = default;
    RankTensor(RankTensor const &) = default;

    virtual ~RankTensor() = default;

    virtual Dim<Rank> dims() const = 0;

    virtual auto dim(int d) const -> size_t = 0;
};

/**
 * @struct TensorNoExtra
 *
 * @brief Specifies that a class is a tensor, just without any template parameters. Internal use only. Use TensorBase instead.
 *
 * Used for checking that a type is a tensor without regard for storage type.
 */
struct TensorNoExtra {
    TensorNoExtra()                      = default;
    TensorNoExtra(TensorNoExtra const &) = default;

    virtual ~TensorNoExtra() = default;
};

/**
 * @struct Tensor
 *
 * @brief Base class for all tensors. Just says that a class is a tensor.
 *
 * Indicates a tensor with a rank and type. Some virtual methods need to be defined.
 *
 * @tparam T The type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct Tensor : virtual TensorNoExtra, virtual TypedTensor<T>, virtual RankTensor<Rank> {
    Tensor()               = default;
    Tensor(Tensor const &) = default;

    ~Tensor() override = default;

    virtual auto full_view_of_underlying() const -> bool { return true; }
    virtual auto name() const -> std::string const   & = 0;
    virtual void set_name(std::string const &new_name) = 0;
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
    virtual ~LockableTensor() = default;

    LockableTensor() { _lock = std::make_shared<std::recursive_mutex>(); }
    LockableTensor(LockableTensor const &) { _lock = std::make_shared<std::recursive_mutex>(); }

    /**
     * @brief Lock the tensor.
     */
    virtual auto lock() const -> void { _lock->lock(); }

    /**
     * @brief Try to lock the tensor. Returns false if a lock could not be obtained, or true if it could.
     */
    virtual auto try_lock() const -> bool { return _lock->try_lock(); }

    /**
     * @brief Unlock the tensor.
     */
    virtual void unlock() const { _lock->unlock(); }

    /**
     * @brief Get the mutex.
     */
    auto get_mutex() const -> std::shared_ptr<std::recursive_mutex> { return _lock; }

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
    CoreTensor()                   = default;
    CoreTensor(CoreTensor const &) = default;

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
    DeviceTensor()                     = default;
    DeviceTensor(DeviceTensor const &) = default;

    virtual ~DeviceTensor() = default;
};
#endif

/**
 * @struct DiskTensor
 *
 * @brief Represents a tensor stored on disk.
 */
struct DiskTensor {
    DiskTensor()                   = default;
    DiskTensor(DiskTensor const &) = default;

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
    TensorViewNoExtra()                          = default;
    TensorViewNoExtra(TensorViewNoExtra const &) = default;

    virtual ~TensorViewNoExtra() = default;
};

/**
 * @struct TensorViewOnlyViewed
 *
 * @brief Internal property that specifies that a tensor is a view.
 *
 * This specifies that a tensor is a view without needing to specify type and rank.
 * This struct is not intended to be used directly by the user, and is used as a base class
 * for TensorViewBase.
 *
 * @tparam Viewed The tensor type viewed by this tensor.
 */
template <typename Viewed>
struct TensorViewOnlyViewed {
    TensorViewOnlyViewed()                             = default;
    TensorViewOnlyViewed(TensorViewOnlyViewed const &) = default;

    virtual ~TensorViewOnlyViewed() = default;
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
template <typename T, size_t Rank, typename UnderlyingType>
struct TensorView : public virtual TensorViewNoExtra, virtual TensorViewOnlyViewed<UnderlyingType>, virtual Tensor<T, Rank> {
  public:
    using underlying_type = UnderlyingType;

    TensorView()                   = default;
    TensorView(TensorView const &) = default;

    ~TensorView() override = default;

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
    BasicTensorNoExtra()                           = default;
    BasicTensorNoExtra(BasicTensorNoExtra const &) = default;

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
    BasicTensor()                    = default;
    BasicTensor(BasicTensor const &) = default;

    ~BasicTensor() override = default;

    virtual auto data() -> T             * = 0;
    virtual auto data() const -> T const * = 0;

    virtual auto stride(int d) const -> size_t   = 0;
    virtual auto strides() const -> Stride<Rank> = 0;
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
    CollectedTensorNoExtra()                               = default;
    CollectedTensorNoExtra(CollectedTensorNoExtra const &) = default;

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
    CollectedTensorOnlyStored()                                  = default;
    CollectedTensorOnlyStored(CollectedTensorOnlyStored const &) = default;

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
 * @tparam T The type of data stored by the tensors in the collection.
 * @tparam Rank The rank of the tensor.
 * @tparam TensorType The type of tensor stored by the collection.
 */
template <typename T, size_t Rank, typename TensorType>
struct CollectedTensor : public virtual CollectedTensorNoExtra, virtual CollectedTensorOnlyStored<TensorType>, virtual Tensor<T, Rank> {
    using tensor_type = TensorType;

    CollectedTensor()                        = default;
    CollectedTensor(CollectedTensor const &) = default;

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
    TiledTensorNoExtra()                           = default;
    TiledTensorNoExtra(TiledTensorNoExtra const &) = default;

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
    BlockTensorNoExtra()                           = default;
    BlockTensorNoExtra(BlockTensorNoExtra const &) = default;

    virtual ~BlockTensorNoExtra() = default;
};

// Large class. See BlockTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct BlockTensorBase;

/**
 * @struct FunctionTensorNoExtra
 *
 * @brief Specifies that a tensor is a function tensor, but with no template parameters. Internal use only. Use FunctionTensorBase instead.
 *
 * Used for checking to see if a tensor is a function tensor. Internal use only. Use FunctionTensorBase instead.
 */
struct FunctionTensorNoExtra {
    FunctionTensorNoExtra()                              = default;
    FunctionTensorNoExtra(FunctionTensorNoExtra const &) = default;

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
    AlgebraOptimizedTensor()                               = default;
    AlgebraOptimizedTensor(AlgebraOptimizedTensor const &) = default;

    virtual ~AlgebraOptimizedTensor() = default;
};

} // namespace einsums::tensor_base