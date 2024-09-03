#pragma once

#include "einsums/_Common.hpp"

#include <memory>
#include <mutex>

#ifdef __HIP__
#    include <hip/hip_complex.h>
#endif

namespace einsums::tensor_props {
/*===============
 * Base classes
 *===============*/

/**
 * @struct TypedTensorBase
 *
 * @brief Represents a tensor that stores a data type.
 *
 * @tparam T The data type the tensor stores.
 */
template <typename T>
struct TypedTensorBase {
  public:
    /**
     * @typedef data_type
     *
     * @brief Gets the stored data type.
     */
    using data_type = T;

    TypedTensorBase()                        = default;
    TypedTensorBase(const TypedTensorBase &) = default;

    virtual ~TypedTensorBase() = default;
};

#ifdef __HIP__
/**
 * @struct DevTypedTensorBase
 *
 * Represents a tensor that stores different data types on the host and the device.
 * By default, if the host stores complex<float> or complex<double>, then it converts
 * it to hipComplex or hipDoubleComplex. The two types should have the same storage size.
 * @tparam HostT The host data type.
 * @tparam DevT The device type.
 */
template <typename HostT, typename DevT = void>
    requires(std::is_void_v<DevT> || sizeof(HostT) == sizeof(DevT))
struct DevTypedTensorBase : public virtual TypedTensorBase<HostT> {
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

    DevTypedTensorBase()                           = default;
    DevTypedTensorBase(const DevTypedTensorBase &) = default;

    virtual ~DevTypedTensorBase() = default;
};
#endif

/**
 * @struct RankTensorBase
 *
 * @brief Base class for tensors with a rank. Used for querying the rank.
 *
 * @tparam Rank The rank of the tensor.
 */
template <size_t Rank>
struct RankTensorBase {
  public:
    /**
     * @property rank
     *
     * @brief The rank of the tensor.
     */
    static constexpr size_t rank = Rank;

    RankTensorBase()                       = default;
    RankTensorBase(const RankTensorBase &) = default;

    virtual ~RankTensorBase() = default;

    virtual Dim<Rank> dims() const = 0;

    virtual auto dim(int d) const -> size_t = 0;
};

/**
 * @struct TensorBaseNoExtra
 *
 * @brief Specifies that a class is a tensor, just without any template parameters. Internal use only. Use TensorBase instead.
 *
 * Used for checking that a type is a tensor without regard for storage type.
 */
struct TensorBaseNoExtra {
  public:
    TensorBaseNoExtra()                          = default;
    TensorBaseNoExtra(const TensorBaseNoExtra &) = default;

    virtual ~TensorBaseNoExtra() = default;
};

/**
 * @struct TensorBase
 *
 * @brief Base class for all tensors. Just says that a class is a tensor.
 *
 * Indicates a tensor with a rank and type. Some virtual methods need to be defined.
 *
 * @tparam T The type stored by the tensor.
 * @tparam Rank The rank of the tensor.
 */
template <typename T, size_t Rank>
struct TensorBase : public virtual TensorBaseNoExtra, virtual TypedTensorBase<T>, virtual RankTensorBase<Rank> {
  public:
    TensorBase() = default;

    TensorBase(const TensorBase &) = default;

    virtual ~TensorBase() = default;

    virtual bool full_view_of_underlying() const { return true; }

    virtual const std::string &name() const = 0;

    virtual void set_name(const std::string &new_name) = 0;
};

/**
 * @struct LockableTensorBase
 *
 * @brief Base class for lockable tensors. Works with all of the std:: locking functions. Uses a recursive mutex.
 */
struct LockableTensorBase {
  protected:
    /**
     * @property _lock
     *
     * @brief The base mutex for locking the tensor.
     */
    mutable std::shared_ptr<std::recursive_mutex> _lock; // Make it mutable so that it can be modified even in const methods.

  public:
    LockableTensorBase() { _lock = std::make_shared<std::recursive_mutex>(); }

    LockableTensorBase(const LockableTensorBase &) { _lock = std::make_shared<std::recursive_mutex>(); }

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
    std::shared_ptr<std::recursive_mutex> get_mutex() const { return _lock; }

    /**
     * @brief Set the mutex.
     */
    void set_mutex(std::shared_ptr<std::recursive_mutex> mutex) { _lock = mutex; }
};

/*==================
 * Location-based.
 *==================*/

/**
 * @struct CoreTensorBase
 *
 * @brief Represents a tensor only available to the core.
 */
struct CoreTensorBase {
  public:
    CoreTensorBase()                       = default;
    CoreTensorBase(const CoreTensorBase &) = default;

    virtual ~CoreTensorBase() = default;
};

#ifdef __HIP__
/**
 * @struct DeviceTensorBase
 *
 * @brief Represents a tensor available to graphics hardware.
 */
struct DeviceTensorBase {
  public:
    DeviceTensorBase()                         = default;
    DeviceTensorBase(const DeviceTensorBase &) = default;

    virtual ~DeviceTensorBase() = default;
};
#endif

/**
 * @struct DiskTensorBase
 *
 * @brief Represents a tensor stored on disk.
 */
struct DiskTensorBase {
  public:
    DiskTensorBase()                       = default;
    DiskTensorBase(const DiskTensorBase &) = default;

    virtual ~DiskTensorBase() = default;
};

/*===================
 * Other properties.
 *===================*/

/**
 * @struct TensorViewBaseNoExtra
 *
 * @brief Internal property that specifies that a tensor is a view.
 *
 * This specifies that a tensor is a view without needing to specify template parameters.
 * This struct is not intended to be used directly by the user, and is used as a base class
 * for TensorViewBase.
 */
struct TensorViewBaseNoExtra {
  public:
    TensorViewBaseNoExtra()                              = default;
    TensorViewBaseNoExtra(const TensorViewBaseNoExtra &) = default;

    virtual ~TensorViewBaseNoExtra() = default;
};

/**
 * @struct TensorViewBaseOnlyViewed
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
struct TensorViewBaseOnlyViewed {
  public:
    TensorViewBaseOnlyViewed()                                 = default;
    TensorViewBaseOnlyViewed(const TensorViewBaseOnlyViewed &) = default;

    virtual ~TensorViewBaseOnlyViewed() = default;
};

/**
 * @struct TensorViewBase
 *
 * @brief Represents a view of a different tensor.
 *
 * @tparam T The type stored by the underlying tensor.
 * @tparam Rank The rank of the view.
 * @tparam UnderlyingType The tensor type viewed by this view.
 */
template <typename T, size_t Rank, typename UnderlyingType>
struct TensorViewBase : public virtual TensorViewBaseNoExtra,
                        virtual TensorViewBaseOnlyViewed<UnderlyingType>,
                        virtual TensorBase<T, Rank> {
  public:
    using underlying_type = UnderlyingType;

    TensorViewBase()                       = default;
    TensorViewBase(const TensorViewBase &) = default;

    virtual ~TensorViewBase() = default;

    virtual bool full_view_of_underlying() const override = 0;
};

/**
 * @struct BasicTensorBaseNoExtra
 *
 * @brief Represents a regular tensor. Internal use only. Use BasicTensorBase instead.
 *
 * Represents a regular tensor, but without template parameters. See BasicTensorBase for more details.
 */
struct BasicTensorBaseNoExtra {
  public:
    BasicTensorBaseNoExtra()                               = default;
    BasicTensorBaseNoExtra(const BasicTensorBaseNoExtra &) = default;

    virtual ~BasicTensorBaseNoExtra() = default;
};

/**
 * @struct BasicTensorBase
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
struct BasicTensorBase : virtual TensorBase<T, Rank>, virtual BasicTensorBaseNoExtra {
    BasicTensorBase()                        = default;
    BasicTensorBase(const BasicTensorBase &) = default;

    virtual ~BasicTensorBase() = default;

    virtual T       *data()       = 0;
    virtual const T *data() const = 0;

    virtual size_t       stride(int d) const = 0;
    virtual Stride<Rank> strides() const     = 0;
};

/**
 * @struct CollectedTensorBaseNoExtra
 *
 * @brief Represents a tensor that stores things in a collection. Internal use only.
 *
 * Specifies that a tensor is actually a collection of tensors without needing to specify
 * template parameters. Only used internally. Use CollectedTensorBase for your code.
 */
struct CollectedTensorBaseNoExtra {
    CollectedTensorBaseNoExtra()                                   = default;
    CollectedTensorBaseNoExtra(const CollectedTensorBaseNoExtra &) = default;

    virtual ~CollectedTensorBaseNoExtra() = default;
};

/**
 * @struct CollectedTensorBaseOnlyStored
 *
 * @brief Represents a tensor that stores things in a collection. Internal use only.
 *
 * Specifies that a tensor is actually a collection of tensors without needing to specify
 * template parameters. Only used internally. Use CollectedTensorBase for your code.
 *
 * @tparam Stored The type of tensor stored.
 */
template <typename Stored>
struct CollectedTensorBaseOnlyStored {
    CollectedTensorBaseOnlyStored()                                      = default;
    CollectedTensorBaseOnlyStored(const CollectedTensorBaseOnlyStored &) = default;

    virtual ~CollectedTensorBaseOnlyStored() = default;
};

/**
 * @struct CollectedTensorBase
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
struct CollectedTensorBase : public virtual CollectedTensorBaseNoExtra,
                             virtual CollectedTensorBaseOnlyStored<TensorType>,
                             virtual TensorBase<T, Rank> {
  public:
    using tensor_type = TensorType;

    CollectedTensorBase()                            = default;
    CollectedTensorBase(const CollectedTensorBase &) = default;

    virtual ~CollectedTensorBase() = default;
};

/**
 * @struct TiledTensorBaseNoExtra
 *
 * @brief Specifies that a tensor is a tiled tensor without needing to specify type parameters.
 *
 * Only used internally. Use TiledTensorBase in your code.
 */
struct TiledTensorBaseNoExtra {
    TiledTensorBaseNoExtra()                               = default;
    TiledTensorBaseNoExtra(const TiledTensorBaseNoExtra &) = default;

    virtual ~TiledTensorBaseNoExtra() = default;
};

// Large class. See TiledTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct TiledTensorBase;

/**
 * @struct BlockTensorBaseNoExtra
 *
 * @brief Specifies that a tensor is a block tensor. Internal use only. Use BlockTensorBase instead.
 *
 * Specifies that a tensor is a block tensor without needing template parameters. Internal use only.
 * Use BlockTensorBase in your code.
 */
struct BlockTensorBaseNoExtra {
    BlockTensorBaseNoExtra()                               = default;
    BlockTensorBaseNoExtra(const BlockTensorBaseNoExtra &) = default;

    virtual ~BlockTensorBaseNoExtra() = default;
};

// Large class. See BlockTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct BlockTensorBase;

/**
 * @struct FunctionTensorBaseNoExtra
 *
 * @brief Specifies that a tensor is a function tensor, but with no template parameters. Internal use only. Use FunctionTensorBase instead.
 *
 * Used for checking to see if a tensor is a function tensor. Internal use only. Use FunctionTensorBase instead.
 */
struct FunctionTensorBaseNoExtra {
  public:
    FunctionTensorBaseNoExtra()                                  = default;
    FunctionTensorBaseNoExtra(const FunctionTensorBaseNoExtra &) = default;

    virtual ~FunctionTensorBaseNoExtra() = default;
};

// Large class. See FunctionTensor.hpp for code.
template <typename T, size_t Rank>
struct FunctionTensorBase;

/**
 * @struct AlgebraOptimizedTensor
 *
 * @brief Specifies that the tensor type can be used by einsum to select different routines other than the generic algorithm.
 */
struct AlgebraOptimizedTensor {
  public:
    AlgebraOptimizedTensor()                               = default;
    AlgebraOptimizedTensor(const AlgebraOptimizedTensor &) = default;

    virtual ~AlgebraOptimizedTensor() = default;
};

/**
 * @struct PyTensorBase
 *
 * @brief Tensor base for tensors that can be bound with Python.
 *
 * This class stores its rank as a variable rather than as a template parameter.
 * This means that the einsum call can not be optimized at compile time.
 * This is the base class, so it doesn't store much in the way of implementation.
 * This class can not be bound with Python. Only its derived classes should be able to be
 * bound with Python.
 *
 * @tparam T The data type stored.
 */
struct PyTensorBase : public
                        virtual tensor_props::LockableTensorBase {

protected:
    size_t _rank{0};

public:
    PyTensorBase() = default;
    PyTensorBase(const PyTensorBase &) = default;

    PyTensorBase(size_t rank) : _rank{rank} {}

    virtual size_t get_rank() const {
        return _rank;
    }
};

} // namespace einsums::tensor_props