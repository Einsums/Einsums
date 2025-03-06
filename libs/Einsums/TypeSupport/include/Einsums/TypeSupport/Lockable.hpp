#pragma once

namespace einsums {

namespace design_pats {

/**
 * @class Lockable
 *
 * @brief Base class that enables mutexes on an object, making thread safety a breeze.
 *
 * Simply inherit this class and the new class will have everything it needs to satisfy
 * the Lockable requirement. You can even specify what kind of mutex to use to handle
 * the locks.
 */
template <typename Mutex>
class Lockable {
  public:
    /**
     * Default constructor.
     */
    Lockable() = default;

    /**
     * "Copy constructor." It doesn't perform any copies and is only
     * necessary for subclasses so that they can define copy constructors.
     */
    Lockable(Lockable<Mutex> const &) : lock_{} {};

    /**
     * @brief Lock the object.
     */
    void lock() const { this->lock_.lock(); }

    /**
     * @brief Try to lock the object. Returns true if successful.
     */
    bool try_lock() const { return this->lock_.try_lock(); }

    /**
     * @brief Unlock the object.
     */
    void unlock() const { this->lock_.unlock(); }

    /**
     * @brief Get the underlying mutex.
     */
    Mutex &get_mutex() { return lock_; }

  protected:
    /**
     * @property lock_
     *
     * @brief The underlying locking object. Usually some sort of mutex.
     */
    mutable Mutex lock_;
};

} // namespace design_pats

} // namespace einsums