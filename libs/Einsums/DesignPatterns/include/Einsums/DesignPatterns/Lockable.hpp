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
    Lockable()                        = default;
    Lockable(Lockable<Mutex> const &) : lock_{} {
    };

    void lock() const { this->lock_.lock(); }

    bool try_lock() const { return this->lock_.try_lock(); }

    void unlock() const { this->lock_.unlock(); }

    Mutex &get_mutex() {
        return lock_;
    }

  protected:
    mutable Mutex lock_;
};

} // namespace design_pats

} // namespace einsums