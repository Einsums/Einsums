#pragma once

namespace einsums {

namespace design_pats {

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