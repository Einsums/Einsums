#pragma once

namespace einsums{

namespace design_pats {

template<typename Mutex>
class Lockable {
public:
    Lockable() = default;

    void lock() const {
        this->lock_.lock();
    }

    bool try_lock() const {
        return this->lock_.try_lock();
    }

    void unlock() const {
        this->lock_.unlock();
    }

protected:
    mutable Mutex lock_;
};

}

}