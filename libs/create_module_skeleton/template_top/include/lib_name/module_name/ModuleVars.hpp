#pragma once

#include <Einsums/DesignPatterns/Singleton.hpp>
#include <Einsums/Config.hpp>

namespace einsums {{
namespace detail {{

// TODO: This class can be freely changed. It is provided as a starting point for your convenience.

class EINSUMS_EXPORT {lib_name}_{module_name}_vars final {{
    EINSUMS_SINGLETON_DEF({lib_name}_{module_name}_vars)

public:
    // Put module-global variables here.

    inline void lock() const {
        lock_.lock();
    }

    inline bool try_lock() const {
        return lock_.try_lock();
    }

    inline void unlock() const {
        lock_.unlock();
    }

private:
    explicit {lib_name}_{module_name}_vars() = default;

    mutable std::recursive_mutex lock_;

}};

}}

}}