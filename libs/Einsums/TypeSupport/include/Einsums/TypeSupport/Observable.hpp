//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <condition_variable>
#include <functional>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <vector>

namespace einsums {

namespace design_pats {

/**
 * @struct Observable
 *
 * @brief Implementation of the Subject part of the Observable pattern from the Gang of Four.
 *
 * Here is an example of the use of this class.
 *
 * @code
 *  void thread_observer(Observable<int> &observable) {
 *      while (true) {
 *          observable.wait_for_change();
 *          int new_value = observable; // Use casting operator to get the value
 *          std::cout << "Thread observer detected change: " << new_value << '\n';
 *      }
 *  }
 *
 *  int main() {
 *      Observable<int> observable(0); // Initialize with 0
 *
 *      // Register a callback observer
 *      observable.add_observer([](const int& value) {
 *          std::cout << "Callback observer detected change: " << value << '\n';
 *      });
 *
 *      // Launch a thread-based observer
 *      std::thread observer_thread(thread_observer, std::ref(observable));
 *
 *      // Simulate changes in the variable
 *      for (int i = 1; i <= 5; ++i) {
 *          std::this_thread::sleep_for(std::chrono::seconds(1));
 *          observable = i; // Use assignment operator
 *      }
 *
 *      observer_thread.join();
 *      return 0;
 *  }
 *  @endcode
 */
template <typename T>
struct Observable {
  public:
    /**
     * @brief Constructor.
     *
     * @param initial_value The initial value of the observable.
     */
    Observable(T initial_value = T{}) : _state(std::move(initial_value)) {}

    /**
     * @brief Assignment operator for setting the value
     *
     * @param value The value to assign to the observable.
     */
    Observable &operator=(T const &value) {
        set_value(value);
        return *this;
    }

    /**
     * @brief Casting operator for retrievign the value.
     */
    operator T() const { return get_value(); }

    /**
     * @brief Explicit getter.
     */
    T const &get_value() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _state;
    }

    /**
     * @brief Get a lock that updates internal values as well as the state.
     */
    T &get_value() { return _state; }

    /**
     * @brief Lock the state.
     */
    void lock() { _mutex.lock(); }

    /**
     * @brief Unlock the state.
     *
     * This will automatically notify observers when done if no arguments are passed.
     * If false is passed, then it will not notify the observers.
     */
    void unlock(bool notify = true) {
        _mutex.unlock();

        if (notify) {
            notify_observers();
        }
    }

    bool try_lock() { return _mutex.try_lock(); }

    /**
     * @brief Wait for the value to change.
     */
    void wait_for_change() {
        size_t                       check_changed = _value_changed;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this, check_changed]() { return _value_changed == check_changed; });
    }

    /**
     * @brief Register an observer callback.
     */
    void attach(std::function<void(T const &)> observer) {
        std::lock_guard<std::mutex> lock(_observer_mutex);
        _observers.push_back(std::move(observer));
    }

    /**
     * @brief Remove an observer callback.
     */
    void detach(std::function<void(T const &)> const &observer) {
        std::lock_guard<std::mutex> lock(_observer_mutex);

        _observers.remove_if([&observer](auto const &elem) { return observer.target() == elem.target(); });
    }

    /**
     * @brief Check to see if the observable has recently changed.
     */
    bool changed() const { return _value_changed; }

  protected:
    /**
     * @brief Notify all of the observers that observe this observable.
     */
    void notify_observers() {
        // Notify things that are waiting for changes.
        _mutex.lock();
        _value_changed++;
        _mutex.unlock();

        std::lock_guard<std::mutex> lock(_observer_mutex);
        for (auto const &observer : _observers) {
            observer(*this); // Call each registered observer with the new values
        }
    }

    /**
     * @property _state
     *
     * @brief The internal state of the observable.
     */
    T                       _state;
    mutable std::mutex      _mutex{};           /// For thread-safe value access
    std::condition_variable _cv{};              /// For thread synchronization
    size_t                  _value_changed = 0; /// Counter indicating how many times the value has changed.

    std::list<std::function<void(T const &)>> _observers{};      /// List of observers
    std::mutex                                _observer_mutex{}; /// Protects the observer list
};
} // namespace design_pats
} // namespace einsums