//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

namespace einsums {

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
    /**
     * @brief Constructor.
     *
     * @param initial_value The initial value of the observable.
     */
    Observable(T initial_value = T{}) : _value(std::move(initial_value)) {}

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
     * @brief Explicit value assignment.
     *
     * @param value The value to assign to the observable.
     */
    void set_value(T const &value) {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_value != value) {
                _value         = value;
                _value_changed = true;
                notify_observers(value); // Notify registered observers
            }
        }
        _cv.notify_all(); // Notify threads waiting on value change
    }

    /**
     * @brief Explicit getter.
     */
    T get_value() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _value;
    }

    /**
     * @brief Wait for the value to change.
     */
    void wait_for_change() {
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this]() { return _value_changed; });
        _value_changed = false; // Reset the change flag after notification
    }

    /**
     * @brief Register an observer callback.
     */
    void add_observer(std::function<void(T const &)> observer) {
        std::lock_guard<std::mutex> lock(_observer_mutex);
        _observers.push_back(std::move(observer));
    }

  private:
    /**
     * @brief Notify all of the observers that observe this observable.
     */
    void notify_observers(T const &value) {
        std::lock_guard<std::mutex> lock(_observer_mutex);
        for (auto const &observer : _observers) {
            observer(value); // Call each registered observer with the new value
        }
    }

    /**
     * @property _value
     *
     * @brief The internal state of the observable.
     */
    T                       _value;
    mutable std::mutex      _mutex;                 /// For thread-safe value access
    std::condition_variable _cv;                    /// For thread synchronization
    bool                    _value_changed = false; /// Flag indicating value change

    std::vector<std::function<void(T const &)>> _observers;      /// List of observers
    std::mutex                                  _observer_mutex; /// Protects the observer list
};

} // namespace einsums