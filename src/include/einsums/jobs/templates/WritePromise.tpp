/**
 * @file WritePromise.tpp
 *
 * Contains method definitions for the WritePromise class in the Jobs.hpp file.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/Timeout.hpp"
#include "einsums/jobs/WritePromise.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

template <typename T>
WritePromise<T>::WritePromise(unsigned long id, Resource<T> *data) : ReadPromise<T>(id, data) {
}

template <typename T>
template <typename Inttype, typename Ratio>
std::shared_ptr<T> WritePromise<T>::get(std::chrono::duration<Inttype, Ratio> time_out) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while (!this->ready()) {
        std::this_thread::yield();

        std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
        if (curr - start >= time_out) {
            throw timeout();
        }
    }

    std::atomic_thread_fence(std::memory_order_acquire);

    return this->data->get_data();
}

template <typename T>
std::shared_ptr<T> WritePromise<T>::get() {
    while (!this->ready()) {
        std::this_thread::yield();
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    return this->data->get_data();
}

template <typename T>
WritePromise<T>::operator T &() {
    return *(this->get());
}

template <typename T>
bool WritePromise<T>::is_exclusive() const {
    return true;
}

template <typename T>
bool WritePromise<T>::ready() {
    return this->data->is_writable(*this);
}

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)