/**
 * @file ReadPromise.tpp
 *
 * Contains code for the ReadPromise class so that it doesn't clutter up Jobs.hpp.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/ReadPromise.hpp"

#include <atomic>
#include <chrono>
#include <thread>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

template <typename T>
ReadPromise<T>::ReadPromise(unsigned long id, Resource<T> *data) : id(id) {
    this->data = data;
}

template <typename T>
ReadPromise<T>::~ReadPromise() {
    this->release();      // Tell the resource to release this lock.
    this->data = nullptr; // Data is owned by someone else, so ignore it.
}

template <typename T>
bool ReadPromise<T>::ready() {
    return this->data->is_readable(*this);
}

template <typename T>
template <typename Inttype, typename Ratio>
const std::shared_ptr<T> ReadPromise<T>::get(std::chrono::duration<Inttype, Ratio> time_out) {
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
const std::shared_ptr<T> ReadPromise<T>::get() {
    while (!this->ready()) {
        std::this_thread::yield();
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    return this->data->get_data();
}

template <typename T>
template <typename Inttype, typename Ratio>
void ReadPromise<T>::wait(std::chrono::duration<Inttype, Ratio> time_out) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while (!this->ready()) {
        std::this_thread::yield();

        std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
        if (curr - start >= time_out) {
            throw jobs::timeout();
        }
    }
}

template <typename T>
void ReadPromise<T>::wait() {
    while (!this->ready()) {
        std::this_thread::yield();
    }
}

template <typename T>
Resource<T> *ReadPromise<T>::get_resource() {
    return this->data;
}

template <typename T>
bool ReadPromise<T>::release() {
    std::atomic_thread_fence(std::memory_order_release);
    return this->data->release(*this);
}

template <typename T>
ReadPromise<T>::operator const T &() {
    std::atomic_thread_fence(std::memory_order_acquire);
    return *(this->get());
}

template <typename T>
bool ReadPromise<T>::operator==(const ReadPromise<T> &other) const {
    return this->id == other.id && this->data == other.data;
}

template <typename T>
bool ReadPromise<T>::is_exclusive() const {
    return false;
}

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)