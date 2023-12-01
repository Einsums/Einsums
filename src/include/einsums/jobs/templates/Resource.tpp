/**
 * @file Resource.tpp
 *
 * Contains definitions of methods in the Resource class from Jobs.hpp.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/Resource.hpp"

#include <mutex>
#include <vector>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

template <typename T>
template <typename... Args>
Resource<T>::Resource(Args &&...args) : locks{}, id(0), mutex() {
    this->data = std::make_shared<T>(args...);
}

template <typename T>
Resource<T>::~Resource() {
    mutex.lock();
    for (auto state : this->locks) {
        state->clear();
    }
    this->locks.clear();
    this->data.reset();
}

template <typename T>
std::shared_ptr<T> Resource<T>::get_data() {
    return std::shared_ptr<T>(this->data);
}

template <typename T>
std::shared_ptr<ReadPromise<T>> Resource<T>::read_promise() {
    // wait to be allowed to edit the resource.
    this->mutex.lock();

    // Make sure there is somewhere to put the locks.
    if (this->locks.size() == 0) {
        this->locks.push_back(new std::vector<std::shared_ptr<ReadPromise<T>>>());
    } else if (this->locks.back()->size() == 1 && this->locks.back()->at(0)->is_exclusive()) {
        this->locks.push_back(new std::vector<std::shared_ptr<ReadPromise<T>>>());
    }

    // Make the lock.
    std::shared_ptr<ReadPromise<T>> out = std::make_shared<ReadPromise<T>>(this->id, this);

    // Increment the serial tracker.
    this->id++;

    // Add the lock.
    this->locks.back()->push_back(out);

    // Release the resource.
    this->mutex.unlock();
    return out;
}

template <typename T>
std::shared_ptr<WritePromise<T>> Resource<T>::write_promise() {
    this->mutex.lock();

    this->locks.push_back(new std::vector<std::shared_ptr<ReadPromise<T>>>());

    std::shared_ptr<WritePromise<T>> out = std::make_shared<WritePromise<T>>(this->id, this);
    this->id++;

    this->locks.back()->push_back(out);

    this->mutex.unlock();
    return out;
}

template <typename T>
bool Resource<T>::release(const ReadPromise<T> &lock) {
    bool ret = false;
    this->mutex.lock();

    for (auto state : this->locks) {
        size_t i = 0;
        while(i < state->size()) {
            if (*(state->at(i)) == lock) {
                state->erase(std::next(state->begin(), i));
                ret = true;
            } else {
                i++;
            }
        }
    }

    // Remove empty states.
    size_t i = 0;
    while(i < this->locks.size()) {
        if (this->locks[i]->empty()) {
            this->locks.erase(std::next(this->locks.begin(), i));
        } else {
            i++;
        }
    }

    // Release the lock on the lock.
    this->mutex.unlock();

    return ret;
}

template <typename T>
bool Resource<T>::is_open() {
    this->mutex.lock();

    if (this->locks.empty()) {
        this->mutex.unlock();
        return true;
    }
    if (this->locks.size() == 1 && this->locks[0].empty()) {
        this->mutex.unlock();
        return true;
    }
    this->mutex.unlock();
    return false;
}

template <typename T>
bool Resource<T>::is_promised(const ReadPromise<T> &lock) {
    this->mutex.lock();

    for (auto state : this->locks) {
        for (const auto &curr_lock : *state) {
            if (curr_lock == lock) {
                this->mutex.unlock();
                return true;
            }
        }
    }

    this->mutex.unlock();
    return false;
}

template <typename T>
bool Resource<T>::is_readable(const ReadPromise<T> &promise) {
    this->mutex.lock();

    if (this->locks.size() == 0) {
        this->mutex.unlock();
        return false;
    }

    for (const auto &curr_lock : *(this->locks[0])) {
        if (*curr_lock == promise) {
            this->mutex.unlock();
            return true;
        }
    }

    this->mutex.unlock();
    return false;
}

template <typename T>
bool Resource<T>::is_writable(const ReadPromise<T> &promise) {

    if (!promise.is_exclusive()) {
        return false; // The lock is not a writable lock. It will never be a writable lock.
    }

    this->mutex.lock();

    if (this->locks.size() == 0) {
        this->mutex.unlock();
        return false; // No locks given.
    }

    if (this->locks[0]->size() != 1) {
        this->mutex.unlock();
        return false; // The state is a read-only state.
    }

    if (*(this->locks[0]->at(0)) == promise) {
        this->mutex.unlock();
        return true; // This lock has sole ownership.
    }

    this->mutex.unlock();
    return false;
}

template <typename T>
void Resource<T>::clear() {
    this->mutex.lock();
    this->data.reset();

    this->mutex.unlock();
}

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)