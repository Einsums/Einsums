/**
 * @file Jobs.hpp
 *
 * Job queues and other things.
 */

#pragma once

#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <atomic>
#include <stdexcept>

#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

EINSUMS_BEGIN_NAMESPACE_HPP(einsums::jobs)

/**
 * @var check_rate
 *
 * The time between checks for blocked jobs.
 */
static constexpr std::chrono::milliseconds check_rate = std::chrono::milliseconds(20);

struct timeout : public std::except {
private:
  static constexpr char message[] = "Timeout";
public:
  timeout() noexcept : std::except() {}

  timeout(const timeout &other) noexcept : std::except(other) {}

  ~timeout() = default;

  const char *what() noexcept {
    return timeout::message;
  }

};

template<typename T> class Resource;

template<typename T>
class ReadLock {
protected:
  // ID for comparing different states of the same item.
  unsigned long id;

  // Pointer to data.
  Resource<T> *data;
  
public:

  /**
   * Constructor.
   */
  ReadLock(unsigned long id, Resource<T> *data) : id(id), data(data) {}

  /**
   * Delete the copy and move constructors.
   */
  ReadLock(const ReadLock<T> &) = delete;
  ReadLock(const ReadLock<T> &&) = delete;

  /**
   * Destructor.
   */
  virtual ~ReadLock() {
    this->data = nullptr; // Data is owned by someone else.
  }

  /**
   * Check if the lock has been resolved and can be used.
   */
  bool ready(void) {
    return this->data->ready(*this);
  }

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @param timeout How long to wait. Throws an exception on time-out.
   * @return A reference to the data protected by this lock.
   */
  const T &get(std::chrono::duration timeout) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while(!this->ready()) {
      std::this_thread::sleep_for(check_rate);

      std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
      if(curr - start >= timeout) {
	throw(*new timeout());
      }
    }

    return this->data->get_data();
  }

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @return A reference to the data protected by this lock.
   */
  const T &get(void) {
    while(!this->ready()) {
      std::this_thread::sleep_for(check_rate);
    }
    return this->data->get_data();
  }

  /**
   * Wait until the data is ready.
   *
   * @param timeout How long to wait. Throws an exception on time-out.
   */
  void wait(std::chrono::duration timeout) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while(!this->ready()) {
      std::this_thread::sleep_for(check_rate);

      std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
      if(curr - start >= timeout) {
	throw(*new timeout());
      }
    }
  }

  /**
   * Wait until the data is ready.
   */
  void wait(void) {
    while(!this->ready()) {
      std::this_thread::sleep_for(check_rate);
    }
  }

  /**
   * Get a pointer to the resource that produced this lock.
   */
  Resource<T> *get_resource(void) {
    return this->data;
  }

  /**
   * Release this lock.
   */
  bool release(void) {
    return this->data->release(*this);
  }

  explicit operator const T&(void) {
    return this->get();
  }

  bool operator==(ReadLock<T> &other) {
    return this->id == other.id && this->data == other.data;
  }

  virtual bool is_exclusive() {
    return false;
  }
};

/**
 * @class WriteLock<T>
 *
 * Represents an exclusive lock.
 */
template<T>
class WriteLock : public ReadLock<T> {
public:
  /**
   * Constructor.
   */
  WriteLock(unsigned long id, Resource<T> *data) : ReadLock<T>(id, data) {}

  WriteLock(const WriteLock<T> &) = delete;
  WriteLock(const WriteLock<T> &&) = delete;
  
  virtual ~WriteLock() = default;

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @param timeout How long to wait. Throws an exception on time-out.
   * @return A reference to the data protected by this lock.
   */
  T &get(std::chrono::duration timeout) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while(!this->ready()) {
      std::this_thread::sleep_for(check_rate);

      std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
      if(curr - start >= timeout) {
	throw(*new timeout());
      }
    }

    return this->data->get_data();
  }

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @return A reference to the data protected by this lock.
   */
  T &get(void) {
    while(!this->ready()) {
      std::this_thread::sleep_for(check_rate);
    }
    return this->data->get_data();
  }

  explicit operator T&(void) {
    return this->get();
  }

  bool is_exclusive() override {
    return true;
  }
};


/**
 * @class Resource<T>
 *
 * Represents a resource with locks.
 */
template<typename T>
class Resource {
private:
  std::vector<std::vector<std::shared_ptr<ReadLock<T> > *> locks;

  unsigned long id;

  T *data;

  std::atomic_bool is_locked;

public:

  Resource(T *data) : locks{}, id(0), data(data), is_locked(false) {}

  Resource(const Resource<T> &) = delete;
  Resource(const Resource<T> &&) = delete;

  virtual ~Resource() {
    for(auto state : this->locks) {
      for(auto lock : *state) {
	delete lock;
      }
      state->clear();
    }
    this->locks.clear();
  }

  /**
   * Obtain a shared lock.
   *
   * @return A pointer to the shared lock.
   */
  std::shared_ptr<ReadLock<T>> lock_shared() {
    while(this->is_locked.load()) {
      std::this_thread::sleep_for(check_rate);
    }
    this->is_locked = true;
    if(this->locks.size() == 0) {
      this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>());
    } else if((this->locks.end())->size() == 1 && (this->locks.end())->at(0)->is_exclusive()) {
      this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>());
    }

    std::shared_ptr<ReadLock<T>> out = std::make_shared<ReadLock<T>>(this->id, this);

    this->id++;

    (this->locks.end())->push_back(out);
    this->is_locked = false;
    return out;
  }

  std::shared_ptr<WriteLock<T>> lock() {
    while(this->is_locked.load()) {
      std::this_thread::sleep_for(check_rate);
    }
    this->is_locked = true;
    this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>());

    std::shared_ptr<WriteLock<T>> out = std::make_shared<WriteLock<T>>(this->id, this);
    this->id++;

    (this->locks.end())->push_back(out);
    this->is_locked = false;
    return out;
  }

  /**
   * Release a lock.
   *
   * @param The lock to release.
   */
  bool release(const ReadLock<T> &lock) {
    while(this->is_locked.load()) {
      std::this_thread::sleep_for(check_rate);
    }
    this->is_locked = true;
    for(auto state : this->locks) {
      for(auto lock : *state) {
	if(
  
  


EINSUMS_END_NAMESPACE_HPP(einsums::jobs)
