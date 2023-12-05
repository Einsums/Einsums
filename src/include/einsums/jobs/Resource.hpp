/**
 * @file Resource.hpp
 *
 * Contains class data for the Resource class.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/ReadPromise.hpp"
#include "einsums/jobs/WritePromise.hpp"

#include <mutex>
#include <vector>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

template <typename T>
class ReadPromise;

template <typename T>
class WritePromise;

/**
 * @class Resource<T>
 *
 * Represents a resource with locks.
 */
template <typename T>
class Resource {
  private:
    /**
     * A list of states. Each state is a list of locks that can read or write each state.
     */
    std::vector<std::vector<std::shared_ptr<ReadPromise<T>>>> locks;

    /**
     * Internal counter. It ensures that each promies produced will have a unique identifier.
     */
    unsigned long id;

    /// A pointer to the data managed by the resource.
    std::shared_ptr<T> data;

  protected:
    /// A mutex to help eliminate data races.
    std::mutex mutex;

    /**
     * Check each of the locks held to make sure that they are still viable, and remove the ones that are not.
     */
    void update_locks();

  public:
    /**
     * Make a new resource constructed with the given arguments. The arguments passed will be passed along to the constructor for the
     * managed data type.
     */
    template <typename... Args>
    Resource(Args &&...args);

    /**
     * Don't allow copy or move.
     */
    Resource(const Resource<T> &)  = delete;
    Resource(const Resource<T> &&) = delete;

    /**
     * Destructor.
     */
    virtual ~Resource();

    /**
     * Return a shared pointer to the data.
     *
     * @return A shared pointer that has shared ownership on the data managed by this resource.
     */
    std::shared_ptr<T> get_data();

    /**
     * Obtain a read promise.
     *
     * @return A shared pointer to the read promise.
     */
    std::shared_ptr<ReadPromise<T>> read_promise();

    /**
     * Obtain an exclusive write promise.
     *
     * @return a shared pointer to the write promise.
     */
    std::shared_ptr<WritePromise<T>> write_promise();

    /**
     * Release a promise.
     *
     * @param The promise to release.
     * @return true if a lock was released, false if no lock was found.
     */
    bool release(const ReadPromise<T> &lock);

    /**
     * Tests whether this resource is completely unlocked.
     *
     * @return True if no locks are held, false if some locks are held.
     */
    bool is_open();

    /**
     * Tests whether a promised lock to this resource is held by the given lock. Does not tell whether the promise has ownership and can
     * read or modify. Only tells whether the promise will be honored at some point.
     *
     * @param lock The lock to test.
     * @return True if a lock is held. False if a lock is not held.
     */
    bool is_promised(const ReadPromise<T> &lock);

    /**
     * Test if the given promise allows reading.
     *
     * @param promise The promise to test.
     * @return True if a lock is held and currently allows reading. False if the promise does not allow reading, or is not held.
     */
    bool is_readable(const ReadPromise<T> &promise);

    /**
     * Check whether a given promise is available for writing.
     *
     * @param promise The promise to test.
     * @return True if the given lock currently allows writing. False if the current promise is not available for writing.
     */
    bool is_writable(const ReadPromise<T> &promise);

    /**
     * Clear the memory. Useful if the memory is owned by another scope.
     */
    void clear();
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)

#include "einsums/jobs/templates/Resource.tpp"