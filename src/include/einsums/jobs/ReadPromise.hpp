/**
 * @file ReadPromise.hpp
 *
 * Contains the class data for the ReadPromise class.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/Resource.hpp"
#include "einsums/jobs/Timeout.hpp"

#include <chrono>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

template <typename T>
class Resource;

/**
 * @class ReadPromise
 *
 * A read-only promise on a resource. Allows for multiple jobs to access a resource at once.
 */
template <typename T>
class ReadPromise {
  protected:
    // ID for comparing different states of the same item.
    unsigned long id;

    // Pointer to data.
    Resource<T> *data;

  public:
    /**
     * Constructor.
     * @param id The unique ID of the lock. Used for resolving ownership.
     * @param data A pointer to the resource. The lock does not take ownership of the resource.
     */
    ReadPromise(unsigned long id, Resource<T> *data);

    /**
     * Delete the copy and move constructors.
     */
    ReadPromise(const ReadPromise<T> &)  = delete;
    ReadPromise(const ReadPromise<T> &&) = delete;

    /**
     * Destructor.
     */
    virtual ~ReadPromise();

    /**
     * Check if the lock has been resolved and can be used.
     */
    virtual bool ready();

    /**
     * Get the data that this promise wraps. Only gets the data when it is ready. This version has a timeout.
     *
     * @param time_out How long to wait. Throws an exception on timeout.
     * @return A reference to the data protected by this lock.
     * @throw An einsums::jobs::timeout exception on timeout.
     */
    template <typename Inttype, typename Ratio>
    const std::shared_ptr<T> get(std::chrono::duration<Inttype, Ratio> time_out);

    /**
     * Get the data that this lock protects. Only gets the data when it is ready. Does not have a timeout.
     *
     * @return A reference to the data protected by this lock.
     */
    const std::shared_ptr<T> get();

    /**
     * Wait until the data is ready. Has a timeout.
     *
     * @param time_out How long to wait. Throws an exception on time-out.
     * @throw An einsums::jobs::timeout exception on timeout.
     */
    template <typename Inttype, typename Ratio>
    void wait(std::chrono::duration<Inttype, Ratio> time_out);

    /**
     * Wait until the data is ready. Blocks without a timeout.
     */
    void wait();

    /**
     * Get a pointer to the resource that produced this lock.
     *
     * @return A pointer to a resource.
     */
    Resource<T> *get_resource();

    /**
     * Release this lock. It will never be able to be resolved after this.
     *
     * @return True if the promise is successfully released, false if it was not. For instance, if it was already removed.
     */
    bool release();

    /**
     * Get the data contained in the resource. This is a wrapper around einsums::jobs::Resource::get.
     *
     * @return A const reference to the data held by the promise.
     */
    explicit operator const T &();

    /**
     * Compare two locks to see if they are the same.
     *
     * @return True if the two promises are the same.
     */
    bool operator==(const ReadPromise<T> &other) const;
    /**
     * Whether a lock is exclusive or not.
     *
     * @return False for an einsums::jobs::ReadPromise.
     */
    [[nodiscard]] virtual bool is_exclusive() const;
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)

#include "einsums/jobs/templates/ReadPromise.tpp"