/**
 * @file WritePromise.hpp
 *
 * Contains class data for the WritePromise class.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/ReadPromise.hpp"
#include "einsums/jobs/Resource.hpp"

#include <chrono>
#include <thread>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

template <typename T>
class ReadPromise;

template <typename T>
class Resource;

/**
 * @class WriteLock<T>
 *
 * Represents an exclusive read-write lock.
 */
template <typename T>
class WritePromise : public ReadPromise<T> {
  public:
    /**
     * Constructor.
     */
    WritePromise(unsigned long id, Resource<T> *data);

    /// Delete the copy and move constructors.
    WritePromise(const WritePromise<T> &)  = delete;
    WritePromise(const WritePromise<T> &&) = delete;

    /// Default destructor.
    virtual ~WritePromise() = default;

    /**
     * Get the data that this lock protects. Only gets the data when it is ready.
     *
     * @param time_out How long to wait. Throws an exception on time-out.
     * @return A reference to the data protected by this lock.
     * @throw An einsums::jobs::timeout exception on timeout.
     */
    template <typename Inttype, typename Ratio>
    std::shared_ptr<T> get(std::chrono::duration<Inttype, Ratio> time_out);

    /**
     * Get the data that this lock protects. Only gets the data when it is ready.
     *
     * @return A reference to the data protected by this lock.
     */
    std::shared_ptr<T> get();

    /**
     * A non-const cast. This is a wrapper around einsums::jobs::WritePromise::get.
     *
     * @return A reference to the data protected by this lock.
     */
    explicit operator T &();

    /**
     * Tells whether the promise is exclusive.
     *
     * @return True for an einsums::jobs::WritePromise.
     */
    [[nodiscard]] bool is_exclusive() const override;

    /**
     * Tells whether the promise is ready for access.
     *
     * @return True if the promise has a lock. False if it does not.
     */
    bool ready() override;
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)

#include "einsums/jobs/templates/WritePromise.tpp"
