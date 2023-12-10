/**
 * @file Job.hpp
 *
 * Contains class data for the Job class.
 */

#pragma once

#include "einsums/_Common.hpp"

#include <atomic>
#include <functional>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

BEGIN_EINSUMS_NAMESPACE_HPP(detail)

/**
 * @enum JobState
 *
 * Represents the possible job states.
 */
enum JobState {
    /**
     * @var CREATED
     *
     * The job has been created only. This is the first state.
     */
    CREATED,

    /**
     * @var QUEUED
     *
     * The job has been added to the queue and is waiting to run.
     */
    QUEUED,

    /**
     * @var STARTING
     *
     * The job has had resources allocated for it and is waiting to be picked up by its threads.
     */
    STARTING,

    /**
     * @var RUNNING
     *
     * The job is now actively running.
     */
    RUNNING,

    /**
     * @var FINISHED
     *
     * The job has finished running.
     */
    FINISHED,

    /**
     * @var ERROR
     *
     * The job has encountered an error and has stopped.
     */
    ERROR
};

END_EINSUMS_NAMESPACE_HPP(detail)

/**
 * @class Job
 *
 * Represents an abstract job.
 */
class Job {
  protected:
    /**
     * @var curr_serial
     *
     * This holds a counter so that jobs have a serial identifier. It is instantiated in JobManager.cpp since there is no C++ file for this
     * empty virtual class.
     */
    EINSUMS_EXPORT static size_t curr_serial;

    /**
     * @var serialized_mutex
     *
     * This is a mutex that protects Job::curr_serial.
     */
    EINSUMS_EXPORT static std::mutex serialized_mutex;

    int priority; /// Priority of the job. Will run before other jobs when it can.

    size_t serial;

    detail::JobState curr_state;

    friend struct std::hash<Job>;

  public:
    Job(int priority = 0) : priority(priority), curr_state(detail::CREATED) {
        Job::serialized_mutex.lock();
        serial = Job::curr_serial;
        Job::curr_serial++;
        Job::serialized_mutex.unlock();
    };

    virtual ~Job() = default;

    /**
     * The function to run when the job is called.
     */
    virtual void run(void) { ; }

    /**
     * Get the current state of the job.
     */
    virtual detail::JobState get_state() const { return this->curr_state; }

    /**
     * Set the current job state.
     *
     * @param new_state The new state of the job.
     *
     * @return The new state of the job.
     */
    virtual detail::JobState set_state(detail::JobState new_state) {
        this->curr_state = new_state;

        return this->curr_state;
    }

    /**
     * Return any error that the job may have encountered.
     */
    virtual const std::exception &get_error() const { return *new std::exception(); }

    /**
     * Return the priority.
     */
    virtual int get_priority() const { return this->priority; }

    /**
     * Return the serial idenitfier.
     */
    virtual int get_serial() const { return this->serial; }

    /**
     * Use an id number to compare for equality.
     */
    virtual bool operator==(const Job &second) const { return this->serial == second.serial; }

    /**
     * Get number of threads requested.
     */
    virtual int num_threads() { return 1; }

    /**
     * Check whether the number of threads is a hard limit, or if fewer can be requested.
     */
    virtual bool can_have_fewer() { return false; }
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)

/**
 * @type std::hash<Job>
 *
 * Implementation of std::hash for the Job class. Note that this will be inserted into the std namespace, not the einsums::jobs namespace.
 */
template <>
struct std::hash<einsums::jobs::Job> {

    /**
     * Hash function operator for the job.
     */
    std::size_t operator()(const einsums::jobs::Job &job) { return job.serial; }
};
