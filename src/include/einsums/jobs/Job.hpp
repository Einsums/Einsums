/**
 * @file Job.hpp
 *
 * Contains class data for the Job class.
 */

#pragma once

#include "einsums/_Common.hpp"

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

/**
 * @class Job
 *
 * Represents an abstract job.
 */
class Job {
  protected:
    int priority; /// Priority of the job. Will run before other jobs when it can.

  public:
    Job(int priority = 0) : priority(priority){};
    virtual ~Job() = default;

    /**
     * The function to run when the job is called.
     */
    virtual void run(void) { ; }

    /**
     * Whether the job is currently able to run.
     */
    virtual bool is_runnable() const { return false; }

    /**
     * Whether a job is running.
     */
    virtual bool is_running() const { return false; }

    /**
     * Whether the job is finished.
     */
    virtual bool is_finished() const { return true; }

    /**
     * Return the priority.
     */
    virtual int get_priority() const { return this->priority; }

    /**
     * Compare priorities.
     */
    virtual bool operator<(const Job &other) const { return this->get_priority() < other.get_priority(); }

    virtual bool operator<=(const Job &other) const { return this->get_priority() <= other.get_priority(); }

    virtual bool operator>(const Job &other) const { return this->get_priority() > other.get_priority(); }

    virtual bool operator>=(const Job &other) const { return this->get_priority() >= other.get_priority(); }

    virtual bool operator==(const Job &other) const { return this->get_priority() == other.get_priority(); }

    virtual bool operator!=(const Job &other) const { return this->get_priority() != other.get_priority(); }
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)