/**
 * @file JobManager.hpp
 *
 * Contains class data for the JobManager data structure.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/Job.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

/**
 * @class JobManager
 *
 * Manages jobs.
 */
class JobManager final {
  private:
    /// Lists of running and waiting jobs.
    std::vector<std::shared_ptr<Job>> jobs;

    /// Whether the job manager is running or not.
    std::atomic_bool _is_running;

    /// Mutex for the job manager.
    std::mutex mutex;

    /// Pointer to the thread that is running the manager.
    std::thread *thread;

    /// Constructor.
    EINSUMS_EXPORT JobManager();

    /// No copy or move.
    JobManager(const JobManager &)  = delete;
    JobManager(const JobManager &&) = delete;

    EINSUMS_EXPORT ~JobManager();

    /**
     * Clean up the job manager on exit from program.
     */
    EINSUMS_EXPORT static void cleanup();

  protected: // I know protecting this is useless, but future-proofing never hurt anyone.
    /**
     * Main loop of the manager.
     */
    EINSUMS_EXPORT static void manager_loop();

    /**
     * One loop through the manager.
     */
    EINSUMS_EXPORT void manager_event();

  public:
    /**
     * Get the one single instance of the job manager.
     *
     * @return A reference to the single job manager.
     */
    EINSUMS_EXPORT static JobManager &get_singleton();

    /**
     * Queue a job. The job manager will take ownership of the job.
     *
     * @param The job to queue.
     * @return A weak pointer to the job. The job is now managed by the job manager and can no longer be destroyed safely by the caller.
     */
    template <typename U>
    std::shared_ptr<U> queue_job(U *__restrict__ job) {
        this->mutex.lock();
        job->set_state(detail::QUEUED);
        std::shared_ptr<U> &out = (std::shared_ptr<U> &) this->jobs.emplace_back(job); // Hint to the end of the list.
        this->mutex.unlock();

        return out;
    }

    /**
     * Start the job manager in a different thread. Raises an exception if it is already running.
     */
    EINSUMS_EXPORT void start_manager();

    /**
     * Stop the manager.
     */
    EINSUMS_EXPORT void stop_manager();

    /**
     * Check whether the manager is running or not.
     *
     * @return True if the manager is running. False if the manager is not running.
     */
    EINSUMS_EXPORT bool is_running();

    /**
     * Clear the job queue of all waiting jobs.
     */
    EINSUMS_EXPORT void clear_waiting();

    /**
     * Destroy the job queue. This deletes the singleton instance and stops the manager.
     */
    EINSUMS_EXPORT static void destroy();

    /**
     * Wait until all jobs are finished.
     */
    EINSUMS_EXPORT void wait_on_jobs();

    /**
     * Get the number of currently running jobs.
     *
     * @return The number of currently running jobs.
     */
    EINSUMS_EXPORT int running_jobs();

    /**
     * Get the number of jobs waiting to run.
     *
     * @return The number of jobs waiting to run.
     */
    EINSUMS_EXPORT int queued_jobs();

    /**
     * Get the number of running and waiting jobs. Equivalent to einsums::jobs::JobManager::running_jobs +
     * einsums::jobs::JobManager::queued_jobs.
     *
     * @return The number of jobs waiting and running.
     */
    EINSUMS_EXPORT int total_jobs();
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)