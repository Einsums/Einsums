/**
 * @file ThreadPool.hpp
 *
 * Contains class data for the ThreadPool data structure.
 */

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/jobs/Job.hpp"

#include <atomic>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

/**
 * @class ThreadPool
 *
 * Pools threads together for running computations.
 */
class ThreadPool {

  public:
    /**
     * @type The type for a function to be run.
     */
    using function_type = void (*)(std::weak_ptr<Job> &);

  private:
    /**
     * @var stop
     *
     * The stop condition. If it is false, then the threads will keep running. When it is true, the threads will stop.
     */
    /**
     * @var exists
     *
     * Whether the thread pool exists. This is normally true. It is only false when the destructor is in the process of running.
     */
    std::atomic_bool stop, exists;

    /**
     * @var mutex
     *
     * The mutex protecting the thread pool.
     */
    std::mutex mutex;

    /**
     * @var max_threads
     *
     * The maximum number of threads to spawn.
     */
    int max_threads;

    /**
     * @var threads
     *
     * A list of the threads that have been spawned by the thread pool.
     */
    std::vector<std::shared_ptr<std::thread>> threads;

    /**
     * @var avail
     *
     * A list of threads that don't have jobs running currently.
     */
    /**
     * @var running
     *
     * A list of threads that are currently running jobs.
     */
    std::vector<std::weak_ptr<std::thread>> avail, running;

    /**
     * @var thread_info
     *
     * Information to pass to each of the threads. There are special values for stopping a thread and indicating that the thread should be
     * idle.
     */
    std::map<std::thread::id, std::tuple<int, int, function_type, std::weak_ptr<Job>>> thread_info;

    /**
     * Constructor. Just sets the maximum number of threads.
     */
    EINSUMS_EXPORT ThreadPool(int threads);

    /**
     * No copy or move constructors.
     */
    ThreadPool(const ThreadPool &)  = delete;
    ThreadPool(const ThreadPool &&) = delete;

    /**
     * Destructor.
     */
    EINSUMS_EXPORT ~ThreadPool();

    /**
     * Spawn a number of threads.
     */
    EINSUMS_EXPORT void init_pool(int threads);

  public:
    /**
     * Initialize the thread pool.
     *
     * @param threads The number of threads to hold.
     */
    EINSUMS_EXPORT static void init(int threads);

    /**
     * Destroy the thread pool.
     */
    EINSUMS_EXPORT static void destroy();

    /**
     * Return the singleton instance. The thread pool needs to be initialized first.
     */
    EINSUMS_EXPORT static ThreadPool &get_singleton();

    /**
     * Return whether the singleton instance is constructed.
     */
    EINSUMS_EXPORT static bool singleton_exists();

    /**
     * Request a set number of threads. Gives an empty list if there are not enough threads available.
     *
     * @param count The number of threads to request.
     * @return The number of threads requested, or zero if the request could not be fulfilled.
     */
    EINSUMS_EXPORT int request(unsigned int count, function_type func, std::weak_ptr<Job> &job);

    /**
     * Request up to a set number of resources. May give fewer than the requested.
     *
     * @param count The maximum number of resources to request.
     * @return The number of threads obtained.
     */
    EINSUMS_EXPORT int request_upto(unsigned int count, function_type func, std::weak_ptr<Job> &job);

    /**
     * Release certain threads.
     *
     * @param threads A vector of pointers to threads to release.
     */
    EINSUMS_EXPORT void release(std::vector<std::weak_ptr<std::thread>> &threads);

    /**
     * Get the index of a thread in a compute kernel.
     *
     * @param id A thread id.
     * @return The index of the thread in the calculation.
     */
    EINSUMS_EXPORT int index(std::thread::id id);

    /**
     * Get the index of a thread in a compute kernel.
     *
     * @param thread The thread to check.
     * @return The index of the thread in the calculation.
     */
    EINSUMS_EXPORT int index(const std::thread &thread);

    /**
     * Get the kernel size for a computation.
     *
     * @param id A thread id.
     * @return The number of threads in a computation.
     */
    EINSUMS_EXPORT int compute_threads(std::thread::id id);

    /**
     * Get the kernel size for a computation.
     *
     * @param thread The thread to check.
     * @return The number of threads in a computation.
     */
    EINSUMS_EXPORT int compute_threads(const std::thread &thread);

    /**
     * Get the function that a thread should run.
     *
     * @param id A thread id.
     * @return The function a thread should run.
     */
    EINSUMS_EXPORT function_type compute_function(std::thread::id id);

    /**
     * Get the function that a thread should run.
     *
     * @param thread The thread to check.
     * @return The function a thread should run.
     */
    EINSUMS_EXPORT function_type compute_function(const std::thread &thread);

    /**
     * Get the job associated with a thread.
     *
     * @param id A thread id.
     * @return A pointer to the job.
     */
    EINSUMS_EXPORT std::weak_ptr<Job> &thread_job(std::thread::id id);

    /**
     * Get the job associated with a thread.
     *
     * @param thread The thread to check.
     * @return A pointer to the job.
     */
    EINSUMS_EXPORT std::weak_ptr<Job> &thread_job(const std::thread &thread);

    /**
     * Signal that a thread has finished.
     *
     * @param id A thread id.
     */
    EINSUMS_EXPORT void thread_finished(std::thread::id id);

    /**
     * Signal that a thread has finished.
     *
     * @param thread The thread to check.
     */
    EINSUMS_EXPORT void thread_finished(const std::thread &thread);

    /**
     * Returns true when the threads are supposed to stop.
     *
     * @return True if the threads should stop. False if the threads should continue.
     */
    EINSUMS_EXPORT bool stop_condition();

    /**
     * Returns how many threads are working on jobs.
     *
     * @return The number of currently running threads.
     */
    EINSUMS_EXPORT int has_running();

    /**
     * Waits until all jobs are finished running.
     */
    EINSUMS_EXPORT void wait_on_running();
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)