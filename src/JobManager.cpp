/**
 * @file JobManager.cpp
 *
 * This file contains definitions for the functions and methods defined in Jobs.hpp.
 */

#include "einsums/Jobs.hpp"
#include "einsums/jobs/Job.hpp"

#include <atomic>
#include <cstdio>
#include <thread>

/**
 * @def debug(fmt, ...)
 *
 * Print debug information, but only if NDEBUG is undefined.
 */
#ifndef NDEBUG
#    define debug(fmt, ...)                                                                                                                \
        std::fprintf(stderr, fmt __VA_OPT__(, ) __VA_ARGS__);                                                                              \
        std::fflush(stderr)
#else
#    define debug(fmt, args...)
#endif

using namespace einsums::jobs;

// Putting this here since it is needed here.
size_t Job::curr_serial = 0;
std::mutex Job::serialized_mutex = std::mutex();

/**
 * @var instance
 *
 * Single instance to the job manager.
 */
static JobManager *mgr_instance = nullptr;

/**
 * @var added_job_exit_handler
 *
 * Whether the atexit handler has been added to stop and delete the job manager when exit is called, or the program returns from main.
 */

static bool added_job_exit_handler = false;

JobManager::JobManager() : jobs{}, mutex(), _is_running(false), thread(nullptr) {
    if (!added_job_exit_handler) {
        std::atexit(JobManager::cleanup);
        added_job_exit_handler = true;
    }
}

JobManager::~JobManager() {
    debug("Deleting job manager.\n");
    this->_is_running = false;
    if (this->thread != nullptr) {
        this->thread->join();
        delete this->thread;
    }
    this->jobs.clear();
}

void JobManager::cleanup() {
    if (mgr_instance != nullptr) {
        delete mgr_instance;
    }
    mgr_instance = nullptr;
}

void JobManager::manager_loop() {
    // Infinite loop.
    JobManager &inst = JobManager::get_singleton();
    inst._is_running = true;
    while (inst._is_running) {
        inst.manager_event(); // Process an event.
        std::atomic_thread_fence(std::memory_order_acq_rel);
        std::this_thread::yield(); // Yield for another thread;
    }
    return;
}

/**
 * Runs a job.
 *
 * @param job The job to run.
 */
static void run_job(std::weak_ptr<Job> &job) {
    std::atomic_thread_fence(std::memory_order_acq_rel);
    job.lock()->run();

    std::atomic_thread_fence(std::memory_order_acq_rel);

    // Job finished.
    if (ThreadPool::singleton_exists()) {
        ThreadPool::get_singleton().thread_finished(std::this_thread::get_id());
    }
}


void JobManager::manager_event() {
    // Obtain a lock on the manager.
    this->mutex.lock();

    if (!ThreadPool::singleton_exists()) {
        this->mutex.unlock();
        return;
    }

    // Go through each of the waiting jobs and try to queue them up.
    std::atomic_thread_fence(std::memory_order_acquire);

    for(ssize_t i = 0; i < this->jobs.size(); i++) {
        if(this->jobs[i]->get_state() == detail::QUEUED) {
            auto job_ptr = std::weak_ptr<Job>(this->jobs[i]);
            ThreadPool::get_singleton().request(1, run_job, job_ptr);
        }
    }

    // Go through jobs and remove finished ones.
    ssize_t i = 0;
    while(i < this->jobs.size()) {
        if(this->jobs[i]->get_state() == detail::FINISHED || this->jobs[i]->get_state() == detail::ERROR) {
            this->jobs.erase(std::next(this->jobs.begin(), i));
        } else {
            i++;
        }
    }

    this->mutex.unlock();
}

JobManager &JobManager::get_singleton() {
    std::atomic_thread_fence(std::memory_order_acq_rel);
    if (mgr_instance == nullptr) {
        mgr_instance = new JobManager();
    }
    return *mgr_instance;
}



void JobManager::start_manager() {
    this->mutex.lock();

    if (this->_is_running) {
        throw std::runtime_error("Job manager already running!");
    }

    this->_is_running = true;

    // Start the thread.
    if (this->thread == nullptr) {
        this->thread = new std::thread(this->manager_loop);
    }

    this->mutex.unlock();
}

void JobManager::stop_manager() {
    this->mutex.lock();

    this->_is_running = false;
    this->mutex.unlock();

    this->thread->join();

    this->mutex.lock();

    delete this->thread;
    this->thread = nullptr;

    this->mutex.unlock();
}

bool JobManager::is_running() {

    return this->_is_running;
}

void JobManager::clear_waiting() {
    this->mutex.lock();

    ssize_t i = 0;
    while(i < this->jobs.size()) {
        if(this->jobs[i]->get_state() == detail::QUEUED) {
            this->jobs.erase(std::next(this->jobs.begin(), i));
        } else {
            i++;
        }
    }

    this->mutex.unlock();
}

void JobManager::destroy() {
    if (mgr_instance != nullptr) {
        delete mgr_instance;
        mgr_instance = nullptr;
    }
}

void JobManager::wait_on_jobs() {
    while(!this->jobs.empty()) {
        std::this_thread::yield();
    }
}

int JobManager::running_jobs() {
    this->mutex.lock();

    int ret = 0;
    
    for(ssize_t i = 0; i < this->jobs.size(); i++) {
        if(this->jobs[i]->get_state() == detail::STARTING || this->jobs[i]->get_state() == detail::RUNNING) {
            ret++;
        }
    }

    this->mutex.unlock();
    return ret;
}

int JobManager::queued_jobs() {
    this->mutex.lock();

    int ret = 0;
    
    for(ssize_t i = 0; i < this->jobs.size(); i++) {
        if(this->jobs[i]->get_state() == detail::QUEUED) {
            ret++;
        }
    }

    this->mutex.unlock();
    return ret;
}

int JobManager::total_jobs() {
    this->mutex.lock();

    int ret = this->jobs.size();
    
    this->mutex.unlock();
    return ret;
}
