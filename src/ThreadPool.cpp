/**
 * @file ThreadPool.cpp
 *
 * Contains definitions for the methods defined by the ThreadPool class.
 */

#include "einsums/Jobs.hpp"
#include "einsums/jobs/Job.hpp"

#include <atomic>
#include <cstdio>
#include <thread>

using namespace einsums;
using namespace einsums::jobs;

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

/**
 * @var thread_instance
 *
 * Single instance to the thread pool.
 */
static ThreadPool *thread_instance = nullptr;

/**
 * @var added_thread_exit_handler
 *
 * Whether the atexit handler has been added to stop and delete the thread pool when exit is called or the program returns from main.
 */
static bool added_thread_exit_handler = false;

/**
 * The main loop for the threads in the thread pool.
 */
static void thread_loop() {
    // Wait for the info to be active. Avoid a race condition.
    std::this_thread::yield();
    while (true) {
        try {
            ThreadPool::get_singleton().index(std::this_thread::get_id());
        } catch (std::exception &exc) {
            std::this_thread::yield();
            continue;
        }
        break;
    }

    // -1 is the stop condition.
    ThreadPool &instance = ThreadPool::get_singleton();
    while (ThreadPool::singleton_exists() && !instance.stop_condition()) {
        // Check for a job.
        volatile auto func = instance.compute_function(std::this_thread::get_id());

        if (func == nullptr) {
            // No job. Wait for a bit and check again.
            std::this_thread::yield();
            std::atomic_thread_fence(std::memory_order_acq_rel);
            continue;
        }

        // Found a job. Run it.
        func(instance.thread_job(std::this_thread::get_id()));

        // Job is finished. Tell the pool.
        if (ThreadPool::singleton_exists()) {
            instance.thread_finished(std::this_thread::get_id());
        }
        std::atomic_thread_fence(std::memory_order_acq_rel);
    }
}

ThreadPool::ThreadPool(int threads) : mutex(), max_threads(threads), avail{}, running{}, thread_info{}, threads{}, exists(true) {
    if (!added_thread_exit_handler) {
        std::atexit(ThreadPool::destroy);
        added_thread_exit_handler = true;
    }

    this->stop = false;
}

ThreadPool::~ThreadPool() {
    debug("Deleting thread pool.\n");

    thread_info.clear();
    avail.clear();
    running.clear();
    threads.clear();
    max_threads = 0;

    debug("Finished deleting thread pool.\n");
}

void ThreadPool::init(int threads) {
    if (thread_instance != nullptr) {
        throw std::runtime_error("Double initialization of the thread pool!");
    } else {
        thread_instance = new ThreadPool(threads);
        thread_instance->init_pool(threads);
    }
}

void ThreadPool::init_pool(int thread) {
    this->mutex.lock();
    for (int i = 0; i < thread; i++) {
        this->threads.push_back(std::make_shared<std::thread>(thread_loop));
        this->avail.push_back(this->threads[this->threads.size() - 1]);

        this->thread_info[this->threads[this->threads.size() - 1]->get_id()] =
            std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{0, 0, nullptr, nullptr};
    }
    this->mutex.unlock();
}

void ThreadPool::destroy() {
    debug("Destroying thread pool.\n");

    if (thread_instance != nullptr) {
        // Send the stop signal.
        thread_instance->mutex.lock();
        thread_instance->stop = true;
        thread_instance->mutex.unlock();

        // Wait for all threads to stop.
        while (thread_instance->has_running()) {
            std::this_thread::yield();
        }
        thread_instance->mutex.lock();
        thread_instance->exists = false;

        for (auto &kv : thread_instance->thread_info) {
            auto &val = std::get<1>(kv);

            std::get<0>(val) = -1;
            std::get<1>(val) = -1;
            std::get<2>(val) = nullptr;
            std::get<3>(val) = std::weak_ptr<Job>();
        }

        thread_instance->mutex.unlock();

        for (auto &thr : thread_instance->threads) {
            while (!thr->joinable()) {
                std::this_thread::yield();
            }
            thr->join();
        }
        thread_instance->mutex.lock();
        delete thread_instance;
        thread_instance = nullptr;
    }
}

ThreadPool &ThreadPool::get_singleton() {
    if (thread_instance == nullptr) {
        throw std::runtime_error("Thread pool needs to be initialized!");
    }
    return *thread_instance;
}

bool ThreadPool::singleton_exists() {
    return thread_instance != nullptr && thread_instance->exists;
}

int ThreadPool::request(unsigned int count, ThreadPool::function_type func,
                                                             std::weak_ptr<Job> &job) {
    this->mutex.lock();

    // Check that the number of threads can reasonably be allocated.
    if (count > this->max_threads) {
        this->mutex.unlock();
        throw std::runtime_error("Could not allocate threads! Requested too many.");
    }

    // Check that there are enough threads to satisfy the request.
    if(count >= this->avail.size()) {
        this->mutex.unlock();
        return 0;
    }

    for (int i = 0; i < count; i++) {
        debug("7: Running size: %ld\n", this->running.size());
        // Get the new thread.
        std::weak_ptr<std::thread> thread = this->avail.back();
        for (auto &thr : this->threads) {
            if (thr->get_id() == thread.lock()->get_id()) {
                this->running.push_back(thr);
                this->avail.pop_back();
                break;
            }
        }

        // Push the thread info.
        (this->thread_info)[thread.lock()->get_id()] =
            std::tuple<int, int, ThreadPool::function_type, std::weak_ptr<Job>>{i, count, func, job};
    }

    assert(this->threads.size() == this->max_threads);
    assert(this->avail.size() + this->running.size() == this->threads.size());

    job.lock()->set_state(detail::STARTING);

    this->mutex.unlock();

    return count;
}

int ThreadPool::request_upto(unsigned int count, ThreadPool::function_type func,
                                                                  std::weak_ptr<Job> &job) {
    this->mutex.lock();
    auto out_threads = std::vector<std::weak_ptr<std::thread>>();

    int total = 0;

    for (int i = 0; i < count; i++) {
        debug("6: Running size: %ld\n", this->running.size());
        // Check that there are threads available.
        if (this->avail.size() == 0) {
            break;
        }
        // Get the new thread.
        std::weak_ptr<std::thread> thread = this->avail.back();
        for (auto &thr : this->threads) {
            if (thr->get_id() == thread.lock()->get_id()) {
                this->running.push_back(thr);
                out_threads.push_back(thr);
                this->avail.pop_back();
                break;
            }
        }
        total++;
    }

    // Push the thread info.
    for (int i = 0; i < total; i++) {
        (this->thread_info)[out_threads[i].lock()->get_id()] =
            std::tuple<int, int, ThreadPool::function_type, std::weak_ptr<Job>>{i, total, func, job};
    }

    assert(this->threads.size() == this->max_threads);
    assert(this->avail.size() + this->running.size() == this->threads.size());

    job.lock()->set_state(detail::STARTING);

    this->mutex.unlock();
    
    return total;
}

void ThreadPool::release(std::vector<std::weak_ptr<std::thread>> &threads) {
    this->mutex.lock();

    // Move running threads back to the waiting pool.
    for (const auto &thread : threads) {
        ssize_t i = 0;
        while (this->running.size() < this->max_threads && i < this->running.size()) {
            debug("4: Size of running: %ld\n", this->running.size());
            if (this->running.at(i).lock()->get_id() == thread.lock()->get_id()) {
                for (ssize_t j = 0; j < this->threads.size(); j++) {
                    if (this->running[i].lock()->get_id() == this->threads[j]->get_id()) {
                        this->avail.push_back(this->threads[j]);
                        break;
                    }
                }
                (this->thread_info)[(this->running)[i].lock()->get_id()] =
                    std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{0, 0, nullptr, nullptr};
                size_t before = this->running.size();
                if (before <= 0 || std::next(this->running.begin(), i) == this->running.end()) {
                    debug("5: Out of bounds oopsie.\n");
                }
                this->running.erase(this->running.begin() + i);
                assert(before - 1 == this->running.size());
            } else {
                i++;
            }
        }
    }

    this->mutex.unlock();
    assert(this->threads.size() == this->max_threads);
    assert(this->avail.size() + this->running.size() == this->threads.size());
}

int ThreadPool::index(std::thread::id id) {
    this->mutex.lock();

    int out = std::get<0>((this->thread_info)[id]);

    this->mutex.unlock();
    return out;
}

int ThreadPool::compute_threads(std::thread::id id) {
    this->mutex.lock();

    int out = std::get<1>((this->thread_info)[id]);

    this->mutex.unlock();
    return out;
}

ThreadPool::function_type ThreadPool::compute_function(std::thread::id id) {
    this->mutex.lock();

    ThreadPool::function_type out = std::get<2>((this->thread_info)[id]);

    this->mutex.unlock();
    return out;
}

std::weak_ptr<Job> &ThreadPool::thread_job(std::thread::id id) {
    this->mutex.lock();

    std::weak_ptr<Job> &out = std::get<3>((this->thread_info)[id]);

    this->mutex.unlock();
    return out;
}

int ThreadPool::index(const std::thread &thread) {
    this->mutex.lock();

    int out = std::get<0>((this->thread_info)[thread.get_id()]);

    this->mutex.unlock();
    return out;
}

int ThreadPool::compute_threads(const std::thread &thread) {
    this->mutex.lock();

    int out = std::get<1>((this->thread_info)[thread.get_id()]);

    this->mutex.unlock();
    return out;
}

ThreadPool::function_type ThreadPool::compute_function(const std::thread &thread) {
    this->mutex.lock();

    ThreadPool::function_type out = std::get<2>((this->thread_info)[thread.get_id()]);

    this->mutex.unlock();
    return out;
}

std::weak_ptr<Job> &ThreadPool::thread_job(const std::thread &thread) {
    this->mutex.lock();

    std::weak_ptr<Job> &out = std::get<3>((this->thread_info)[thread.get_id()]);

    this->mutex.unlock();
    return out;
}

void ThreadPool::thread_finished(std::thread::id id) {
    this->mutex.lock();

    (this->thread_info)[id] = std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{0, 0, nullptr, nullptr};

    ssize_t i = 0;
    while (this->running.size() > 0 && i < this->running.size()) {
        debug("1: Running size: %ld\n", this->running.size());
        assert(this->avail.size() + this->running.size() == this->threads.size());
        if (this->running.at(i).lock()->get_id() == id) {
            for (ssize_t j = 0; j < this->threads.size(); j++) {
                if (this->threads[j]->get_id() == id) {
                    size_t before = this->avail.size();
                    this->avail.push_back(this->threads[j]);
                    assert(before + 1 == this->avail.size());
                    break;
                }
            }
            //            this->avail.push_back(std::move((this->running)[i]));
            size_t before = this->running.size();
            this->running.erase(this->running.begin() + i);
            if (before - 1 != this->running.size()) {
                debug("2: Running size: %ld\n", this->running.size());
            }
            assert(before - 1 == this->running.size());
            assert(this->avail.size() + this->running.size() == this->threads.size());
            break;
        }
        i++;
    }

    assert(this->threads.size() == this->max_threads);
#ifndef NDEBUG
    if (this->avail.size() + this->running.size() != this->threads.size()) {
        debug("3: Size of avail: %ld\n", this->avail.size());
        debug("3: Address of avail: %p\n", &(this->avail));
        debug("3: Size of running: %ld\n", this->running.size());
        debug("3: Address of running: %p\n", &(this->running));
    }
#endif
    assert(this->avail.size() + this->running.size() == this->threads.size());
    this->mutex.unlock();
}

void ThreadPool::thread_finished(const std::thread &thread) {
    thread_finished(thread.get_id());
}

bool ThreadPool::stop_condition() {
    return this->stop;
}

int ThreadPool::has_running() {
    return this->running.size();
}

void ThreadPool::wait_on_running() {
    while (this->has_running()) {
        std::this_thread::yield();
    }
}
