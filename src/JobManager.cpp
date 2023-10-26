#include "einsums/Jobs.hpp"

#include <cstdio>
#include <thread>

using namespace einsums::jobs;

template <typename... T>
static void debug(const char *fmt, T... args) {
    std::fprintf(stderr, fmt, args...);
    std::fflush(stderr);
}

static JobManager *instance = nullptr;

static ThreadPool *thread_instance = nullptr;

static bool added_job_exit_handler = false, added_thread_exit_handler = false;

JobManager::JobManager() : jobs{}, running{}, is_locked(false), is_running(false), thread(nullptr) {
  if(!added_job_exit_handler) {
    std::atexit(JobManager::cleanup);
    added_job_exit_handler = true;
  }
}

JobManager::~JobManager() {
  std::fprintf(stderr, "Deleting job manager.\n");
  this->is_running = false;
  if(this->thread != nullptr) {
    this->thread->join();
    delete this->thread;
  }
  this->jobs.clear();
  this->running.clear();
}

void JobManager::cleanup() {
    if (instance != nullptr) {
        delete instance;
    }
    instance = nullptr;
}

void JobManager::manager_loop() {
    // Infinite loop.
    JobManager &inst = JobManager::get_singleton();
    inst.is_running  = true;
    while (inst.is_running) {
        inst.manager_event();      // Process an event.
	std::atomic_thread_fence(std::memory_order_acq_rel);
        std::this_thread::yield(); // Yield for another thread;
    }
    return;
}

static void run_job(std::shared_ptr<Job> &&job) {
    std::atomic_thread_fence(std::memory_order_acquire);
    job->run();

    std::atomic_thread_fence(std::memory_order_release);

    // Job finished.
    ThreadPool::get_singleton().thread_finished(std::this_thread::get_id());
}

#define JOB_THREADS 8

void JobManager::manager_event() {
    // Obtain a lock on the manager.

    while (this->is_locked) {
        std::this_thread::yield();
    }

    this->is_locked = true;

    // Go through each of the running jobs and remove finished jobs.
    for (size_t i = 0; i < this->running.size(); i++) {
        if (std::get<0>(this->running[i])->is_finished()) {
	  ThreadPool::get_singleton().release(std::get<1>(this->running[i]));
	  this->running.erase(std::next(this->running.begin(), i));
            i--;
        }
    }

    // Go through each of the waiting jobs and try to queue them up.
    std::atomic_thread_fence(std::memory_order_acquire);
    for (size_t i = 0; i < this->jobs.size(); i++) {
        if (this->jobs[i]->is_runnable()) {
            auto threads = ThreadPool::get_singleton().request(1, run_job, this->jobs[i]);
            if (threads.size() != 0) {
                this->running.emplace(this->running.cend(), this->jobs[i], threads);
                this->jobs.erase(std::next(this->jobs.begin(), i));
                i--;

                continue;
            }
        }
    }

    this->is_locked = false;
}

JobManager &JobManager::get_singleton() {
    if (instance == nullptr) {
        instance = new JobManager();
    }
    return *instance;
}

void JobManager::queue_job(const std::shared_ptr<Job> &job) {
    this->jobs.insert(this->jobs.cend(), job); // Hint to the end of the list.
}

void JobManager::start_manager() {
    while (this->is_locked) {
        std::this_thread::yield();
    }
    this->is_locked = true;

    if (this->is_running) {
        throw std::runtime_error("Job manager already running!");
    }

    this->is_running = true;

    // Start the thread.
    if(this->thread == nullptr) {
      this->thread = new std::thread(this->manager_loop);
    }

    this->is_locked = false;
}

void JobManager::stop_manager() {
    while (this->is_locked) {
        std::this_thread::yield();
    }

    this->is_locked = true;

    this->is_running = false;
    this->is_locked = false;
    this->thread->join();

    while (this->is_locked) {
        std::this_thread::yield();
    }

    delete this->thread;
    this->thread = nullptr;

    this->is_locked = false;
}

bool JobManager::isrunning() {
    while (this->is_locked) {
        std::this_thread::yield();
    }

    return this->is_running;
}

void JobManager::clear() {
  while(this->is_locked) {
    std::this_thread::yield();
  }

  this->is_locked = true;

  this->jobs.clear();
  this->running.clear();

  this->is_locked = false;
}

void JobManager::destroy() {
  if(instance != nullptr) {
    delete instance;
    instance = nullptr;
  }
}

/**
 * The main loop for the threads.
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
    while (!ThreadPool::get_singleton().stop_condition()) {
        // Check for a job.
        auto func = ThreadPool::get_singleton().compute_function(std::this_thread::get_id());

        if (func == nullptr) {
            // No job. Wait for a bit and check again.
            std::this_thread::yield();
            std::atomic_thread_fence(std::memory_order_acquire);
            continue;
        }

        // Found a job. Run it.
        func(ThreadPool::get_singleton().thread_job(std::this_thread::get_id()));

        // Job is finished. Tell the pool.
        ThreadPool::get_singleton().thread_finished(std::this_thread::get_id());
        std::atomic_thread_fence(std::memory_order_release);
        std::atomic_thread_fence(std::memory_order_acquire);
    }
}

ThreadPool::ThreadPool(int threads) : max_threads(threads), avail{}, running{}, thread_info{} {
  if(!added_thread_exit_handler) {
    std::atexit(ThreadPool::destroy);
    added_thread_exit_handler = true;
  }
  this->is_locked = true;
  this->stop      = false;
  for (int i = 0; i < threads; i++) {
    std::shared_ptr<std::thread> thread = std::make_shared<std::thread>(thread_loop);
    this->avail.push_back(thread);

        this->thread_info[thread->get_id()] = std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{0, 0, nullptr, nullptr};
    }
    this->is_locked = false;
}

ThreadPool::~ThreadPool() {
  std::fprintf(stderr, "Deleting thread pool.\n");

    lock();
    stop = true;

    for (auto kv : thread_info) {
        auto val = std::get<1>(kv);

        std::get<0>(val) = -1;
        std::get<1>(val) = -1;
        std::get<2>(val) = nullptr;
        std::get<3>(val) = nullptr;
    }

    unlock();
    std::atomic_thread_fence(std::memory_order_release);

    for (auto thr : avail) {
        while(!thr->joinable()) {
            std::this_thread::yield();
        }
        thr->join();
    }

    for (auto thr : running) {
        while(!thr->joinable()) {
            std::this_thread::yield();
        }
        thr->join();
    }

    lock();

    thread_info.clear();
    avail.clear();
    running.clear();
    max_threads = 0;

    unlock();
}

void ThreadPool::lock() {
    while (this->is_locked) {
        std::this_thread::yield();
    }
    this->is_locked = true;
}

void ThreadPool::unlock() {
    std::atomic_thread_fence(std::memory_order_release);
    this->is_locked = false;
}

void ThreadPool::init(int threads) {
    if (thread_instance != nullptr) {
        throw std::runtime_error("Double initialization of the thread pool!");
    } else {
        thread_instance = new ThreadPool(threads);
    }
}

void ThreadPool::destroy() {
  if(thread_instance != nullptr) {
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

std::vector<std::shared_ptr<std::thread>> &ThreadPool::request(unsigned int count, ThreadPool::function_type func,
                                                              std::shared_ptr<Job> &job) {
    this->lock();

    // Check that the number of threads can reasonably be allocated.
    if (count > this->max_threads) {
        this->unlock();
        throw std::runtime_error("Could not allocate threads! Requested too many.");
    }

    auto *out = new std::vector<std::shared_ptr<std::thread>>();

    for (int i = 0; i < count; i++) {
        // Check that there are available threads to take.
        if (this->avail.size() == 0) {
            this->unlock();
            return *out;
        }
        // Get the new thread.
        std::shared_ptr<std::thread> thread = this->avail.back();
        this->running.push_back(thread);
        out->push_back(thread); 
        this->avail.pop_back();               

        // Push the thread info.
        this->thread_info[thread->get_id()] = std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{i, count, func, job};
    }

    this->unlock();
    return *out;
}

std::vector<std::shared_ptr<std::thread>> &ThreadPool::request_upto(unsigned int count, ThreadPool::function_type func,
                                                                   std::shared_ptr<Job> &job) {
    this->lock();
    auto *out = new std::vector<std::shared_ptr<std::thread>>();

    int total = 0;

    for (int i = 0; i < count; i++) {
        // Check that there are threads available.
        if (this->avail.size() == 0) {
            break;
        }
        // Get the new thread.
        std::shared_ptr<std::thread> thread = this->avail.back();
        this->running.push_back(thread);
        out->push_back(thread); 
        this->avail.pop_back(); 
        total++;
    }

    // Push the thread info.
    for (int i = 0; i < total; i++) {
        this->thread_info[out->at(i)->get_id()] = std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{i, total, func, job};
    }

    this->unlock();
    return *out;
}

void ThreadPool::release(std::vector<std::shared_ptr<std::thread>> &threads) {
    this->lock();

    // Move running threads back to the waiting pool.
    for(auto thread : threads) {
      for(size_t i = 0; i < this->running.size(); i++) {
	if(this->running[i]->get_id() == thread->get_id()) {
	  this->avail.push_back(this->running[i]);
	  this->thread_info[this->running[i]->get_id()] =
	    std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{0, 0, nullptr, nullptr};
	  this->running.erase(std::next(this->running.begin(), i));
	  i--;
	}
      }
    }

    this->unlock();
}

int ThreadPool::index(std::thread::id id) {
    this->lock();

    int out = std::get<0>(this->thread_info[id]);

    this->unlock();
    return out;
}

int ThreadPool::compute_threads(std::thread::id id) {
    this->lock();

    int out = std::get<1>(this->thread_info[id]);

    this->unlock();
    return out;
}

ThreadPool::function_type ThreadPool::compute_function(std::thread::id id) {
    this->lock();

    ThreadPool::function_type out = std::get<2>(this->thread_info[id]);

    this->unlock();
    return out;
}

std::shared_ptr<Job> ThreadPool::thread_job(std::thread::id id) {
    this->lock();

    std::shared_ptr<Job> out = std::get<3>(this->thread_info[id]);

    this->unlock();
    return out;
}

void ThreadPool::thread_finished(std::thread::id id) {
    this->lock();

    this->thread_info[id] = std::tuple<int, int, ThreadPool::function_type, std::shared_ptr<Job>>{0, 0, nullptr, nullptr};

    for (size_t i = 0; i < this->running.size(); i++) {
        if (this->running[i]->get_id() == id) {
            this->avail.push_back(this->running[i]);
            this->running.erase(std::next(this->running.begin(), i));
            i--;
        }
    }

    this->unlock();
}

bool ThreadPool::stop_condition() {
    return this->stop;
}
