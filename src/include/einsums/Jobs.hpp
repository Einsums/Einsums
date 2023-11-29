/**
 * @file Jobs.hpp
 *
 * Job queues and resource management.
 */

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

#include "einsums/TensorAlgebra.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

/**
 * @struct timeout
 *
 * Exception thrown when a waiting thread times out.
 */
struct timeout : public std::exception {
  private:
    static constexpr char message[] = "Timeout";

  public:
    /// Constructor.
    timeout() noexcept : std::exception() {}

    /// Copy constructor.
    timeout(const timeout &other) noexcept = default;

    /// Destructor.
    ~timeout() = default;

    /// Returns the message.
    const char *what() noexcept { return timeout::message; }
};

// Declare ahead of time.
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
    ReadPromise(unsigned long id, Resource<T> *data) : id(id) { this->data = data; }

    /**
     * Delete the copy and move constructors.
     */
    ReadPromise(const ReadPromise<T> &)  = delete;
    ReadPromise(const ReadPromise<T> &&) = delete;

    /**
     * Destructor.
     */
    virtual ~ReadPromise() {
        this->release(); // Tell the resource to release this lock.
        this->data = nullptr; // Data is owned by someone else, so ignore it.
    }

    /**
     * Check if the lock has been resolved and can be used.
     */
    virtual bool ready() { return this->data->is_readable(*this); }

    /**
     * Get the data that this promise wraps. Only gets the data when it is ready. This version has a timeout.
     *
     * @param time_out How long to wait. Throws an exception on timeout.
     * @return A reference to the data protected by this lock.
     * @throw An einsums::jobs::timeout exception on timeout.
     */
    template <typename Inttype, typename Ratio>
    const std::shared_ptr<T> get(std::chrono::duration<Inttype, Ratio> time_out) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        while (!this->ready()) {
            std::this_thread::yield();

            std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
            if (curr - start >= time_out) {
                throw timeout();
            }
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        return this->data->get_data();
    }

    /**
     * Get the data that this lock protects. Only gets the data when it is ready. Does not have a timeout.
     *
     * @return A reference to the data protected by this lock.
     */
    const std::shared_ptr<T> get() {
        while (!this->ready()) {
            std::this_thread::yield();
        }

        std::atomic_thread_fence(std::memory_order_acquire);
        return this->data->get_data();
    }

    /**
     * Wait until the data is ready. Has a timeout.
     *
     * @param time_out How long to wait. Throws an exception on time-out.
     * @throw An einsums::jobs::timeout exception on timeout.
     */
    template <typename Inttype, typename Ratio>
    void wait(std::chrono::duration<Inttype, Ratio> time_out) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        while (!this->ready()) {
            std::this_thread::yield();

            std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
            if (curr - start >= time_out) {
                throw jobs::timeout();
            }
        }
    }

    /**
     * Wait until the data is ready. Blocks without a timeout.
     */
    void wait() {
        while (!this->ready()) {
            std::this_thread::yield();
        }
    }

    /**
     * Get a pointer to the resource that produced this lock.
     *
     * @return A pointer to a resource.
     */
    Resource<T> *get_resource() { return this->data; }

    /**
     * Release this lock. It will never be able to be resolved after this.
     *
     * @return True if the promise is successfully released, false if it was not. For instance, if it was already removed.
     */
    bool release() {
        std::atomic_thread_fence(std::memory_order_release);
        return this->data->release(*this);
    }

    /**
     * Get the data contained in the resource. This is a wrapper around einsums::jobs::Resource::get.
     *
     * @return A const reference to the data held by the promise.
     */
    explicit operator const T &() {
        std::atomic_thread_fence(std::memory_order_acquire);
        return *(this->get());
    }

    /**
     * Compare two locks to see if they are the same.
     *
     * @return True if the two promises are the same.
     */
    bool operator==(const ReadPromise<T> &other) const { return this->id == other.id && this->data == other.data; }

    /**
     * Whether a lock is exclusive or not.
     *
     * @return False for an einsums::jobs::ReadPromise.
     */
    [[nodiscard]] virtual bool is_exclusive() const { return false; }
};

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
    WritePromise(unsigned long id, Resource<T> *data) : ReadPromise<T>(id, data) {}

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
    std::shared_ptr<T> get(std::chrono::duration<Inttype, Ratio> time_out) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        while (!this->ready()) {
            std::this_thread::yield();

            std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
            if (curr - start >= time_out) {
                throw timeout();
            }
        }

        std::atomic_thread_fence(std::memory_order_acquire);

        return this->data->get_data();
    }

    /**
     * Get the data that this lock protects. Only gets the data when it is ready.
     *
     * @return A reference to the data protected by this lock.
     */
    std::shared_ptr<T> get() {
        while (!this->ready()) {
            std::this_thread::yield();
        }

        std::atomic_thread_fence(std::memory_order_acquire);
        return this->data->get_data();
    }

    /**
     * A non-const cast. This is a wrapper around einsums::jobs::WritePromise::get.
     *
     * @return A reference to the data protected by this lock.
     */
    explicit operator T &() { return *(this->get()); }

    /**
     * Tells whether the promise is exclusive.
     *
     * @return True for an einsums::jobs::WritePromise.
     */
    [[nodiscard]] bool is_exclusive() const override { return true; }

    /**
     * Tells whether the promise is ready for access.
     *
     * @return True if the promise has a lock. False if it does not.
     */
    bool ready() override { return this->data->is_writable(*this); }
};

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
    std::vector<std::vector<std::shared_ptr<ReadPromise<T>>> *> locks;

    /**
     * Internal counter. It ensures that each promies produced will have a unique identifier.
     */
    unsigned long id;

    /// A pointer to the data managed by the resource.
    std::shared_ptr<T> data;

  protected:
    /// A mutex to help eliminate data races.
    std::mutex mutex;

  public:
    /**
     * Make a new resource constructed with the given arguments. The arguments passed will be passed along to the constructor for the
     * managed data type.
     */
    template <typename... Args>
    Resource(Args &&...args) : locks{}, id(0), mutex() {
        this->data = std::make_shared<T>(args...);
    }

    /**
     * Don't allow copy or move.
     */
    Resource(const Resource<T> &)  = delete;
    Resource(const Resource<T> &&) = delete;

    /**
     * Destructor.
     */
    virtual ~Resource() {
        mutex.lock();
        for (auto state : this->locks) {
            state->clear();
        }
        this->locks.clear();
        this->data.reset();
    }

    /**
     * Return a shared pointer to the data.
     *
     * @return A shared pointer that has shared ownership on the data managed by this resource.
     */
    std::shared_ptr<T> get_data() { return std::shared_ptr<T>(this->data); }

    /**
     * Obtain a read promise.
     *
     * @return A shared pointer to the read promise.
     */
    std::shared_ptr<ReadPromise<T>> read_promise() {
        // wait to be allowed to edit the resource.
        this->mutex.lock();

        // Make sure there is somewhere to put the locks.
        if (this->locks.size() == 0) {
            this->locks.push_back(new std::vector<std::shared_ptr<ReadPromise<T>>>());
        } else if (this->locks.back()->size() == 1 && this->locks.back()->at(0)->is_exclusive()) {
            this->locks.push_back(new std::vector<std::shared_ptr<ReadPromise<T>>>());
        }

        // Make the lock.
        std::shared_ptr<ReadPromise<T>> out = std::make_shared<ReadPromise<T>>(this->id, this);

        // Increment the serial tracker.
        this->id++;

        // Add the lock.
        this->locks.back()->push_back(out);

        // Release the resource.
        this->mutex.unlock();
        return out;
    }

    /**
     * Obtain an exclusive write promise.
     *
     * @return a shared pointer to the write promise.
     */
    std::shared_ptr<WritePromise<T>> write_promise() {
        this->mutex.lock();

        this->locks.push_back(new std::vector<std::shared_ptr<ReadPromise<T>>>());

        std::shared_ptr<WritePromise<T>> out = std::make_shared<WritePromise<T>>(this->id, this);
        this->id++;

        this->locks.back()->push_back(out);

        this->mutex.unlock();
        return out;
    }

    /**
     * Release a promise.
     *
     * @param The promise to release.
     * @return true if a lock was released, false if no lock was found.
     */
    bool release(const ReadPromise<T> &lock) {
        bool ret = false;
        this->mutex.lock();

        for (auto state : this->locks) {
            size_t size = state->size();
            for (size_t i = 0; i < size; i++) {
                if (*(state->at(i)) == lock) {
                    state->erase(std::next(state->begin(), i));
                    size = state->size();
                    i--;
                    ret = true;
                }
            }
        }

        // Remove empty states.
        size_t size = this->locks.size();
        for (size_t i = 0; i < size; i++) {
            if (this->locks[i]->empty()) {
                this->locks.erase(std::next(this->locks.begin(), i));
                size = this->locks.size();
                i--;
            }
        }

        // Release the lock on the lock.
        this->mutex.unlock();

        return ret;
    }

    /**
     * Tests whether this resource is completely unlocked.
     *
     * @return True if no locks are held, false if some locks are held.
     */
    bool is_open() {
        this->mutex.lock();

        if (this->locks.empty()) {
            this->mutex.unlock();
            return true;
        }
        if (this->locks.size() == 1 && this->locks[0].empty()) {
            this->mutex.unlock();
            return true;
        }
        this->mutex.unlock();
        return false;
    }

    /**
     * Tests whether a promised lock to this resource is held by the given lock. Does not tell whether the promise has ownership and can read or modify.
     * Only tells whether the promise will be honored at some point.
     *
     * @param lock The lock to test.
     * @return True if a lock is held. False if a lock is not held.
     */
    bool is_promised(const ReadPromise<T> &lock) {
        this->mutex.lock();

        for (auto state : this->locks) {
            for (auto curr_lock : *state) {
                if (curr_lock == lock) {
                    this->mutex.unlock();
                    return true;
                }
            }
        }

        this->mutex.unlock();
        return false;
    }

    /**
     * Test if the given promise allows reading.
     *
     * @param promise The promise to test.
     * @return True if a lock is held and currently allows reading. False if the promise does not allow reading, or is not held.
     */
    bool is_readable(const ReadPromise<T> &promise) {
        this->mutex.lock();

        if (this->locks.size() == 0) {
            this->mutex.unlock();
            return false;
        }

        for (auto &curr_lock : *(this->locks[0])) {
            if (*curr_lock == promise) {
                this->mutex.unlock();
                return true;
            }
        }

        this->mutex.unlock();
        return false;
    }

    /**
     * Check whether a given promise is available for writing.
     *
     * @param promise The promise to test.
     * @return True if the given lock currently allows writing. False if the current promise is not available for writing.
     */
    bool is_writable(const ReadPromise<T> &promise) {

        if (!promise.is_exclusive()) {
            return false; // The lock is not a writable lock. It will never be a writable lock.
        }

        this->mutex.lock();

        if (this->locks.size() == 0) {
            this->mutex.unlock();
            return false; // No locks given.
        }

        if (this->locks[0]->size() != 1) {
            this->mutex.unlock();
            return false; // The state is a read-only state.
        }

        if (*(this->locks[0]->at(0)) == promise) {
            this->mutex.unlock();
            return true; // This lock has sole ownership.
        }

        this->mutex.unlock();
        return false;
    }

    /**
     * Clear the memory. Useful if the memory is owned by another scope.
     */
    void clear() {
        this->mutex.lock();
        this->data.reset();

        this->mutex.unlock();
    }
};

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
    virtual void run(void) = 0;

    /**
     * Whether the job is currently able to run.
     */
    virtual bool is_runnable() const = 0;

    /**
     * Whether a job is running.
     */
    virtual bool is_running() const = 0;

    /**
     * Whether the job is finished.
     */
    virtual bool is_finished() const = 0;

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

/**
 * @class JobManager
 *
 * Manages jobs.
 */
class JobManager final {
  private:
    /// Lists of running and waiting jobs.
    std::vector<std::shared_ptr<Job>> jobs;

    std::vector<std::pair<std::shared_ptr<Job>, std::vector<std::weak_ptr<std::thread>>>> running;

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
     * Queue a job.
     *
     * @param The job to queue.
     */
    EINSUMS_EXPORT void queue_job(const std::shared_ptr<Job> &job);

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
    EINSUMS_EXPORT void clear();

    /**
     * Destroy the job queue. This deletes the singleton instance and stops the manager.
     */
    EINSUMS_EXPORT static void destroy();
};

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
    using function_type = void (*)(std::shared_ptr<Job> &&);

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
    std::mutex       mutex;

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
     * Information to pass to each of the threads. There are special values for stopping a thread and indicating that the thread should be idle.
     */
    std::map<std::thread::id, std::tuple<int, int, function_type, std::shared_ptr<Job>>> thread_info;

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
     * @return A vector containing pointers to the threads requested.
     */
    EINSUMS_EXPORT std::vector<std::weak_ptr<std::thread>> &request(unsigned int count, function_type func, std::shared_ptr<Job> &job);

    /**
     * Request up to a set number of resources. May give fewer than the requested.
     *
     * @param count The maximum number of resources to request.
     * @return A vector containing pointers to threads that are being used.
     */
    EINSUMS_EXPORT std::vector<std::weak_ptr<std::thread>> &request_upto(unsigned int count, function_type func, std::shared_ptr<Job> &job);

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
    EINSUMS_EXPORT std::shared_ptr<Job> thread_job(std::thread::id id);

    /**
     * Get the job associated with a thread.
     *
     * @param thread The thread to check.
     * @return A pointer to the job.
     */
    EINSUMS_EXPORT std::shared_ptr<Job> thread_job(const std::thread &thread);

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
};

/**
 * @struct EinsumJob
 *
 * Holds information for running einsum as a job.
 */
template <typename AType, typename ABDataType, typename BType, typename CType, typename CDataType, typename CIndices, typename AIndices,
          typename BIndices>
struct EinsumJob : public Job {
  protected:
    /**
     * @var _A
     * 
     * A read promise to the left tensor in the calculation.
     */
    std::shared_ptr<ReadPromise<AType>>  _A;

    /**
     * @var _B
     *
     * A read promise to the right tensor in the calculation.
     */
    std::shared_ptr<ReadPromise<BType>>  _B;

    /**
     * @var _C
     *
     * A write promise to the result tensor.
     */
    std::shared_ptr<WritePromise<CType>> _C;

    /**
     * @var _C_prefactor
     *
     * The prefactor for the result tensor.
     */
    const CDataType  _C_prefactor;

    /**
     * @var _AB_prefactor
     *
     * The prefactor to multiply the A and B terms by.
     */
    const ABDataType _AB_prefactor;

    /**
     * @var _Cs
     *
     * The indices for the result tensor.
     */
    const CIndices &_Cs;

    /**
     * @var _As
     *
     * The indices for the left tensor.
     */
    const AIndices &_As;

    /**
     * @var _Bs
     *
     * The indices for the right tensor.
     */
    const BIndices &_Bs;

    /**
     * @var _running
     *
     * Whether the job has been picked up by a thread and is working.
     */
    /**
     * @var _done
     *
     * Whether the job is finished or not.
     */
    std::atomic_bool _running, _done;

  public:
    /**
     * Constructor.
     */
    EinsumJob(CDataType C_prefactor, const CIndices &Cs, std::shared_ptr<WritePromise<CType>> C, const ABDataType AB_prefactor,
              const AIndices &As, std::shared_ptr<ReadPromise<AType>> A, const BIndices &Bs, std::shared_ptr<ReadPromise<BType>> B)
        : Job(), _C_prefactor(C_prefactor), _AB_prefactor(AB_prefactor), _Cs(Cs), _As(As), _Bs(Bs), _running(false), _done(false) {
        _A = A;
        _B = B;
        _C = C;
    }

    /**
     * Destructor.
     */
    virtual ~EinsumJob() = default;

    /*
     * Overrides for the base class.
     */

    /**
     * The function to run when the job is called.
     */
    void run() override {
        std::atomic_thread_fence(std::memory_order_acq_rel);
        _running = true;
        auto A   = _A->get();
        auto B   = _B->get();
        auto C   = _C->get();

        einsums::tensor_algebra::einsum(_C_prefactor, _Cs, C.get(), _AB_prefactor, _As, *A, _Bs, *B);

        _C->release();
        _A->release();
        _B->release();
        _running = false;
        _done    = true;
        std::atomic_thread_fence(std::memory_order_acq_rel);
    }

    /**
     * Whether the job is currently able to run.
     */
    [[nodiscard]] bool is_runnable() const override { return _A->ready() && _B->ready() && _C->ready() && !_running && !_done; }

    /**
     * Whether a job is running.
     */
    [[nodiscard]] bool is_running() const override { return this->_running; }

    /**
     * Whether the job is finished.
     */
    [[nodiscard]] bool is_finished() const override { return this->_done; }
};

/**
 * Creates an einsum job and adds it to the job queue.
 *
 * @param C_prefactor The prefactor for the C tensor.
 * @param Cs The C indices.
 * @param C The C tensor.
 * @param AB_prefactor The prefactor for A and B.
 * @param As The A indices.
 * @param A The A tensor.
 * @param Bs The B indices.
 * @param B The B tensor.
 * @return A shared pointer to the job created and queued by this function.
 */
template <typename AType, typename ABDataType, typename BType, typename CType, typename CDataType, typename... CIndices,
          typename... AIndices, typename... BIndices>
auto einsum(CDataType C_prefactor, const std::tuple<CIndices...> &Cs, std::shared_ptr<Resource<CType>> &C, const ABDataType AB_prefactor,
            const std::tuple<AIndices...> &As, std::shared_ptr<Resource<AType>> &A, const std::tuple<BIndices...> &Bs,
            std::shared_ptr<Resource<BType>> &B)
    -> std::shared_ptr<EinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>,
                                 std::tuple<BIndices...>>> & {
    // Start of function
    using outtype =
        EinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>;

    std::shared_ptr<WritePromise<CType>> C_lock = C->write_promise();
    std::shared_ptr<ReadPromise<AType>>  A_lock = A->read_promise();
    std::shared_ptr<ReadPromise<BType>>  B_lock = B->read_promise();
    std::shared_ptr<outtype>         *out =
        new std::shared_ptr<outtype>(std::make_shared<outtype>(C_prefactor, Cs, C_lock, AB_prefactor, As, A_lock, Bs, B_lock));

    // Queue the job.
    JobManager::get_singleton().queue_job(*out);

    return *out;
}

/**
 * Creates an einsum job and adds it to the job queue. Has default prefactors.
 *
 * @param Cs The C indices.
 * @param C The C tensor.
 * @param As The A indices.
 * @param A The A tensor.
 * @param Bs The B indices.
 * @param B The B tensor.
 * @return A shared pointer to the job created and queued by this function.
 */
template <typename AType, typename BType, typename CType, typename... CIndices, typename... AIndices, typename... BIndices>
auto einsum(const std::tuple<CIndices...> &C_indices, std::shared_ptr<Resource<CType>> &C, const std::tuple<AIndices...> &A_indices,
            std::shared_ptr<Resource<AType>> &A, const std::tuple<BIndices...> &B_indices, std::shared_ptr<Resource<BType>> &B)
    -> std::shared_ptr<
        EinsumJob<AType,
                  std::conditional_t<sizeof(typename AType::datatype) < sizeof(typename BType::datatype), typename BType::datatype,
                                     typename AType::datatype>,
                  BType, CType, typename CType::datatype, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>> & {
    return einsum((typename CType::datatype)0, C_indices, C,
                  (std::conditional_t<sizeof(typename AType::datatype) < sizeof(typename BType::datatype), typename BType::datatype,
                                      typename AType::datatype>)1,
                  A_indices, A, B_indices, B);
}

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)
