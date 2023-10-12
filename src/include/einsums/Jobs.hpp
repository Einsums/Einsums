/**
 * @file Jobs.hpp
 *
 * Job queues and other things.
 */

#pragma once

#include "einsums/TensorAlgebra.hpp"
#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

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
    timeout() noexcept : std::exception() {}

    timeout(const timeout &other) noexcept : std::exception(other) {}

    ~timeout() = default;

    const char *what() noexcept { return timeout::message; }
};

template <typename T>
class Resource;

/**
 * @class ReadLock
 *
 * A read-only lock on a resource. Allows for multiple read access.
 */
template <typename T>
class ReadLock {
  protected:
    // ID for comparing different states of the same item.
    unsigned long id;

    // Pointer to data.
    Resource<T> *data;

  public:
    /**
     * Constructor.
     */
    ReadLock(unsigned long id, Resource<T> *data) : id(id) { this->data = data; }

    /**
     * Delete the copy and move constructors.
     */
    ReadLock(const ReadLock<T> &)  = delete;
    ReadLock(const ReadLock<T> &&) = delete;

    /**
     * Destructor.
     */
    virtual ~ReadLock() {
        this->data = nullptr; // Data is owned by someone else.
    }

    /**
     * Check if the lock has been resolved and can be used.
     */
    virtual bool ready(void) { return this->data->is_readable(*this); }

    /**
     * Get the data that this lock protects. Only gets the data when it is ready.
     *
     * @param timeout How long to wait. Throws an exception on time-out.
     * @return A reference to the data protected by this lock.
     */
    const std::shared_ptr<T> get(std::chrono::duration<size_t> timeout) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        while (!this->ready()) {
            std::this_thread::yield();

            std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
            if (curr - start >= timeout) {
                throw timeout;
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
    const std::shared_ptr<T> get(void) {
        while (!this->ready()) {
            std::this_thread::yield();
        }

        std::atomic_thread_fence(std::memory_order_acquire);
        return this->data->get_data();
    }

    /**
     * Wait until the data is ready.
     *
     * @param timeout How long to wait. Throws an exception on time-out.
     */
    void wait(std::chrono::duration<size_t> timeout) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        while (!this->ready()) {
            std::this_thread::yield();

            std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
            if (curr - start >= timeout) {
                throw timeout;
            }
        }
    }

    /**
     * Wait until the data is ready.
     */
    void wait(void) {
        while (!this->ready()) {
            std::this_thread::yield();
        }
    }

    /**
     * Get a pointer to the resource that produced this lock.
     */
    Resource<T> *get_resource(void) { return this->data; }

    /**
     * Release this lock.
     */
    bool release(void) {
        std::atomic_thread_fence(std::memory_order_release);
        return this->data->release(*this);
    }

    /**
     * Get the data contained in the resource.
     */
    explicit operator const T &(void) {
        std::atomic_thread_fence(std::memory_order_acquire);
        return *(this->get());
    }

    /**
     * Compare two locks to see if they are the same.
     */
    bool operator==(const ReadLock<T> &other) const { return this->id == other.id && this->data == other.data; }

    /**
     * Whether a lock is exclusive or not.
     */
    virtual bool is_exclusive() const { return false; }
};

/**
 * @class WriteLock<T>
 *
 * Represents an exclusive read-write lock.
 */
template <typename T>
class WriteLock : public ReadLock<T> {
  public:
    /**
     * Constructor.
     */
    WriteLock(unsigned long id, Resource<T> *data) : ReadLock<T>(id, data) {}

    WriteLock(const WriteLock<T> &)  = delete;
    WriteLock(const WriteLock<T> &&) = delete;

    virtual ~WriteLock() = default;

    /**
     * Get the data that this lock protects. Only gets the data when it is ready.
     *
     * @param timeout How long to wait. Throws an exception on time-out.
     * @return A reference to the data protected by this lock.
     */
    std::shared_ptr<T> get(std::chrono::duration<size_t> timeout) {
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        while (!this->ready()) {
            std::this_thread::yield();

            std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
            if (curr - start >= timeout) {
                throw timeout;
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
    std::shared_ptr<T> get(void) {
        while (!this->ready()) {
            std::this_thread::yield();
        }

        std::atomic_thread_fence(std::memory_order_acquire);
        return this->data->get_data();
    }

    explicit operator T &(void) { return *(this->get()); }

    bool is_exclusive() const override { return true; }

    bool ready(void) override { return this->data->is_writable(*this); }
};

/**
 * @class Resource<T>
 *
 * Represents a resource with locks.
 */
template <typename T>
class Resource {
  private:
    std::vector<std::vector<std::shared_ptr<ReadLock<T>>> *> locks;

    unsigned long id;

    std::shared_ptr<T> data;

    std::atomic_bool is_locked;

  public:
    /**
     * Constructor.
     */
    Resource(T *data) : locks{}, id(0), data(std::shared_ptr<T>(data)), is_locked(false) {}

    /**
     * Don't allow copy or move.
     */
    Resource(const Resource<T> &)  = delete;
    Resource(const Resource<T> &&) = delete;

    /**
     * Destructor.
     */
    virtual ~Resource() {
        for (auto state : this->locks) {
            state->clear();
        }
        this->locks.clear();
    }

    std::shared_ptr<T> get_data() { return std::shared_ptr<T>(this->data); }

    /**
     * Obtain a shared lock.
     *
     * @return A pointer to the shared lock.
     */
    std::shared_ptr<ReadLock<T>> lock_shared() {
        // wait to be allowed to edit the resource.
        while (this->is_locked) {
            std::this_thread::yield();
        }

        this->is_locked = true;

        // Make sure there is somewhere to put the locks.
        if (this->locks.size() == 0) {
            this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>>());
        } else if (this->locks.back()->size() == 1 && this->locks.back()->at(0)->is_exclusive()) {
            this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>>());
        }

        // Make the lock.
        std::shared_ptr<ReadLock<T>> out = std::make_shared<ReadLock<T>>(this->id, this);

        // Increment the serial tracker.
        this->id++;

        // Add the lock.
        this->locks.back()->push_back(out);

        // Release the resource.
        this->is_locked = false;
        return out;
    }

    /**
     * Obtain an exclusive lock.
     */
    std::shared_ptr<WriteLock<T>> lock() {
        while (this->is_locked.load()) {
            std::this_thread::yield();
        }
        this->is_locked = true;
        this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>>());

        std::shared_ptr<WriteLock<T>> out = std::make_shared<WriteLock<T>>(this->id, this);
        this->id++;

        this->locks.back()->push_back(out);
        this->is_locked = false;
        return out;
    }

    /**
     * Release a lock.
     *
     * @param The lock to release.
     * @return true if a lock was released, false if no lock was found.
     */
    bool release(const ReadLock<T> &lock) {
        bool ret = false;
        while (this->is_locked.load()) {
            std::this_thread::yield();
        }
        this->is_locked = true;
        for (auto state : this->locks) {
            for (auto curr_lock = state->begin(); curr_lock != state->end(); curr_lock++) {
                if (**curr_lock == lock) {
                    state->erase(curr_lock);
                    curr_lock--; // Need to put back to the right place.
                    ret = true;
                }
            }
        }

        // Remove empty states.
        for (auto state = this->locks.begin(); state != this->locks.end(); state++) {
            if ((*state)->empty()) {
                delete *state;
                this->locks.erase(state);
                state--;
            }
        }

        // Release the lock on the lock.
        this->is_locked = false;

        return ret;
    }

    /**
     * Tests whether this resource is completely unlocked.
     *
     * @return True if no locks are held, false if some locks are held.
     */
    bool is_open(void) {
        while (this->is_locked.load()) {
            std::this_thread::yield();
        }

        this->is_locked = true;

        if (this->locks.empty()) {
            this->is_locked = false;
            return true;
        }
        if (this->locks.size() == 1 && this->locks[0].empty()) {
            this->is_locked = false;
            return true;
        }
        this->is_locked = false;
        return false;
    }

    /**
     * Tests whether a promised lock to this resource is held by the given lock.
     *
     * @param lock The lock to test.
     * @return True if a lock is held. False if a lock is not held.
     */
    bool is_promised(const ReadLock<T> &lock) {
        while (this->is_locked.load()) {
            std::this_thread::yield();
        }

        this->is_locked = true;

        for (auto state : this->locks) {
            for (auto curr_lock : *state) {
                if (curr_lock == lock) {
                    this->is_locked = false;
                    return true;
                }
            }
        }

        this->is_locked = false;
        return false;
    }

    /**
     * Test if the current lock allows reading.
     *
     * @param lock The lock to test.
     * @return True if a lock is held and currently allows reading. False if the lock does not allow reading, or is not held.
     */
    bool is_readable(const ReadLock<T> &lock) {
        while (this->is_locked.load()) {
            std::this_thread::yield();
        }

        this->is_locked = true;

        if (this->locks.size() == 0) {
            this->is_locked = false;
            return false;
        }

        for (auto curr_lock : *(this->locks[0])) {
            if (*curr_lock == lock) {
                this->is_locked = false;
                return true;
            }
        }

        this->is_locked = false;
        return false;
    }

    /**
     * Check whether a given lock is available for writing.
     *
     * @param lock The lock to test.
     * @return True if the given lock currently allows writing. False if the current lock is not available for writing.
     */
    bool is_writable(const ReadLock<T> &lock) {

        if (!lock.is_exclusive()) {
            return false; // The lock is not a writable lock. It will never be a writable lock.
        }

        while (this->is_locked.load()) {
            std::this_thread::yield();
        }

        this->is_locked = true;

        if (this->locks.size() == 0) {
            this->is_locked = false;
            return false; // No locks given.
        }

        if (this->locks[0]->size() != 1) {
            this->is_locked = false;
            return false; // The state is a read-only state.
        }

        if (*(this->locks[0]->at(0)) == lock) {
            this->is_locked = false;
            return true; // This lock has sole ownership.
        }

        this->is_locked = false;
        return false;
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

    std::vector<std::pair<std::shared_ptr<Job>, std::vector<std::shared_ptr<std::thread>>>> running;

    /// Whether the job manager is running or not.
    std::atomic_bool is_running;

    /// Whether the manager is locked for modification.
    std::atomic_bool is_locked;

    std::thread *thread;

    JobManager() : jobs{}, running{}, is_locked(false), is_running(false), thread(nullptr) { std::atexit(JobManager::cleanup); }

    // No copy or move.
    JobManager(const JobManager &)  = delete;
    JobManager(const JobManager &&) = delete;

    ~JobManager();

    EINSUMS_EXPORT static void cleanup();

    friend class std::multiset<std::shared_ptr<Job>>;
    friend class std::thread;

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
    EINSUMS_EXPORT void queue_job(std::shared_ptr<Job> job);

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
     */
    EINSUMS_EXPORT bool isrunning();
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
    std::atomic_bool is_locked, stop;

    int max_threads;

    std::vector<std::shared_ptr<std::thread>> avail, running;

    std::map<std::thread::id, std::tuple<int, int, function_type, std::shared_ptr<Job>>> thread_info;

    /// Lock and unlock the pool for editing.
    void lock();

    void unlock();

    EINSUMS_EXPORT ThreadPool(int threads);

    ThreadPool(const ThreadPool &)  = delete;
    ThreadPool(const ThreadPool &&) = delete;

    EINSUMS_EXPORT ~ThreadPool();

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
     * Return the singleton instance.
     */
    EINSUMS_EXPORT static ThreadPool &get_singleton();

    /**
     * Request a set number of resources.
     *
     * @param count The number of resource (cores, stream processors, etc.) to request.
     * @return Whether the resources have been allocated.
     */
    EINSUMS_EXPORT std::vector<std::shared_ptr<std::thread>> request(unsigned int count, function_type func, std::shared_ptr<Job> job);

    /**
     * Request up to a set number of resources.
     *
     * @param count The maximum number of resources to request.
     * @return The number of resources that have been requested.
     */
    EINSUMS_EXPORT std::vector<std::shared_ptr<std::thread>> request_upto(unsigned int count, function_type func, std::shared_ptr<Job> job);

    /**
     * Release a number of compute resources.
     *
     * @param count The number of resources to release.
     */
    EINSUMS_EXPORT void release(std::vector<std::shared_ptr<std::thread>> &threads);

    /**
     * Get the index of a thread in a compute kernel.
     *
     * @param id A thread id.
     * @return The index of the thread in the calculation.
     */
    EINSUMS_EXPORT int index(std::thread::id id);

    /**
     * Get the kernel size for a computation.
     *
     * @param id A thread id.
     * @return The number of threads in a computation.
     */
    EINSUMS_EXPORT int compute_threads(std::thread::id id);

    /**
     * Get the function that a thread should run.
     *
     * @param id A thread id.
     * @return The function a thread should run.
     */
    EINSUMS_EXPORT function_type compute_function(std::thread::id id);

    /**
     * Get the job associated with a thread.
     *
     * @param id A thread id.
     * @return A pointer to the job.
     */
    EINSUMS_EXPORT std::shared_ptr<Job> thread_job(std::thread::id id);

    /**
     * Signal that a thread has finished.
     *
     * @param id A thread id.
     */
    EINSUMS_EXPORT void thread_finished(std::thread::id id);

    /**
     * Returns true when the threads are supposed to stop.
     */
    EINSUMS_EXPORT bool stop_condition();
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
    std::shared_ptr<ReadLock<AType>>  _A;
    std::shared_ptr<ReadLock<BType>>  _B;
    std::shared_ptr<WriteLock<CType>> _C;

    const CDataType  _C_prefactor;
    const ABDataType _AB_prefactor;

    const CIndices &_Cs;
    const AIndices &_As;
    const BIndices &_Bs;

    bool _running, _done;

  public:
    /**
     * Constructor.
     */
    EinsumJob(CDataType C_prefactor, const CIndices &Cs, std::shared_ptr<WriteLock<CType>> C, const ABDataType AB_prefactor,
              const AIndices &As, std::shared_ptr<ReadLock<AType>> A, const BIndices &Bs, std::shared_ptr<ReadLock<BType>> B)
        : Job(), _C_prefactor(C_prefactor), _AB_prefactor(AB_prefactor), _Cs(Cs), _As(As), _Bs(Bs), _running(false), _done(false) {
        _A = A;
        _B = B;
        _C = C;
    }

    virtual ~EinsumJob() = default;

    /*
     * Overrides for the base class.
     */

    /**
     * The function to run when the job is called.
     */
    virtual void run(void) override {
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
    }

    /**
     * Whether the job is currently able to run.
     */
    virtual bool is_runnable() const override { return _A->ready() && _B->ready() && _C->ready() && !_running && !_done; }

    /**
     * Whether a job is running.
     */
    virtual bool is_running() const override { return this->_running; }

    /**
     * Whether the job is finished.
     */
    virtual bool is_finished() const override { return this->_done; }
};

/**
 * Creates an einsum job and adds it to the job queue.
 *
 * @tparam OnlyUseGenericAlgorithm This is passed to the einsum function being run. Defaults to false in the wrappers.
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
    -> std::shared_ptr<
        EinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>> {
    // Start of function
    using outtype =
        EinsumJob<AType, ABDataType, BType, CType, CDataType, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>;

    std::shared_ptr<WriteLock<CType>> C_lock = C->lock();
    std::shared_ptr<ReadLock<AType>>  A_lock = A->lock_shared();
    std::shared_ptr<ReadLock<BType>>  B_lock = B->lock_shared();
    std::shared_ptr<outtype>          out    = std::make_shared<outtype>(C_prefactor, Cs, C_lock, AB_prefactor, As, A_lock, Bs, B_lock);

    // Queue the job.
    JobManager::get_singleton().queue_job(out);

    return out;
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
                  BType, CType, typename CType::datatype, std::tuple<CIndices...>, std::tuple<AIndices...>, std::tuple<BIndices...>>> {
    return einsum((typename CType::datatype)0, C_indices, C,
                  (std::conditional_t<sizeof(typename AType::datatype) < sizeof(typename BType::datatype), typename BType::datatype,
                                      typename AType::datatype>)1,
                  A_indices, A, B_indices, B);
}

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)
