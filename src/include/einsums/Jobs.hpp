/**
 * @file Jobs.hpp
 *
 * Job queues and other things.
 */

#pragma once

#include <vector>
#include <multiset>
#include <map>
#include <chrono>
#include <thread>
#include <atomic>
#include <stdexcept>

#include "einsums/_Common.hpp"
#include "einsums/_Export.hpp"

EINSUMS_BEGIN_NAMESPACE_HPP(einsums::jobs)

/**
 * @struct timeout
 *
 * Exception thrown when a waiting thread times out.
 */
struct timeout : public std::except {
private:
  static constexpr char message[] = "Timeout";
public:
  timeout() noexcept : std::except() {}

  timeout(const timeout &other) noexcept : std::except(other) {}

  ~timeout() = default;

  const char *what() noexcept {
    return timeout::message;
  }

};

template<typename T> class Resource;

/**
 * @class ReadLock
 *
 * A read-only lock on a resource. Allows for multiple read access.
 */
template<typename T>
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
  ReadLock(unsigned long id, Resource<T> *data) : id(id), data(data) {}

  /**
   * Delete the copy and move constructors.
   */
  ReadLock(const ReadLock<T> &) = delete;
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
  bool ready(void) {
    return this->data->ready(*this);
  }

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @param timeout How long to wait. Throws an exception on time-out.
   * @return A reference to the data protected by this lock.
   */
  const T &get(std::chrono::duration timeout) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while(!this->ready()) {
      std::this_thread::yield();

      std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
      if(curr - start >= timeout) {
	throw(*new timeout());
      }
    }

    return this->data->get_data();
  }

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @return A reference to the data protected by this lock.
   */
  const T &get(void) {
    while(!this->ready()) {
      std::this_thread::yield();
    }
    return this->data->get_data();
  }

  /**
   * Wait until the data is ready.
   *
   * @param timeout How long to wait. Throws an exception on time-out.
   */
  void wait(std::chrono::duration timeout) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while(!this->ready()) {
      std::this_thread::yield();

      std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
      if(curr - start >= timeout) {
	throw(*new timeout());
      }
    }
  }

  /**
   * Wait until the data is ready.
   */
  void wait(void) {
    while(!this->ready()) {
      std::this_thread::yield();
    }
  }

  /**
   * Get a pointer to the resource that produced this lock.
   */
  Resource<T> *get_resource(void) {
    return this->data;
  }

  /**
   * Release this lock.
   */
  bool release(void) {
    return this->data->release(*this);
  }

  /**
   * Get the data contained in the resource.
   */
  explicit operator const T&(void) {
    return this->get();
  }

  /**
   * Compare two locks to see if they are the same.
   */
  bool operator==(ReadLock<T> &other) {
    return this->id == other.id && this->data == other.data;
  }

  /**
   * Whether a lock is exclusive or not.
   */
  virtual bool is_exclusive() {
    return false;
  }
};

/**
 * @class WriteLock<T>
 *
 * Represents an exclusive read-write lock.
 */
template<T>
class WriteLock : public ReadLock<T> {
public:
  /**
   * Constructor.
   */
  WriteLock(unsigned long id, Resource<T> *data) : ReadLock<T>(id, data) {}

  WriteLock(const WriteLock<T> &) = delete;
  WriteLock(const WriteLock<T> &&) = delete;
  
  virtual ~WriteLock() = default;

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @param timeout How long to wait. Throws an exception on time-out.
   * @return A reference to the data protected by this lock.
   */
  T &get(std::chrono::duration timeout) {
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    while(!this->ready()) {
      std::this_thread::yield();

      std::chrono::system_clock::time_point curr = std::chrono::system_clock::now();
      if(curr - start >= timeout) {
	throw(*new timeout());
      }
    }

    return this->data->get_data();
  }

  /**
   * Get the data that this lock protects. Only gets the data when it is ready.
   *
   * @return A reference to the data protected by this lock.
   */
  T &get(void) {
    while(!this->ready()) {
      std::this_thread::yield();
    }
    return this->data->get_data();
  }

  explicit operator T&(void) {
    return this->get();
  }

  bool is_exclusive() override {
    return true;
  }
};


/**
 * @class Resource<T>
 *
 * Represents a resource with locks.
 */
template<typename T>
class Resource {
private:
  std::vector<std::vector<std::shared_ptr<ReadLock<T> >> *> locks;

  unsigned long id;

  T *data;

  std::atomic_bool is_locked;

public:

  /**
   * Constructor.
   */
  Resource(T *data) : locks{}, id(0), data(data), is_locked(false) {}

  /**
   * Don't allow copy or move.
   */
  Resource(const Resource<T> &) = delete;
  Resource(const Resource<T> &&) = delete;

  /**
   * Destructor.
   */
  virtual ~Resource() {
    for(auto state : this->locks) {
      for(auto lock : *state) {
	delete lock;
      }
      state->clear();
    }
    this->locks.clear();
  }

  /**
   * Obtain a shared lock.
   *
   * @return A pointer to the shared lock.
   */
  std::shared_ptr<ReadLock<T>> lock_shared() {
    // wait to be allowed to edit the resource.
    while(this->is_locked.load()) {
      std::this_thread::yield();
    }
    this->is_locked = true;

    // Make sure there is somewhere to put the locks.
    if(this->locks.size() == 0) {
      this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>());
    } else if((this->locks.end())->size() == 1 && (this->locks.end())->at(0)->is_exclusive()) {
      this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>());
    }

    // Make the lock.
    std::shared_ptr<ReadLock<T>> out = std::make_shared<ReadLock<T>>(this->id, this);

    // Increment the serial tracker.
    this->id++;

    // Add the lock.
    (this->locks.end())->push_back(out);
    // Release the resource.
    this->is_locked = false;
    return out;
  }

  /**
   * Obtain an exclusive lock.
   */
  std::shared_ptr<WriteLock<T>> lock() {
    while(this->is_locked.load()) {
      std::this_thread::yield();
    }
    this->is_locked = true;
    this->locks.push_back(new std::vector<std::shared_ptr<ReadLock<T>>());

    std::shared_ptr<WriteLock<T>> out = std::make_shared<WriteLock<T>>(this->id, this);
    this->id++;

    (this->locks.end())->push_back(out);
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
    while(this->is_locked.load()) {
      std::this_thread::yield();
    }
    this->is_locked = true;
    for(auto state : this->locks) {
      for(auto curr_lock = state->begin(); curr_lock != state->end(); curr_lock++) {
	if(*curr_lock == lock) {
	  delete *curr_lock;
	  state->erase(curr_lock);
	  curr_lock--; // Need to put back to the right place.
	  ret = true;
	}
      }
    }

    // Remove empty states.
    for(auto state = this->locks.begin(); state != this->locks.end(); state++) {
      if((*state)->empty()) {
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
    while(this->is_locked.load()) {
      std::this_thread::yield();
    }

    this->is_locked = true;

    if(this->locks.empty()) {
      this->is_locked = false;
      return true;
    }
    if(this->locks.size() == 1 && this->locks[0].empty()) {
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
    while(this->is_locked.load()) {
      std::this_thread::yield();
    }

    this->is_locked = true;

    for(auto state : this->locks) {
      for(auto curr_lock : *state) {
	if(curr_lock == lock) {
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
    while(this->is_locked.load()) {
      std::this_thread::yield();
    }

    this->is_locked = true;

    if(this->locks.size() == 0) {
      this->is_locked = false;
      return false;
    }

    for(auto curr_lock : this->locks[0]) {
      if(*curr_lock == lock) {
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

    if(!lock.is_exclusive()) {
      return false; // The lock is not a writable lock. It will never be a writable lock.
    }
    
    while(this->is_locked.load()) {
      std::this_thread::yield();
    }

    this->is_locked = true;

    if(this->locks.size() == 0) {
      this->is_locked = false;
      return false; // No locks given.
    }

    if(this->locks[0]->size() != 1) {
      this->is_locked = false;
      return false; // The state is a read-only state.
    }

    if(this->locks[0]->at(0) == lock) {
      this->is_locked = false;
      return true; // This lock has sole ownership.
    }

    this->is_locked = false;
    return false;
  }
};
    

/**
 * @enum ComputeResource
 *
 * Flags for the different kinds of compute resources a job can request. Can be or'ed together to request multiple.
 */
enum ComputeResource : public unsigned char {
  SYNCHRONOUS = 0, /// Wait for this job to come up, then run it in place.
  MULTI_THREAD = 0x1, /// Run this job with multiple threads.
  GPU = 0x2, /// Run this job on the GPU.
  MPI = 0x4 /// Run this job through MPI.
};

/**
 * @class Job
 * 
 * Represents an abstract job.
 */
class Job {
protected:
  /// Compute resource flags or'ed together.
  unsigned char compute_res;

  int priority; /// Priority of the job. Will run before other jobs when it can.
  
public:
  Job(unsigned char compute_resources, int priority = 0) : compute_res(compute_resources), priority(priority) {};
  virtual ~Job() = default;

  /**
   * The function to run when the job is called.
   */
  virtual void run(void) = 0;

  /**
   * Whether the job is currently able to run.
   */
  virtual bool is_runnable() = 0;

  /**
   * Whether a job is running.
   */
  virtual bool is_running() = 0;

  /**
   * Whether the job is finished.
   */
  virtual bool is_finished() = 0;

  /**
   * Get the requested compute resources.
   */
  virtual unsigned char compute_resources() {
    return this->compute_res;
  }

  /**
   * Return the priority.
   */
  virtual int get_priority() {
    return this->priority;
  }

  /**
   * Compare priorities.
   */
  virtual auto operator<=>(const Job &other) {
    return this->get_priority() <=> other.get_priority();
  }
};

/**
 * @class JobManager
 *
 * Manages jobs.
 */
final class JobManager {
private:
  
  /// Lists of running and waiting jobs.
  std::multiset<std::shared_ptr<Job>, JobManager::compare> jobs;

  std::vector<std::pair<std::shared_ptr<Job>, std::thread *>> running;

  /// Whether the job manager is running or not.
  std::atomic_bool is_running;

  /// Whether the manager is locked for modification.
  std::atomic_bool is_locked;

  std::thread *thread;
  
  JobManager() : jobs{}, running{}, is_locked(false), is_running(false), thread(nullptr) {
    std::atexit(JobManager::cleanup);
  }

  // No copy or move.
  JobManager(const JobManager &) = delete;
  JobManager(const JobManager &&) = delete;

  ~JobManager();

  static void cleanup();

  static bool compare(std::shared_ptr<Job>, std::shared_ptr<Job>);

  friend class std::multiset<std::shared_ptr<Job>, JobManager::compare>;
  friend class std::thread;

protected: // I know protecting this is useless, but future-proofing never hurt anyone.

  /**
   * Main loop of the manager.
   */
  void manager_loop();

  /**
   * One loop through the manager.
   */
  void manager_event();

public:

  /**
   * Get the one single instance of the job manager.
   *
   * @return A reference to the single job manager.
   */
  static JobManager &get_singleton();

  /**
   * Queue a job.
   *
   * @param The job to queue.
   */
  void queue_job(std::shared_ptr<Job> job);

  /**
   * Start the job manager in a different thread. Raises an exception if it is already running.
   */
  void start_manager();

  /**
   * Stop the manager.
   */
  void stop_manager();

  /**
   * Check whether the manager is running or not.
   */
  bool isrunning();
};

/**
 * @class ComputePool
 * 
 * Represents a pool of computing resources.
 */
class ComputePool {
public:
  /**
   * Create a new compute pool.
   */
  ComputePool() = default;

  // Can't copy or move a pool.
  ComputePool(const ComputePool &) = delete;
  ComputePool(const ComputePool &&) = delete;

  virtual ~ComputePool() = default;

  /**
   * Request a set number of resources.
   *
   * @param count The number of resource (cores, stream processors, etc.) to request.
   * @return Whether the resources have been allocated.
   */
  virtual bool request(int count) = 0;

  /**
   * Request up to a set number of resources.
   * 
   * @param count The maximum number of resources to request.
   * @return The number of resources that have been requested.
   */
  virtual int request_upto(int count) = 0;

  /**
   * Release a number of compute resources.
   *
   * @param count The number of resources to release.
   */
  virtual void release(int count) = 0;
};

/**
 * @class ThreadPool
 *
 * Pools threads together for running computations.
 */
class ThreadPool : public ComputePool {
private:
  
  std::atomic_bool is_locked;

  int max_threads, avail;

  /// Lock and unlock the pool for editing.
  void lock();

  void unlock();

  ThreadPool(int threads) : max_threads(threads), avail(threads), is_locked(false) {}

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool(const ThreadPool &&) = delete;

  ~ThreadPool() = default;

public:
  /**
   * Initialize the thread pool.
   *
   * @param threads The number of threads to hold.
   */
  static void init(int threads);

  /**
   * Return the singleton instance.
   */
  static ThreadPool &get_singleton();

  /**
   * Request a set number of resources.
   *
   * @param count The number of resource (cores, stream processors, etc.) to request.
   * @return Whether the resources have been allocated.
   */
  bool request(unsigned int count) override;

  /**
   * Request up to a set number of resources.
   * 
   * @param count The maximum number of resources to request.
   * @return The number of resources that have been requested.
   */
  int request_upto(unsigned int count) override;

  /**
   * Release a number of compute resources.
   *
   * @param count The number of resources to release.
   */
  void release(unsigned int count) override;
};

/**
 * @struct EinsumJob
 *
 * Holds information for running einsum as a job.
 */
template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
struct EinsumJob : public Job {
protected :

    std::shared_pointer<ReadLock<AType<ADataType, ARank>>> _A;
    std::shared_pointer<ReadLock<BType<BDataType, BRank>>> _B;
    std::shared_pointer<WriteLock<CType<CDataType, CRank>>> _C;

    const CDataType _C_prefactor;
    const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> _AB_prefactor;

    const std::tuple<CIndices...> &_Cs;
    const std::tuple<AIndices...> &_As;
    const std::tuple<BIndices...> &_Bs;

    bool _running, _done;

public :

    /**
     * Constructor.
     */
    EinsumJob(unsigned char compute_resources, CDataType C_prefactor, const std::tuple<CIndices...> & Cs,
            std::shared_ptr<WriteLock<CType<CDataType, CRank>>> &C,
            const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
            const std::tuple<AIndices...> & As, std::shared_ptr<ReadLock<AType<ADataType, ARank>>> &A, const std::tuple<BIndices...> & Bs,
            std::shared_ptr<ReadLock<BType<BDataType, BRank>>> &B) :
	    Job(compute_resources), _A(A), _B(B), _C(C), _C_prefactor(C_prefactor), _AB_prefactor(AB_prefactor),
	    _Cs(Cs), _As(As), _Bs(Bs), _running(false), _done(false) {}

    virtual ~EinsumJob() {
        delete this->_A;
	delete this->_B;
	delete this->_C;
    }

    /*
     * Overrides for the base class.
     */

    /**
     * The function to run when the job is called.
     */
    virtual void run(void) override {
        _running = true;
        einsums::tensor_algebra::einsum<OnlyUseGenericAlgorithm>(_C_prefactor, _Cs, &(_C->get()),
	    _AB_prefactor, _As, _A->get(), _Bs, _B->get());

	_C->release();
	_A->release();
	_B->release();
	_running = false;
	_done = true;
    }

    /**
     * Whether the job is currently able to run.
     */
    virtual bool is_runnable() override {
        return _A->ready() && _B->ready() && _C->ready() && !_running && !_done;
    }

    /**
     * Whether a job is running.
     */
    virtual bool is_running() override {
        return this->_running;
    }

    /**
     * Whether the job is finished.
     */
    virtual bool is_finished() override {
        return this->_done;
    }
};

  
template <bool OnlyUseGenericAlgorithm, template <typename, size_t> typename AType, typename ADataType, size_t ARank,
          template <typename, size_t> typename BType, typename BDataType, size_t BRank, template <typename, size_t> typename CType,
          typename CDataType, size_t CRank, typename... CIndices, typename... AIndices, typename... BIndices>
std::shared_ptr<EinsumJob<OnlyUseGenericAlgorithm, AType<ADataType, ARank>, ADataType, ARank,
                              BType<BDataType, BRank>, BDataType, BRank, CType<CDataType, CRank>, CDataType, CRank,
			      CIndices..., AIndices..., BIndices...>>
einsum(CDataType C_prefactor, const std::tuple<CIndices...> & Cs,
            std::shared_ptr<Resource<CType<CDataType, CRank>>> &C,
            const std::conditional_t<(sizeof(ADataType) > sizeof(BDataType)), ADataType, BDataType> AB_prefactor,
            const std::tuple<AIndices...> & As, std::shared_ptr<Resource<AType<ADataType, ARank>>> &A,
	    const std::tuple<BIndices...> & Bs, std::shared_ptr<Resource<BType<BDataType, BRank>>> &B) {
    using outtype = typename EinsumJob<OnlyUseGenericAlgorithm, AType<ADataType, ARank>, ADataType, ARank,
                              BType<BDataType, BRank>, BDataType, BRank, CType<CDataType, CRank>, CDataType, CRank,
			      CIndices..., AIndices..., BIndices...>;

    std::shared_ptr<WriteLock<CType<CDataType, CRank>>> C_lock = C->lock();
    std::shared_ptr<ReadLock<AType<ADataType, ARank>>> A_lock = A->lock_shared();
    std::shared_ptr<ReadLock<BType<BDataType, BRank>>> B_lock = B->lock_shared();
    std::shared_ptr<outtype> out = std::make_shared<outtype>(C_prefactor, Cs, C_lock, AB_prefactor, As, A_lock, Bs, B_lock);

    // Queue the job.
    JobManager::get_singleton().queue_job(out);

    return out;
}
    

}

EINSUMS_END_NAMESPACE_HPP(einsums::jobs)
