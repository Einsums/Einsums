#include "einsums/Jobs.hpp"



using namespace einsums::jobs;

static JobManager *instance = nullptr;

static ThreadPool *thread_instance = nullptr;


JobManager::~JobManager() {
  this->jobs.clear();
  this->running.clear();
}

void JobManager::cleanup() {
  if(instance != nullptr) {
    delete instance;
  }
  instance = nullptr;
}

void JobManager::manager_loop() {
  // Infinite loop.
  JobManager &inst = JobManager::get_singleton();
  inst.is_running = true;
  while(inst.is_running) {
    inst.manager_event(); // Process an event.
    std::this_thread::yield(); // Yield for another thread;
  }
}

static void run_job(Job *job) {
  job->run();
}

#define JOB_THREADS 8

void JobManager::manager_event() {
  // Obtain a lock on the manager.
  while(this->is_locked) {
    std::this_thread::yield();
  }

  this->is_locked = true;

  // Go through each of the running jobs and remove finished jobs.
  for(auto job = this->running.begin(); job != this->running.end(); job++) {
    if(std::get<0>(*job)->is_finished()) {
      std::get<1>(*job)->join();
      this->running.erase(job);
      job--;
    }
  }

  // Go through each of the waiting jobs and try to queue them up.
  for(auto job = this->jobs.begin(); job != this->jobs.end(); job++) {
    if((*job)->is_runnable()) {
      unsigned char res = (*job)->compute_resources();

      if(res & MULTI_THREAD) {
        if(ThreadPool::get_singleton().request(1)) {
	  this->running.push_back(std::pair<std::shared_ptr<Job>, std::thread *>
				  (std::shared_ptr<Job>(*job), new std::thread(run_job, job->get())));
	  this->jobs.erase(job);
	  job--;
	  continue;
	}
      }
    }
  }

  this->is_locked = false;
}
	
      
JobManager &JobManager::get_singleton() {
  if(instance == nullptr) {
    instance = new JobManager();
  }
  return *instance;
}

void JobManager::queue_job(std::shared_ptr<Job> job) {
  this->jobs.insert(this->jobs.end(), job); // Hint to the end of the list.
}

void JobManager::start_manager() {
  while(this->is_locked) {
    std::this_thread::yield();
  }
  this->is_locked = true;
  
  if(this->isrunning()) {
    throw(*new std::runtime_error("Job manager already running!"));
  }

  this->is_running = true;

  // Start the thread.
  this->thread = new std::thread(this->manager_loop);

  this->is_locked = false;
}

void JobManager::stop_manager() {
  while(this->is_locked) {
    std::this_thread::yield();
  }

  this->is_locked = true;

  this->is_running = false;
  this->thread->join();

  delete this->thread;

  this->is_locked = false;
}

bool JobManager::isrunning() {
  while(this->is_locked) {
    std::this_thread::yield();
  }

  return this->is_running;
}

void ThreadPool::lock() {
  while(this->is_locked) {
    std::this_thread::yield();
  }
  this->is_locked = true;
}

void ThreadPool::unlock() {
  this->is_locked = false;
}

void ThreadPool::init(int threads) {
  if(thread_instance != nullptr) {
    thread_instance->max_threads = threads;
    thread_instance->avail = threads;
  } else {
    thread_instance = new ThreadPool(threads);
  }
}

ThreadPool &ThreadPool::get_singleton() {
  if(thread_instance == nullptr) {
    throw std::runtime_error("Thread pool needs to be initialized!");
  }
  return *thread_instance;
}

bool ThreadPool::request(unsigned int count) {
  this->lock();

  if(count <= this->avail) {
    this->avail -= count;
    this->unlock();
    return true;
  }

  this->unlock();
  return false;
}

int ThreadPool::request_upto(unsigned int count) {
  this->lock();

  if(count <= this->avail) {
    this->avail -= count;
    this->unlock();
    return count;
  } else {
    int ret = this->avail;
    this->avail = 0;
    this->unlock();
    return ret;
  }
}

void ThreadPool::release(unsigned int count) {
  this->lock();

  this->avail += count;
  if(this->avail > this->max_threads) {
    this->avail = this->max_threads;
  } else if(this->avail < 0) {
    this->avail = 0;
  }

  this->unlock();
}
