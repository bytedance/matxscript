// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/threadpool/lock_free_thread_pool.h>

namespace matxscript {
namespace runtime {
namespace internal {

void LockFreeThreadPool::ThreadEntry(LockFreeThreadPool* pool, const std::string& name) {
#ifdef __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif
  int64_t sleep_intervals_ns = pool->intervals_ns_;
  if (sleep_intervals_ns <= 0) {
    sleep_intervals_ns = 1;
  }
  for (;;) {
    IRunnablePtr task = nullptr;
    for (;;) {
      if (pool->tasks_.try_dequeue(task)) {
        break;
      }
      if (pool->stop_) {
        break;
      }
      // queue is full, sleep 1us
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
    if (pool->stop_) {
      return;
    } else {
      if (task != nullptr) {
        task->Run();
      }
    }
  }
}

// the constructor just launches some amount of workers
LockFreeThreadPool::LockFreeThreadPool(size_t threads,
                                       const std::string& name,
                                       int64_t intervals_ns)
    : stop_(false), name_(name), tasks_(4096), intervals_ns_(intervals_ns) {
#ifdef _WIN32
  belong_to_pid_ = GetCurrentProcessId();
#else
  belong_to_pid_ = getpid();
#endif
  for (size_t i = 0; i < threads; ++i) {
    char buffer[16] = {0};
    snprintf(buffer, sizeof(buffer), "T%zu.%s", i, name.c_str());
    workers_.emplace_back(LockFreeThreadPool::ThreadEntry, this, std::string(buffer));
  }
}

LockFreeThreadPool::LockFreeThreadPool(size_t threads, const std::string& name)
    : LockFreeThreadPool(threads, name, 1) {
}

size_t LockFreeThreadPool::GetThreadsNum() const {
  return workers_.size();
}

std::vector<std::thread::id> LockFreeThreadPool::GetThreadIds() const {
  std::vector<std::thread::id> ids;
  for (auto& w : workers_) {
    ids.push_back(w.get_id());
  }
  return ids;
}

// add new work item to the pool
void LockFreeThreadPool::EnqueueBulk(std::vector<IRunnablePtr>& runners) {
  for (size_t i = 0; i < runners.size(); ++i) {
    Enqueue(runners[i], i);
  }
}

void LockFreeThreadPool::Enqueue(IRunnablePtr& runner, size_t seq) {
  MXCHECK(runner != nullptr) << "Enqueue arg invalid: runner is null pointer";
  for (;;) {
    if (tasks_.try_enqueue(runner)) {
      return;
    }
    // queue is full, sleep 1us
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
  }
}

// the destructor joins all threads
LockFreeThreadPool::~LockFreeThreadPool() {
#ifdef _WIN32
  auto cur_pid = GetCurrentProcessId();
#else
  auto cur_pid = getpid();
#endif
  stop_ = true;
  if (cur_pid == belong_to_pid_) {
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  } else {
    // After fork, the child process inherits the data-structures of the parent
    // process' thread-pool, but since those threads don't exist, the thread-pool
    // is corrupt. So detach thread here in order to prevent segfaults.
    for (std::thread& worker : workers_) {
      worker.detach();
    }
  }
}

SPSCLockFreeThreadPool::SPSCLockFreeThreadPool(size_t threads,
                                               const std::string& name,
                                               int64_t intervals_ns) {
  for (size_t i = 0; i < threads; ++i) {
    char buffer[16] = {0};
    snprintf(buffer, sizeof(buffer), "T%zu.%s", i, name.c_str());
    workers_.emplace_back(new LockFreeThreadPool(1, std::string(buffer), intervals_ns));
  }
}

SPSCLockFreeThreadPool::~SPSCLockFreeThreadPool() {
  workers_.clear();
}

void SPSCLockFreeThreadPool::Enqueue(IRunnablePtr& runner, size_t seq) {
  if (seq >= workers_.size()) {
    seq = seq % workers_.size();
  }
  workers_[seq]->Enqueue(runner, seq);
}

size_t SPSCLockFreeThreadPool::GetThreadsNum() const {
  return workers_.size();
}

std::vector<std::thread::id> SPSCLockFreeThreadPool::GetThreadIds() const {
  std::vector<std::thread::id> ids;
  for (auto& w : workers_) {
    ids.push_back(w->GetThreadIds()[0]);
  }
  return ids;
}

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript
