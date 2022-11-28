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
#include <matxscript/runtime/threadpool/lock_based_thread_pool.h>

#include <unistd.h>

namespace matxscript {
namespace runtime {
namespace internal {

void LockBasedThreadPool::ThreadEntry(LockBasedThreadPool* pool, const std::string& name) {
#ifdef __linux__
  pthread_setname_np(pthread_self(), name.c_str());
#endif
  while (true) {
    IRunnablePtr task = nullptr;
    {
      Lock lock(pool->mutex_);
      while (pool->tasks_.empty() && !pool->stop_) {
        pool->cond_.wait(lock);
      }
      if (!pool->tasks_.empty()) {
        task = pool->tasks_.front();
        pool->tasks_.pop();
      } else if (pool->stop_) {
        return;
      }
    }
    if (task) {
      task->Run();
    }
  }
}

LockBasedThreadPool::LockBasedThreadPool(size_t thread_num, const std::string& name) {
#ifdef _WIN32
  belong_to_pid_ = GetCurrentProcessId();
#else
  belong_to_pid_ = getpid();
#endif
  for (size_t i = 0; i < thread_num; i++) {
    workers_.emplace_back(ThreadEntry, this, name + "_T" + std::to_string(i));
  }
}

LockBasedThreadPool::~LockBasedThreadPool() {
#ifdef _WIN32
  auto cur_pid = GetCurrentProcessId();
#else
  auto cur_pid = getpid();
#endif
  if (cur_pid == belong_to_pid_) {
    {
      Lock lock(mutex_);
      stop_ = true;
      cond_.notify_all();
    }

    for (auto& t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
  } else {
    // After fork, the child process inherits the data-structures of the parent
    // process' thread-pool, but since those threads don't exist, the thread-pool
    // is corrupt. So detach thread here in order to prevent segfaults.
    for (auto& t : workers_) {
      t.detach();
    }
  }
}

void LockBasedThreadPool::Enqueue(IRunnablePtr& task, size_t seq) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.emplace(task);
  }
  cond_.notify_one();
}

void LockBasedThreadPool::EnqueueBulk(std::vector<IRunnablePtr>& tasks) {
  for (auto& task : tasks) {
    Enqueue(task, 0);
  }
}

size_t LockBasedThreadPool::GetThreadsNum() const {
  return workers_.size();
}

std::vector<std::thread::id> LockBasedThreadPool::GetThreadIds() const {
  std::vector<std::thread::id> ids;
  for (auto& w : workers_) {
    ids.push_back(w.get_id());
  }
  return ids;
}

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript