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
#pragma once

#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
typedef DWORD pid_t;
#else
#include <unistd.h>
#endif

#include <matxscript/runtime/threadpool/i_thread_pool.h>
#include <matxscript/runtime/threadpool/mpmc_bounded_queue.h>

namespace matxscript {
namespace runtime {
namespace internal {

class LockFreeRunnable : public IRunnable {
 public:
  bool Done() override {
    return finish_;
  }

 protected:
  void SetDone() override {
    finish_ = true;
  }

 private:
  volatile bool finish_ = false;
  friend class LockFreeThreadPool;
};

class LockFreeThreadPool : public IThreadPool {
 public:
  explicit LockFreeThreadPool(size_t threads, const std::string& name, int64_t intervals_ns);
  explicit LockFreeThreadPool(size_t threads, const std::string& name);
  ~LockFreeThreadPool() override;

  void Enqueue(IRunnablePtr& runner, size_t seq) override;

  void EnqueueBulk(std::vector<IRunnablePtr>& runners) override;

  size_t GetThreadsNum() const override;
  std::vector<std::thread::id> GetThreadIds() const override;

 protected:
  static void ThreadEntry(LockFreeThreadPool* pool, const std::string& name);

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers_;
  // the task queue
  MPMCBoundedQueue<IRunnablePtr> tasks_;
  // stop flag
  bool stop_ = false;
  std::string name_;
  int64_t intervals_ns_;
  pid_t belong_to_pid_;
};

class SPSCLockFreeThreadPool : public IThreadPool {
 public:
  explicit SPSCLockFreeThreadPool(size_t threads,
                                  const std::string& name,
                                  int64_t intervals_ns = 1);
  ~SPSCLockFreeThreadPool() override;

  void Enqueue(IRunnablePtr& runner, size_t seq) override;

  size_t GetThreadsNum() const override;
  std::vector<std::thread::id> GetThreadIds() const override;

 private:
  std::vector<std::unique_ptr<LockFreeThreadPool>> workers_;
};

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript
