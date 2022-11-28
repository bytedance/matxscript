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

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
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

namespace matxscript {
namespace runtime {
namespace internal {

class LockBasedRunnable : public IRunnable {
 public:
  bool Done() override {
    return finish_;
  }

 protected:
  void SetDone() override {
    std::lock_guard<std::mutex> lock(mutex_);
    finish_ = true;
  }

 private:
  std::mutex mutex_;
  bool finish_ = false;
};

class LockBasedThreadPool : public IThreadPool {
 public:
  typedef std::unique_lock<std::mutex> Lock;

  LockBasedThreadPool(size_t thread_num, const std::string& name);
  ~LockBasedThreadPool() override;

  void Enqueue(IRunnablePtr& runner, size_t seq) override;

  void EnqueueBulk(std::vector<IRunnablePtr>& runners) override;

  size_t GetThreadsNum() const override;
  std::vector<std::thread::id> GetThreadIds() const override;

 private:
  static void ThreadEntry(LockBasedThreadPool* pool, const std::string& name);

 private:
  std::vector<std::thread> workers_;
  std::queue<IRunnablePtr> tasks_;

  bool stop_ = false;

  std::mutex mutex_;
  std::condition_variable cond_;
  pid_t belong_to_pid_;
};

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript
