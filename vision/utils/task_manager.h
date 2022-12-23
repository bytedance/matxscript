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

#include <cstdint>
#include <exception>
#include <memory>
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/threadpool/i_runnable.h"
#include "matxscript/runtime/threadpool/lock_based_thread_pool.h"

namespace byted_matx_vision {
namespace ops {

using namespace matxscript::runtime;
using ThreadPoolPtr = ::matxscript::runtime::internal::IThreadPool*;

class TaskManager {
 public:
  TaskManager(ThreadPoolPtr pool) : thread_pool_ptr_(pool) {
    if (pool != nullptr) {
      thread_pool_size_ = pool->GetThreadsNum();
    }
    thread_pool_size_ += 1;  // All threads contain main-thread
  }
  ~TaskManager() = default;

  template <class T, class I, class O>
  std::vector<O> Execute(std::vector<I>& inputs, int64_t len);

 private:
  ThreadPoolPtr thread_pool_ptr_;
  int64_t thread_pool_size_ = 0;
};

template <class T, class I, class O>
std::vector<O> TaskManager::Execute(std::vector<I>& inputs, int64_t len) {
  std::vector<O> outputs(inputs.size());
  int64_t input_size = inputs.size();

  if (input_size == 0) {
    return outputs;
  }
  // 1. build tasks
  std::vector<internal::IRunnablePtr> tasks;
  auto input_itr_first = inputs.begin();
  auto output_itr_first = outputs.begin();

  if (len <= thread_pool_size_) {
    tasks.reserve(len);
    for (int i = 0; i < len; ++i) {
      tasks.emplace_back(std::make_shared<T>(input_itr_first + i, output_itr_first + i, 1));
    }
  } else {
    tasks.reserve(thread_pool_size_);
    int step = len / thread_pool_size_;
    int remainder = len % thread_pool_size_;
    for (int i = 0; i < remainder; ++i) {
      tasks.emplace_back(std::make_shared<T>(input_itr_first, output_itr_first, step + 1));
      input_itr_first += step + 1;
      output_itr_first += step + 1;
    }
    for (int i = remainder; i < thread_pool_size_; ++i) {
      tasks.emplace_back(std::make_shared<T>(input_itr_first, output_itr_first, step));
      input_itr_first += step;
      output_itr_first += step;
    }
  }
  // 2. run
  if (thread_pool_ptr_ != nullptr) {
    for (size_t i = 1; i < tasks.size(); ++i) {
      thread_pool_ptr_->Enqueue(tasks[i], 0);
    }
  }
  tasks[0]->Run();
  std::exception_ptr eptr;
  for (size_t i = 0; i < tasks.size(); ++i) {
    try {
      tasks[i]->Wait();
    } catch (...) {
      if (!eptr) {
        // store first exception
        eptr = std::current_exception();
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
  return outputs;
};

using TaskManagerPtr = std::shared_ptr<TaskManager>;

}  // namespace ops
}  // namespace byted_matx_vision