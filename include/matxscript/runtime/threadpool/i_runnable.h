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

#include <atomic>
#include <memory>
#include <thread>

namespace matxscript {
namespace runtime {
namespace internal {

class IRunnable {
 public:
  inline void Run() noexcept {
    try {
      this->RunImpl();
    } catch (...) {
      e_ = std::current_exception();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    try {
      this->SetDone();
    } catch (...) {
      if (!e_) {
        e_ = std::current_exception();
      }
    }
  }

  inline void Wait() {
    while (!Done()) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
    if (e_) {
      std::rethrow_exception(e_);
    }
  }

  virtual bool Done() = 0;

 protected:
  virtual void RunImpl() = 0;
  virtual void SetDone() = 0;

 protected:
  std::exception_ptr e_;
};

using IRunnablePtr = std::shared_ptr<IRunnable>;

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript
