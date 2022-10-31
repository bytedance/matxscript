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
#include <vector>
#include "i_runnable.h"

namespace matxscript {
namespace runtime {
namespace internal {

struct IThreadPool {
  virtual ~IThreadPool() = default;
  virtual void Enqueue(IRunnablePtr& runner, size_t seq) = 0;
  virtual void EnqueueBulk(std::vector<IRunnablePtr>& runners) {
    for (size_t i = 0; i < runners.size(); ++i) {
      Enqueue(runners[i], i);
    }
  }
  virtual size_t GetThreadsNum() const = 0;
  virtual std::vector<std::thread::id> GetThreadIds() const = 0;
  static void WaitBulk(std::vector<IRunnablePtr>& runners);
};

}  // namespace internal
}  // namespace runtime
}  // namespace matxscript
