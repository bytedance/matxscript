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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/threadpool/i_thread_pool.h>

#include <memory>
#include <unordered_set>

namespace matxscript {
namespace runtime {

class ThreadPoolExecutor {
 public:
  explicit ThreadPoolExecutor(const std::shared_ptr<internal::IThreadPool>& pool, bool lock_free);
  ~ThreadPoolExecutor() = default;

  List ParallelFor(const UserDataRef& op,
                   const List& inputs,
                   int64_t expt_num_threads,
                   int64_t group_size);
  List ParallelFor(const UserDataRef& op, const List& inputs);

  Tuple ParallelFor(const UserDataRef& op,
                    const Tuple& inputs,
                    int64_t expt_num_threads,
                    int64_t group_size);
  Tuple ParallelFor(const UserDataRef& op, const Tuple& inputs);

  List ParallelStarMap(const UserDataRef& op,
                       const List& inputs,
                       int64_t expt_num_threads,
                       int64_t group_size);
  List ParallelStarMap(const UserDataRef& op, const List& inputs);

  Tuple ParallelStarMap(const UserDataRef& op,
                        const Tuple& inputs,
                        int64_t expt_num_threads,
                        int64_t group_size);
  Tuple ParallelStarMap(const UserDataRef& op, const Tuple& inputs);

  RTValue ApplyAsync(const UserDataRef& op, const PyArgs& args);

  RTValue Submit(PyArgs args);

 private:
  void ParallelForImpl(const UserDataRef& op,
                       const Any* inputs_begin,
                       const Any* inputs_end,
                       int64_t expt_num_threads,
                       int64_t group_size,
                       RTValue* outputs_begin,
                       bool unpack_args);

 private:
  bool lock_free_ = true;
  int thread_num_ = 0;
  std::shared_ptr<internal::IThreadPool> pool_ = nullptr;
  std::atomic<size_t> serial_{0};
  std::unordered_set<std::thread::id> pool_thread_ids_;
};

}  // namespace runtime
}  // namespace matxscript
