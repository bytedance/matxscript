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

#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/threadpool/i_thread_pool.h>

namespace matxscript {
namespace runtime {

class ThreadPoolOp : public OpKernel {
 public:
  // Before fork, the parent process should call this function once
  void AtForkBefore();
  // After fork, the child/parent process should call this function once
  void AtForkAfterInParentOrChild();

  void Init() override;
  const std::shared_ptr<internal::IThreadPool>& GetPool() const {
    return pool_;
  }
  RTValue Process(PyArgs inputs) const override;

 private:
  bool lock_free_ = false;
  int32_t thread_nums_ = false;
  Unicode thread_name_;
  std::shared_ptr<internal::IThreadPool> pool_ = nullptr;
};
}  // namespace runtime
}  // namespace matxscript