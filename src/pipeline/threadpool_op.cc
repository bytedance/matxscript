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
#include <matxscript/pipeline/threadpool_op.h>

#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/threadpool/lock_based_thread_pool.h>
#include <matxscript/runtime/threadpool/lock_free_thread_pool.h>

namespace matxscript {
namespace runtime {

void ThreadPoolOp::Init() {
  lock_free_ = GetAttr<bool>("lock_free");
  thread_nums_ = GetAttr<int32_t>("thread_nums");
  thread_name_ = GetAttr<Unicode>("thread_name");
  auto thread_name = thread_name_.encode();
  if (lock_free_) {
    pool_ = std::make_shared<internal::SPSCLockFreeThreadPool>(
        thread_nums_, std::string(thread_name.data(), thread_name.size()));
  } else {
    pool_ = std::make_shared<internal::LockBasedThreadPool>(
        thread_nums_, std::string(thread_name.data(), thread_name.size()));
  }
}

RTValue ThreadPoolOp::Process(PyArgs inputs) const {
  MXTHROW << "ThreadPoolOp can not be call directly!!!";
  return None;
}

MATX_REGISTER_NATIVE_OP(ThreadPoolOp);

}  // namespace runtime
}  // namespace matxscript