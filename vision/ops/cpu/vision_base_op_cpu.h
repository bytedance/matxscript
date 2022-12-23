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
#include <matxscript/runtime/native_object_registry.h>
#include "matxscript/runtime/container/dict_ref.h"
#include "matxscript/runtime/py_args.h"
#include "matxscript/runtime/threadpool/i_thread_pool.h"

namespace byted_matx_vision {
namespace ops {

class VisionBaseOpCPU {
 public:
  VisionBaseOpCPU(const ::matxscript::runtime::Any& session_info) {
    auto view = session_info.AsObjectView<::matxscript::runtime::Dict>();
    const ::matxscript::runtime::Dict& info = view.data();
    void* pool = info["thread_pool"].As<void*>();
    thread_pool_ = static_cast<::matxscript::runtime::internal::IThreadPool*>(pool);
  }
  ~VisionBaseOpCPU() = default;

 protected:
  ::matxscript::runtime::internal::IThreadPool* thread_pool_;
};

}  // namespace ops
}  // namespace byted_matx_vision