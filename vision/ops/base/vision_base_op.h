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
#include <matxscript/pipeline/op_kernel.h>
#include <matxscript/runtime/native_object_registry.h>
#include "matxscript/pipeline/attributes.h"
#include "matxscript/runtime/py_args.h"

namespace byted_matx_vision {
namespace ops {

class VisionBaseOp {
 public:
  VisionBaseOp(::matxscript::runtime::PyArgs args, ::matxscript::runtime::string_view name);

  ::matxscript::runtime::RTValue process(matxscript::runtime::PyArgs args);

 protected:
  ::matxscript::runtime::String name_;
  int device_id_ = matxscript::runtime::NONE_DEVICE;
  int session_device_id_ = matxscript::runtime::NONE_DEVICE;
  std::shared_ptr<void> obj_ = nullptr;
  ::matxscript::runtime::NativeObjectRegistry::NativeMethod* process_ = nullptr;
};

}  // namespace ops
}  // namespace byted_matx_vision