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

#include "vision_base_op.h"
#include "matxscript/pipeline/attributes.h"
#include "matxscript/pipeline/internal_helper_funcs.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/py_args.h"
#include "utils/opencv_util.h"

namespace byted_matx_vision {
namespace ops {

using namespace ::matxscript::runtime;

VisionBaseOp::VisionBaseOp(PyArgs args, string_view name) : name_(name) {
  cv::setNumThreads(0);
  auto view = (args.end() - 1)->AsObjectView<Dict>();
  const Dict& session_info = view.data();
  device_id_ = session_info["device_id"].As<int>();
  session_device_id_ = session_info["session_device_id"].As<int>();
  MXCHECK(device_id_ != NONE_DEVICE) << "VisionOp must specify backend(device) info";
  if (device_id_ < 0) {
    // cpu
    auto cpu_native_obj = NativeObjectRegistry::Get(name_ + "CPU");
    if (cpu_native_obj == nullptr) {
      MXTHROW << "CPU Version of " << name_ << " is not supported";
    }
    obj_ = cpu_native_obj->construct(args);
    process_ = &cpu_native_obj->function_table_.at("process");
  } else {
    if (session_device_id_ != NONE_DEVICE && session_device_id_ < 0) {
      MXTHROW << name_ << " is set on gpu, but session is on cpu";
    }
    auto gpu_native_obj = NativeObjectRegistry::Get(name_ + "GPU");
    if (gpu_native_obj == nullptr) {
      MXTHROW << "GPU Version of " << name_ << " is not supported";
    }
    obj_ = gpu_native_obj->construct(args);
    process_ = &gpu_native_obj->function_table_.at("process");
  }
}

RTValue VisionBaseOp::process(PyArgs args) {
  return (*process_)(obj_.get(), args);
}

}  // namespace ops
}  // namespace byted_matx_vision