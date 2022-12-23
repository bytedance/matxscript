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
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/py_args.h"
#include "matxscript/runtime/runtime_value.h"
#include "ops/base/vision_base_op.h"
#include "utils/ndarray_helper.h"
#include "utils/opencv_util.h"
#include "utils/type_helper.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

class VisionCastOpCPU : VisionBaseOpCPU {
 public:
  VisionCastOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
  }
  ~VisionCastOpCPU() = default;

  RTValue process(const NDArray& images, const unicode_view& dtype, double alpha, double beta);
};

RTValue VisionCastOpCPU::process(const NDArray& input_images,
                                 const unicode_view& dtype_view,
                                 double alpha,
                                 double beta) {
  // todo 放到resource_manager里？
  cv::setNumThreads(0);
  const Unicode& dtype = dtype_view.data();
  return cast_type(input_images, dtype, alpha, beta);
}

class VisionCastGeneralOp : public VisionBaseOp {
 public:
  VisionCastGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionCastOp") {
  }
  ~VisionCastGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionCastOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionCastOpCPU] Expect 1 argument but get " << args.size();
      return std::make_shared<VisionCastOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 5) << "[VisionCastOpCPU][func: process] Expect 5 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionCastOpCPU*>(self)->process(
          args[0].AsObjectView<NDArray>().data(),
          args[1].As<unicode_view>(),
          args[2].As<double>(),
          args[3].As<double>());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionCastGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionCastGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionCastGeneralOp*>(self)->process(args);
    });
}  // namespace ops
}  // namespace byted_matx_vision
