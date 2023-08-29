// Copyright 2023 ByteDance Ltd. and/or its affiliates.
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

#include <opencv2/imgproc.hpp>
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

class VisionApproxPolyDPOpCPU : VisionBaseOpCPU {
 public:
  VisionApproxPolyDPOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
  }
  ~VisionApproxPolyDPOpCPU() = default;

  RTValue process(const NDArray& curve, double epsilon, bool closed);
};

RTValue VisionApproxPolyDPOpCPU::process(const NDArray& curve, double epsilon, bool closed) {
  cv::Mat&& curve_mat = NDArrayToOpencvMat(curve);
  cv::Mat approx_mat;

  cv::approxPolyDP(curve_mat, approx_mat, epsilon, closed);

  return OpencvMatToNDArray(approx_mat);
}

class VisionApproxPolyDPGeneralOp : public VisionBaseOp {
 public:
  VisionApproxPolyDPGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionApproxPolyDPOp") {
  }
  ~VisionApproxPolyDPGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionApproxPolyDPOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionApproxPolyDPOpCPU] Expect 1 argument but get "
                                << args.size();
      return std::make_shared<VisionApproxPolyDPOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 3)
          << "[VisionApproxPolyDPOpCPU][func: process] Expect 3 arguments but get " << args.size();
      return reinterpret_cast<VisionApproxPolyDPOpCPU*>(self)->process(
          args[0].AsObjectView<NDArray>().data(), args[1].As<double>(), args[2].As<bool>());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionApproxPolyDPGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionApproxPolyDPGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionApproxPolyDPGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision
