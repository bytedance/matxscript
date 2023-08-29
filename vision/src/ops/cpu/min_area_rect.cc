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

class VisionMinAreaRectOpCPU : VisionBaseOpCPU {
 public:
  VisionMinAreaRectOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
  }
  ~VisionMinAreaRectOpCPU() = default;

  RTValue process(const NDArray& points);
};

RTValue VisionMinAreaRectOpCPU::process(const NDArray& points) {
  cv::Mat point_mat = NDArrayToOpencvMat(points);
  const auto& rotated_rect = cv::minAreaRect(point_mat);
  Tuple center({rotated_rect.center.x, rotated_rect.center.y});
  Tuple size({rotated_rect.size.width, rotated_rect.size.height});
  auto angle = rotated_rect.angle;
  return Tuple({center, size, angle});
}

class VisionMinAreaRectGeneralOp : public VisionBaseOp {
 public:
  VisionMinAreaRectGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionMinAreaRectOp") {
  }
  ~VisionMinAreaRectGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionMinAreaRectOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionMinAreaRectOpCPU] Expect 1 argument but get "
                                << args.size();
      return std::make_shared<VisionMinAreaRectOpCPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 1)
                            << "[VisionMinAreaRectOpCPU][func: process] Expect 1 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionMinAreaRectOpCPU*>(self)->process(
                            args[0].AsObjectView<NDArray>().data());
                      });

MATX_REGISTER_NATIVE_OBJECT(VisionMinAreaRectGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionMinAreaRectGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionMinAreaRectGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision
