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
#include <opencv2/imgproc.hpp>

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

class VisionBoxPointsOpCPU : VisionBaseOpCPU {
 public:
  VisionBoxPointsOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
  }
  ~VisionBoxPointsOpCPU() = default;

  RTValue process(const Tuple & rotated_rect_tuple);
};

RTValue VisionBoxPointsOpCPU::process(const Tuple & rotated_rect_tuple) {
  const auto & center_tuple_view = rotated_rect_tuple[0].AsObjectView<Tuple>();
  const auto & center_tuple = center_tuple_view.data();
  const auto & size_tuple_view = rotated_rect_tuple[1].AsObjectView<Tuple>();
  const auto & size_tuple = size_tuple_view.data();
  cv::Point2f center(center_tuple[0].As<float>(), center_tuple[1].As<float>());
  cv::Size2f size(size_tuple[0].As<float>(), size_tuple[1].As<float>());
  float angle = rotated_rect_tuple[2].As<float>();
  cv::RotatedRect rotated_rect(std::move(center), std::move(size), std::move(angle));
  cv::Mat point_mat;
  cv::boxPoints(std::move(rotated_rect), point_mat);
  return OpencvMatToNDArray(std::move(point_mat));
}


class VisionBoxPointsGeneralOp : public VisionBaseOp {
 public:
  VisionBoxPointsGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionBoxPointsOp") {
  }
  ~VisionBoxPointsGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionBoxPointsOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionBoxPointsOpCPU] Expect 1 argument but get " << args.size();
      return std::make_shared<VisionBoxPointsOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 1) << "[VisionBoxPointsOpCPU][func: process] Expect 1 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionBoxPointsOpCPU*>(self)->process(
          args[0].AsObjectView<Tuple>().data());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionBoxPointsGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionBoxPointsGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionBoxPointsGeneralOp*>(self)->process(args);
    });



}  // namespace ops
}  // namespace byted_matx_vision
