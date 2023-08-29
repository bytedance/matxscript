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

#include <opencv2/core/mat.hpp>
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

class VisionFindContoursOpCPU : VisionBaseOpCPU {
 public:
  VisionFindContoursOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
  }
  ~VisionFindContoursOpCPU() = default;

  RTValue process(
      const NDArray& image, int mode, int method, double offset_x = 0, double offset_y = 0);
};

RTValue VisionFindContoursOpCPU::process(
    const NDArray& image, int mode, int method, double offset_x, double offset_y) {
  cv::Mat&& image_mat = NDArrayToOpencvMat(image);
  std::vector<cv::Mat> contours;
  cv::Mat hierarchy;
  cv::Point offset(offset_x, offset_y);
  cv::findContours(image_mat, contours, hierarchy, mode, method);
  NDArray&& hierarchy_nd = OpencvMatToNDArray(hierarchy);
  List contour_nds;
  contour_nds.reserve(contours.size());
  for (auto& contour : contours) {
    contour_nds.push_back(std::move(OpencvMatToNDArray(contour)));
  }
  // std::pair<std::vector<NDArray>, NDArray> p(std::move(contour_nds), std::move(hierarchy_nd));
  return Tuple::dynamic(std::move(contour_nds), std::move(hierarchy_nd));
}

class VisionFindContoursGeneralOp : public VisionBaseOp {
 public:
  VisionFindContoursGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionFindContoursOp") {
  }
  ~VisionFindContoursGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionFindContoursOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionFindContoursOpCPU] Expect 1 argument but get "
                                << args.size();
      return std::make_shared<VisionFindContoursOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 5)
          << "[VisionFindContoursOpCPU][func: process] Expect 5 arguments but get " << args.size();
      return reinterpret_cast<VisionFindContoursOpCPU*>(self)->process(
          args[0].AsObjectView<NDArray>().data(),
          args[1].As<int>(),
          args[2].As<int>(),
          args[3].As<double>(),
          args[4].As<double>());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionFindContoursGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionFindContoursGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionFindContoursGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision