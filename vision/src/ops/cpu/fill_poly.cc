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

class VisionFillPolyOpCPU : VisionBaseOpCPU {
 public:
  VisionFillPolyOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
  }
  ~VisionFillPolyOpCPU() = default;

  RTValue process(const NDArray& img,
                  const List& pts,
                  const Tuple& color,
                  int lineType,
                  int shift,
                  int offset_x = 0,
                  int offset_y = 0);
};

RTValue VisionFillPolyOpCPU::process(const NDArray& img,
                                     const List& pts,
                                     const Tuple& color,
                                     int lineType,
                                     int shift,
                                     int offset_x,
                                     int offset_y) {
  cv::Mat&& img_mat = NDArrayToOpencvMat(img);
  std::vector<cv::Mat> pts_vector;
  pts_vector.reserve(pts.size());
  for (auto& rt_nd : pts) {
    auto&& nd_view = rt_nd.AsObjectView<NDArray>();
    auto&& nd = nd_view.data();
    const auto& dtype = nd.DataType();
    MXCHECK(dtype.is_int() && dtype.bits() == 32) << "The input points have to be of int32";
    pts_vector.push_back(std::move(NDArrayToOpencvMat(nd)));
  }
  cv::Scalar color_scalar;
  switch (color.size()) {
    case 0:
      color_scalar = cv::Scalar(0);
      break;
    case 1:
      color_scalar = cv::Scalar(color[0].As<int>());
      break;
    case 3:
      color_scalar = cv::Scalar(color[0].As<int>(), color[1].As<int>(), color[2].As<int>());
      break;
    case 4:
      color_scalar = cv::Scalar(
          color[0].As<int>(), color[1].As<int>(), color[2].As<int>(), color[3].As<int>());
      break;
    default:
      MXCHECK(false)
          << "VisionFillPolyOpCPU expect color to be a tuple of size 0, 1, 3, or 4. But color.size() = "
          << color.size();
      break;
  }
  cv::Point offset(offset_x, offset_y);
  cv::fillPoly(
      std::move(img_mat), std::move(pts_vector), std::move(color_scalar), lineType, shift, offset);
  return img;
}

class VisionFillPolyGeneralOp : public VisionBaseOp {
 public:
  VisionFillPolyGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionFillPolyOp") {
  }
  ~VisionFillPolyGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionFillPolyOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionFillPolyOpCPU] Expect 1 argument but get "
                                << args.size();
      return std::make_shared<VisionFillPolyOpCPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 7)
                            << "[VisionFillPolyOpCPU][func: process] Expect 7 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionFillPolyOpCPU*>(self)->process(
                            args[0].AsObjectView<NDArray>().data(),
                            args[1].AsObjectView<List>().data(),
                            args[2].AsObjectView<Tuple>().data(),
                            args[3].As<int>(),
                            args[4].As<int>(),
                            args[5].As<int>(),
                            args[6].As<int>());
                      });

MATX_REGISTER_NATIVE_OBJECT(VisionFillPolyGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionFillPolyGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionFillPolyGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision
