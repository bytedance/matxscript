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

#include <opencv2/calib3d.hpp>
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

class VisionEstimateAffinePartial2DOpCPU : VisionBaseOpCPU {
 public:
  VisionEstimateAffinePartial2DOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
  }
  ~VisionEstimateAffinePartial2DOpCPU() = default;

  RTValue process(const NDArray& from,
                  const NDArray& to,
                  int method = cv::RANSAC,
                  double ransacReprojThreshold = 3,
                  size_t maxIters = 2000,
                  double confidence = 0.99,
                  size_t refineIters = 10);
};

RTValue VisionEstimateAffinePartial2DOpCPU::process(const NDArray& from,
                                                    const NDArray& to,
                                                    int method,
                                                    double ransacReprojThreshold,
                                                    size_t maxIters,
                                                    double confidence,
                                                    size_t refineIters) {
  cv::Mat&& from_mat = NDArrayToOpencvMat(from);
  cv::Mat&& to_mat = NDArrayToOpencvMat(to);
  cv::Mat inliers;
  auto&& retval = cv::estimateAffinePartial2D(std::move(from_mat),
                                              std::move(to_mat),
                                              inliers,
                                              method,
                                              ransacReprojThreshold,
                                              maxIters,
                                              confidence,
                                              refineIters);
  return Tuple({OpencvMatToNDArray(std::move(retval)), OpencvMatToNDArray(std::move(inliers))});
}

class VisionEstimateAffinePartial2DGeneralOp : public VisionBaseOp {
 public:
  VisionEstimateAffinePartial2DGeneralOp(PyArgs args)
      : VisionBaseOp(args, "VisionEstimateAffinePartial2DOp") {
  }
  ~VisionEstimateAffinePartial2DGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionEstimateAffinePartial2DOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionEstimateAffinePartial2DOpCPU] Expect 1 argument but get "
                                << args.size();
      return std::make_shared<VisionEstimateAffinePartial2DOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 7)
          << "[VisionEstimateAffinePartial2DOpCPU][func: process] Expect 7 arguments but get "
          << args.size();
      return reinterpret_cast<VisionEstimateAffinePartial2DOpCPU*>(self)->process(
          args[0].AsObjectView<NDArray>().data(),
          args[1].AsObjectView<NDArray>().data(),
          args[2].As<int>(),
          args[3].As<double>(),
          args[4].As<size_t>(),
          args[5].As<double>(),
          args[6].As<size_t>());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionEstimateAffinePartial2DGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionEstimateAffinePartial2DGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionEstimateAffinePartial2DGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision