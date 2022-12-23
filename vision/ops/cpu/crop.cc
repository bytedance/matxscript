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
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/py_args.h"
#include "matxscript/runtime/runtime_value.h"
#include "matxscript/runtime/threadpool/lock_based_thread_pool.h"
#include "ops/base/vision_base_op.h"
#include "utils/opencv_util.h"
#include "utils/task_manager.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

struct CropTaskInput {
  CropTaskInput(NDArray image, std::vector<int> crop_param)
      : image_(std::move(image)), crop_params_(std::move(crop_param)) {
  }

  NDArray image_;
  std::vector<int> crop_params_;
};

using CropTaskInputPtr = std::shared_ptr<CropTaskInput>;

class CropTask : public internal::LockBasedRunnable {
 public:
  CropTask(std::vector<CropTaskInputPtr>::iterator first_input,
           std::vector<NDArray>::iterator first_output,
           int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<CropTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void CropTask::RunImpl() {
  std::vector<CropTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    CropTaskInputPtr crop_task_input_ptr = (*input_it);

    cv::Mat org_image = NDArrayToOpencvMat(crop_task_input_ptr->image_);
    cv::Mat crop_image;
    MXCHECK(crop_task_input_ptr->crop_params_.size() == 4)
        << "crop params sizes must be equals to 4 in CropTaskInput .";
    crop_image = org_image(cv::Rect(crop_task_input_ptr->crop_params_[0],
                                    crop_task_input_ptr->crop_params_[1],
                                    crop_task_input_ptr->crop_params_[2],
                                    crop_task_input_ptr->crop_params_[3]));

    (*output_it) = OpencvMatToNDArray(crop_image);
  }
};

class VisionCropOpCPU : public VisionBaseOpCPU {
 public:
  VisionCropOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
    task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  }
  RTValue process(
      const List& images, const List& x, const List& y, const List& widths, const List& heights);

  TaskManagerPtr task_manager_ptr = nullptr;
};

RTValue VisionCropOpCPU::process(
    const List& images, const List& x, const List& y, const List& widths, const List& heights) {
  cv::setNumThreads(0);
  std::vector<CropTaskInputPtr> crop_task_inputs;
  int batch_size = images.size();
  crop_task_inputs.reserve(batch_size);

  MXCHECK(batch_size == x.size() && batch_size == y.size() && batch_size == widths.size() &&
          batch_size == heights.size())
      << "The params sizes must be match in VisionCropOpCPU. ";
  // construct crop_task_input
  for (int i = 0; i < batch_size; ++i) {
    auto nd_view = images[i].As<NDArray>();
    std::vector<int64_t> src_shape = nd_view.Shape();
    int x_pointer = x[i].As<int>();
    int y_pointer = y[i].As<int>();
    int width = widths[i].As<int>();
    int height = heights[i].As<int>();
    MXCHECK(0 <= x_pointer && 0 <= width && x_pointer + width <= src_shape[1])
        << "X + Width should be less than or equal to image width, but get : " << x_pointer + width
        << ", origin image width: " << src_shape[1];

    MXCHECK(0 <= y_pointer && 0 <= height && y_pointer + height <= src_shape[0])
        << "Y + Height should be less than or equal to image height, but get : "
        << y_pointer + height << ", origin image height: " << src_shape[0];
    std::vector<int> parmes({x_pointer, y_pointer, width, height});
    crop_task_inputs.emplace_back(std::make_shared<CropTaskInput>(nd_view, std::move(parmes)));
  }

  List nd_ret;
  std::vector<NDArray> crop_task_outputs =
      task_manager_ptr->Execute<CropTask, CropTaskInputPtr, NDArray>(crop_task_inputs, batch_size);

  nd_ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    nd_ret.append(std::move(crop_task_outputs[i]));
  }
  return nd_ret;
};

class VisionCropGeneralOp : public VisionBaseOp {
 public:
  VisionCropGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionCropOp") {
  }
  ~VisionCropGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionCropGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionCropGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionCropGeneralOp*>(self)->process(args);
    });

MATX_REGISTER_NATIVE_OBJECT(VisionCropOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[BytedVisionCropOpCPU] Constructor Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionCropOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 6) << "[BytedVisionCropOpCPU] Expect 6 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionCropOpCPU*>(self)->process(args[0].AsObjectView<List>().data(),
                                                               args[1].AsObjectView<List>().data(),
                                                               args[2].AsObjectView<List>().data(),
                                                               args[3].AsObjectView<List>().data(),
                                                               args[4].AsObjectView<List>().data());
    });

}  // namespace ops
}  // namespace byted_matx_vision