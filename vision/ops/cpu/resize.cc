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
#include "utils/task_manager.h"
#include "utils/type_helper.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

struct ResizeTaskInput {
  ResizeTaskInput(NDArray image, int height, int width, int interp)
      : image_(std::move(image)), height_(height), width_(width), interp_(interp) {
  }

  NDArray image_;
  int height_;
  int width_;
  int interp_;
};

using ResizeTaskInputPtr = std::shared_ptr<ResizeTaskInput>;

class ResizeTask : public internal::LockBasedRunnable {
 public:
  ResizeTask(std::vector<ResizeTaskInputPtr>::iterator first_input,
             std::vector<NDArray>::iterator first_output,
             int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<ResizeTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void ResizeTask::RunImpl() {
  std::vector<ResizeTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    ResizeTaskInputPtr resize_task_input_ptr = (*input_it);

    cv::Mat mat_src = NDArrayToOpencvMat(resize_task_input_ptr->image_);
    int height = resize_task_input_ptr->height_;
    int width = resize_task_input_ptr->width_;
    int interp_ = resize_task_input_ptr->interp_;
    cv::Mat mat_dst;
    cv::resize(mat_src, mat_dst, cv::Size(width, height), 1.0, 1.0, interp_);
    (*output_it) = OpencvMatToNDArray(mat_dst);
  }
};

class VisionResizeOpCPU : VisionBaseOpCPU {
 public:
  VisionResizeOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
    task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  }
  ~VisionResizeOpCPU() = default;

  RTValue process(const List& images,
                  const List& height,
                  const List& width,
                  const unicode_view& interp);

 private:
  TaskManagerPtr task_manager_ptr = nullptr;
};

RTValue VisionResizeOpCPU::process(const List& images,
                                   const List& desired_height,
                                   const List& desired_width,
                                   const unicode_view& interpolation) {
  int batch_size = images.size();
  MXCHECK_EQ(desired_height.size(), batch_size)
      << "argument desired_height should be equal to batch size";
  MXCHECK_EQ(desired_width.size(), batch_size)
      << "argument desired_width should be equal to batch size";
  int interp_flags = UnicodeToOpencvInterp(interpolation);
  if (interp_flags < 0) {
    MXCHECK(false) << "Invalid interp type for CPU resize op: " << interpolation;
  }

  // construct resize_task_input
  std::vector<ResizeTaskInputPtr> resize_task_inputs;
  resize_task_inputs.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    auto nd_view = images[i].As<NDArray>();
    int cur_height = desired_height[i].As<int>();
    int cur_width = desired_width[i].As<int>();
    resize_task_inputs.emplace_back(
        std::make_shared<ResizeTaskInput>(nd_view, cur_height, cur_width, interp_flags));
  }
  List ret;

  std::vector<NDArray> resize_task_outputs =
      task_manager_ptr->Execute<ResizeTask, ResizeTaskInputPtr, NDArray>(resize_task_inputs,
                                                                         batch_size);
  ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ret.append(std::move(resize_task_outputs[i]));
  }
  return ret;
}

class VisionResizeGeneralOp : public VisionBaseOp {
 public:
  VisionResizeGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionResizeOp") {
  }
  ~VisionResizeGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionResizeOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionResizeOpCPU] Expect 1 argument but get " << args.size();
      return std::make_shared<VisionResizeOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 5) << "[VisionResizeOpCPU][func: process] Expect 5 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionResizeOpCPU*>(self)->process(
          args[0].AsObjectView<List>().data(),
          args[1].AsObjectView<List>().data(),
          args[2].AsObjectView<List>().data(),
          args[3].As<unicode_view>());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionResizeGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionResizeGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionResizeGeneralOp*>(self)->process(args);
    });
}  // namespace ops
}  // namespace byted_matx_vision
