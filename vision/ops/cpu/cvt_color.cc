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
#include "utils/opencv_util.h"
#include "utils/task_manager.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

struct CvtColorTaskInput {
  CvtColorTaskInput(NDArray image, int color_code)
      : image_(std::move(image)), color_code_(color_code) {
  }

  NDArray image_;
  int color_code_;
};

using CvtColorTaskInputPtr = std::shared_ptr<CvtColorTaskInput>;

class CvtColorTask : public internal::LockBasedRunnable {
 public:
  CvtColorTask(std::vector<CvtColorTaskInputPtr>::iterator first_input,
               std::vector<NDArray>::iterator first_output,
               int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<CvtColorTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void CvtColorTask::RunImpl() {
  std::vector<CvtColorTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    CvtColorTaskInputPtr cvt_color_task_input_ptr = (*input_it);

    cv::Mat mat_src = NDArrayToOpencvMat(cvt_color_task_input_ptr->image_);
    int color_code = cvt_color_task_input_ptr->color_code_;

    cv::Mat mat_dst;
    cv::cvtColor(mat_src, mat_dst, color_code);
    (*output_it) = OpencvMatToNDArray(mat_dst);
  }
};

class VisionCvtColorOpCPU : VisionBaseOpCPU {
 public:
  VisionCvtColorOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
    task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  }
  ~VisionCvtColorOpCPU() = default;

  RTValue process(const List& images, const unicode_view& code);

 private:
  TaskManagerPtr task_manager_ptr = nullptr;
};

RTValue VisionCvtColorOpCPU::process(const List& images, const unicode_view& color_code) {
  int batch_size = images.size();
  // construct cvt_color_task_input
  std::vector<CvtColorTaskInputPtr> cvt_color_task_inputs;
  cvt_color_task_inputs.reserve(batch_size);

  int cv_color_code = UnicodeToOpencvColorCode(color_code);
  for (int i = 0; i < batch_size; ++i) {
    auto nd_view = images[i].As<NDArray>();
    cvt_color_task_inputs.emplace_back(std::make_shared<CvtColorTaskInput>(nd_view, cv_color_code));
  }

  List ret;

  std::vector<NDArray> cvt_color_task_outputs =
      task_manager_ptr->Execute<CvtColorTask, CvtColorTaskInputPtr, NDArray>(cvt_color_task_inputs,
                                                                             batch_size);
  ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ret.append(std::move(cvt_color_task_outputs[i]));
  }
  return ret;
}

class VisionCvtColorGeneralOp : public VisionBaseOp {
 public:
  VisionCvtColorGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionCvtColorOp") {
  }
  ~VisionCvtColorGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionCvtColorOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionCvtColorOpCPU] Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionCvtColorOpCPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        // last argument is sync which does not work in cpu scene
                        MXCHECK_EQ(args.size(), 3)
                            << "[VisionCvtColorOpCPU][func: process] Expect 3 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionCvtColorOpCPU*>(self)->process(
                            args[0].AsObjectView<List>().data(), args[1].As<unicode_view>());
                      });

MATX_REGISTER_NATIVE_OBJECT(VisionCvtColorGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionCvtColorGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionCvtColorGeneralOp*>(self)->process(args);
    });
}  // namespace ops
}  // namespace byted_matx_vision
