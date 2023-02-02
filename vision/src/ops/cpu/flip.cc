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

struct FlipTaskInput {
  FlipTaskInput(NDArray image, int flip_code) : image_(std::move(image)), flip_code_(flip_code) {
  }

  NDArray image_;
  int flip_code_;
};

using FlipTaskInputPtr = std::shared_ptr<FlipTaskInput>;

class FlipTask : public internal::LockBasedRunnable {
 public:
  FlipTask(std::vector<FlipTaskInputPtr>::iterator first_input,
           std::vector<NDArray>::iterator first_output,
           int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<FlipTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void FlipTask::RunImpl() {
  std::vector<FlipTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    FlipTaskInputPtr flip_task_input_ptr = (*input_it);

    cv::Mat mat_src = NDArrayToOpencvMat(flip_task_input_ptr->image_);
    int flip_code = flip_task_input_ptr->flip_code_;

    if (flip_code != -1 && flip_code != 0 and flip_code != 1)
      (*output_it) = OpencvMatToNDArray(mat_src);
    else {
      cv::Mat mat_dst;
      cv::flip(mat_src, mat_dst, flip_code);
      (*output_it) = OpencvMatToNDArray(mat_dst);
    }
  }
};

class VisionFlipOpCPU : VisionBaseOpCPU {
 public:
  VisionFlipOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
    task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  }
  ~VisionFlipOpCPU() = default;

  RTValue process(const List& images, const Any& code);

 private:
  TaskManagerPtr task_manager_ptr = nullptr;
};

RTValue VisionFlipOpCPU::process(const List& images, const Any& code) {
  int batch_size = images.size();
  // construct flip_task_input
  std::vector<FlipTaskInputPtr> flip_task_inputs;
  flip_task_inputs.reserve(batch_size);
  if (code.type_code() == TypeIndex::kRuntimeList) {
    auto view = code.AsObjectViewNoCheck<List>();
    const auto& code_list = view.data();
    MXCHECK(images.size() == code_list.size()) << "VisionFilpOP: input size not match";
    for (int i = 0; i < batch_size; ++i) {
      auto nd_view = images[i].As<NDArray>();
      int cur_flip_code = code_list[i].As<int>();
      flip_task_inputs.emplace_back(std::make_shared<FlipTaskInput>(nd_view, cur_flip_code));
    }
  } else {
    int flip_code = code.As<int>();
    for (int i = 0; i < batch_size; ++i) {
      auto nd_view = images[i].As<NDArray>();
      flip_task_inputs.emplace_back(std::make_shared<FlipTaskInput>(nd_view, flip_code));
    }
  }

  List ret;

  std::vector<NDArray> flip_task_outputs =
      task_manager_ptr->Execute<FlipTask, FlipTaskInputPtr, NDArray>(flip_task_inputs, batch_size);
  ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ret.append(std::move(flip_task_outputs[i]));
  }
  return ret;
}

class VisionFlipGeneralOp : public VisionBaseOp {
 public:
  VisionFlipGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionFlipOp") {
  }
  ~VisionFlipGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionFlipOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionFlipOpCPU] Expect 1 arguments but get " << args.size();
      return std::make_shared<VisionFlipOpCPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      // last argument is sync which does not work in cpu scene
      MXCHECK_EQ(args.size(), 3) << "[VisionFlipOpCPU][func: process] Expect 3 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionFlipOpCPU*>(self)->process(args[0].AsObjectView<List>().data(),
                                                               args[1]);
    });

MATX_REGISTER_NATIVE_OBJECT(VisionFlipGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionFlipGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionFlipGeneralOp*>(self)->process(args);
    });
}  // namespace ops
}  // namespace byted_matx_vision
