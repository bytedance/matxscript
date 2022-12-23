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
#include "matxscript/runtime/container/tuple_ref.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/object.h"
#include "matxscript/runtime/py_args.h"
#include "matxscript/runtime/runtime_value.h"
#include "matxscript/runtime/threadpool/lock_based_thread_pool.h"
#include "opencv2/core/base.hpp"
#include "ops/base/vision_base_op.h"
#include "utils/opencv_util.h"
#include "utils/pad_types.h"
#include "utils/task_manager.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

struct PadTaskInput {
  PadTaskInput(NDArray image,
               std::vector<int> pad_params,
               std::vector<int> pad_values,
               int cv_border_type)
      : image_(std::move(image)),
        pad_params_(std::move(pad_params)),
        pad_values_(std::move(pad_values)),
        cv_border_type_(cv_border_type) {
  }

  NDArray image_;
  std::vector<int> pad_params_;
  std::vector<int> pad_values_;
  int cv_border_type_;
};

using PadTaskInputPtr = std::shared_ptr<PadTaskInput>;
class PadTask : public internal::LockBasedRunnable {
 public:
  PadTask(std::vector<PadTaskInputPtr>::iterator first_input,
          std::vector<NDArray>::iterator first_output,
          int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<PadTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void PadTask::RunImpl() {
  std::vector<PadTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    PadTaskInputPtr pad_task_ptr = (*input_it);

    cv::Mat org_image = NDArrayToOpencvMat(pad_task_ptr->image_);
    cv::Mat padding_image;
    if (pad_task_ptr->cv_border_type_ == cv::BORDER_CONSTANT) {
      cv::copyMakeBorder(org_image,
                         padding_image,
                         pad_task_ptr->pad_params_[0],
                         pad_task_ptr->pad_params_[1],
                         pad_task_ptr->pad_params_[2],
                         pad_task_ptr->pad_params_[3],
                         pad_task_ptr->cv_border_type_,
                         cv::Scalar(pad_task_ptr->pad_values_[0],
                                    pad_task_ptr->pad_values_[1],
                                    pad_task_ptr->pad_values_[2]));
    } else {
      cv::copyMakeBorder(org_image,
                         padding_image,
                         pad_task_ptr->pad_params_[0],
                         pad_task_ptr->pad_params_[1],
                         pad_task_ptr->pad_params_[2],
                         pad_task_ptr->pad_params_[3],
                         pad_task_ptr->cv_border_type_);
    }
    (*output_it) = OpencvMatToNDArray(padding_image);
  }
};

class VisionPadOpCPU : VisionBaseOpCPU {
 public:
  VisionPadOpCPU(const Tuple& pad_values, const Any& session_info) : VisionBaseOpCPU(session_info) {
    if (pad_values.size() != 3 || pad_values[0].type_code() != TypeIndex::kRuntimeInteger) {
      MXTHROW << "[BytedVisionPadCPU]: pad_values size must equals to three !";
    }

    for (const RTValue& pad_value : pad_values) {
      pad_values_.emplace_back(pad_value.As<int>());
    }
    task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  }
  ~VisionPadOpCPU() = default;

  RTValue process(const List& images,
                  const List& top_pads,
                  const List& bottom_pads,
                  const List& left_pads,
                  const List& right_pads,
                  const unicode_view& border_type,
                  bool sync);
  std::vector<PadTaskInputPtr> build_inputs(const List& images,
                                            const List& top_pads,
                                            const List& bottom_pads,
                                            const List& left_pads,
                                            const List& right_pads,
                                            const unicode_view& border_type);

 private:
  std::vector<int> pad_values_;  // CHW
  TaskManagerPtr task_manager_ptr = nullptr;
};

RTValue VisionPadOpCPU::process(const List& images,
                                const List& top_pads,
                                const List& bottom_pads,
                                const List& left_pads,
                                const List& right_pads,
                                const unicode_view& border_type,
                                bool sync) {
  cv::setNumThreads(0);
  int batch_size = images.size();
  std::vector<PadTaskInputPtr> pad_task_inputs =
      build_inputs(images, top_pads, bottom_pads, left_pads, right_pads, border_type);
  List nd_ret;

  std::vector<NDArray> pad_task_outputs =
      task_manager_ptr->Execute<PadTask, PadTaskInputPtr, NDArray>(pad_task_inputs, batch_size);

  nd_ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    nd_ret.append(std::move(pad_task_outputs[i]));
  }
  return nd_ret;
};

std::vector<PadTaskInputPtr> VisionPadOpCPU::build_inputs(const List& images,
                                                          const List& top_pads,
                                                          const List& bottom_pads,
                                                          const List& left_pads,
                                                          const List& right_pads,
                                                          const unicode_view& border_type) {
  int batch_size = images.size();
  MXCHECK((top_pads.size() == batch_size) && (bottom_pads.size() == batch_size) &&
          (left_pads.size() == batch_size) && (right_pads.size() == batch_size))
      << "The params sizes must be match in VisionPadOpCPU. ";

  std::vector<PadTaskInputPtr> pad_task_inputs;
  pad_task_inputs.reserve(batch_size);
  // convert to opencv border_type
  int cv_border_type = UnicodePadTypesToCVBorderTypes(border_type);
  // construct pad_task_input
  for (int i = 0; i < batch_size; i++) {
    auto nd_view = images[i].As<NDArray>();
    pad_task_inputs.emplace_back(
        std::make_shared<PadTaskInput>(nd_view,
                                       std::vector<int>({top_pads[i].As<int>(),
                                                         bottom_pads[i].As<int>(),
                                                         left_pads[i].As<int>(),
                                                         right_pads[i].As<int>()}),
                                       pad_values_,
                                       cv_border_type));
  }
  return pad_task_inputs;
};

class VisionPadGeneralOp : public VisionBaseOp {
 public:
  VisionPadGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionPadOp") {
  }
  ~VisionPadGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionPadOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 2) << "[BytedVisionPadCPU] Constructor Expect 2 arguments but get "
                                 << args.size();
      return std::make_shared<VisionPadOpCPU>(args[0].AsObjectView<Tuple>().data(), args[1]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 7) << "[BytedVisionPadCPU] Expect 7 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionPadOpCPU*>(self)->process(args[0].AsObjectView<List>().data(),
                                                              args[1].AsObjectView<List>().data(),
                                                              args[2].AsObjectView<List>().data(),
                                                              args[3].AsObjectView<List>().data(),
                                                              args[4].AsObjectView<List>().data(),
                                                              args[5].As<unicode_view>(),
                                                              args[6].As<bool>());
    });

MATX_REGISTER_NATIVE_OBJECT(VisionPadGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionPadGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionPadGeneralOp*>(self)->process(args);
    });
}  // namespace ops
}  // namespace byted_matx_vision