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

struct NormalizeTaskInput {
  NormalizeTaskInput(NDArray image, std::vector<float>& alpha, std::vector<float>& beta, int dtype)
      : image_(std::move(image)), alpha_(alpha), beta_(beta), dtype_(dtype) {
  }

  NDArray image_;
  std::vector<float> alpha_;
  std::vector<float> beta_;
  int dtype_;
};

using NormalizeTaskInputPtr = std::shared_ptr<NormalizeTaskInput>;

class NormalizeTask : public internal::LockBasedRunnable {
 public:
  NormalizeTask(std::vector<NormalizeTaskInputPtr>::iterator first_input,
                std::vector<NDArray>::iterator first_output,
                int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<NormalizeTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void NormalizeTask::RunImpl() {
  std::vector<NormalizeTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    NormalizeTaskInputPtr normalize_task_input_ptr = (*input_it);

    cv::Mat mat_src = NDArrayToOpencvMat(normalize_task_input_ptr->image_);
    std::vector<float> alpha_ = normalize_task_input_ptr->alpha_;
    std::vector<float> beta_ = normalize_task_input_ptr->beta_;
    int dtype_ = normalize_task_input_ptr->dtype_;
    int channel_size_ = alpha_.size();
    cv::Mat mat_dst;
    const int channels = mat_src.channels();
    if (channels != channel_size_) {
      MXTHROW << "The channel of input should be equal to the channel of mean and std";
    }
    std::vector<cv::Mat> bgrChannels(channels);
    cv::split(mat_src, bgrChannels);
    for (auto c = 0; c < channels; c++) {
      bgrChannels[c].convertTo(bgrChannels[c], CV_MAKETYPE(dtype_, 1), alpha_[c], beta_[c]);
    }
    cv::merge(bgrChannels, mat_dst);

    (*output_it) = OpencvMatToNDArray(mat_dst);
  }
};

class VisionNormalizeOpCPU : VisionBaseOpCPU {
 public:
  VisionNormalizeOpCPU(const Any& session_info,
                       const List& mean,
                       const List& std,
                       float global_shift,
                       float global_scale,
                       const unicode_view& out_fmt);
  ~VisionNormalizeOpCPU() = default;

  RTValue process(const List& image_ndarray);

 private:
  float global_scale_;
  float global_shift_;
  float epsilon_ = 0.0;
  int channel_size_;
  std::vector<float> alpha_;
  std::vector<float> beta_;
  int cv_depth_type;
  TaskManagerPtr task_manager_ptr = nullptr;
};

VisionNormalizeOpCPU::VisionNormalizeOpCPU(const Any& session_info,
                                           const List& list_mean,
                                           const List& list_std,
                                           float global_shift,
                                           float global_scale,
                                           const unicode_view& rtype)
    : VisionBaseOpCPU(session_info) {
  if (list_mean.size() != list_std.size()) {
    MXTHROW << "The size of mean and std should be equal";
  }
  task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  global_scale_ = global_scale;
  global_shift_ = global_shift;
  channel_size_ = list_mean.size();
  for (int i = 0; i < channel_size_; i++) {
    float cur_std = list_std[i].As<float>();
    float cur_mean = list_mean[i].As<float>();
    float s = 1.0f / sqrt(cur_std * cur_std + epsilon_);
    float cur_alpha = s * global_scale_;
    alpha_.push_back(cur_alpha);
    float cur_beta = global_shift_ + (0.0 - cur_mean) * s * global_scale_;
    beta_.push_back(cur_beta);
  }
  cv_depth_type = UnicodeTypeToOpencvDepth(rtype);
}

RTValue VisionNormalizeOpCPU::process(const List& input) {
  int batch_size = input.size();
  // construct norm_task_input
  std::vector<NormalizeTaskInputPtr> norm_task_inputs;
  norm_task_inputs.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    auto nd_view = input[i].As<NDArray>();
    norm_task_inputs.emplace_back(
        std::make_shared<NormalizeTaskInput>(nd_view, alpha_, beta_, cv_depth_type));
  }
  List ret;
  std::vector<NDArray> normalize_task_outputs =
      task_manager_ptr->Execute<NormalizeTask, NormalizeTaskInputPtr, NDArray>(norm_task_inputs,
                                                                               batch_size);
  ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ret.append(std::move(normalize_task_outputs[i]));
  }
  return ret;
}

class VisionNormalizeGeneralOp : public VisionBaseOp {
 public:
  VisionNormalizeGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionNormalizeOp") {
  }
  ~VisionNormalizeGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionNormalizeOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 6) << "[VisionNormalizeOpCPU] Expect 6 arguments but get "
                                << args.size();
      return std::make_shared<VisionNormalizeOpCPU>(args[5],
                                                    args[0].AsObjectView<List>().data(),
                                                    args[1].AsObjectView<List>().data(),
                                                    args[2].As<float>(),
                                                    args[3].As<float>(),
                                                    args[4].As<unicode_view>());
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 2)
                            << "[VisionNormalizeOpCPU][func: process] Expect 2 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionNormalizeOpCPU*>(self)->process(
                            args[0].AsObjectView<List>().data());
                      });

MATX_REGISTER_NATIVE_OBJECT(VisionNormalizeGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionNormalizeGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionNormalizeGeneralOp*>(self)->process(args);
    });
}  // namespace ops
}  // namespace byted_matx_vision
