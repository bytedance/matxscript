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
#include <math.h>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>
#include "matxscript/runtime/builtins_modules/_randommodule.h"
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/runtime_value.h"
#include "ops/base/vision_base_op.h"
#include "utils/opencv_util.h"
#include "utils/task_manager.h"
#include "vision_base_op_cpu.h"

namespace byted_matx_vision {
namespace ops {
using namespace ::matxscript::runtime;

struct RandomResizedCropTaskInput {
  RandomResizedCropTaskInput(NDArray image,
                             std::vector<int> crop_param,
                             std::vector<int> resize_params,
                             int interp)
      : image_(std::move(image)),
        crop_params_(std::move(crop_param)),
        resize_params_(std::move(resize_params)),
        interp_(interp) {
  }

  NDArray image_;
  std::vector<int> crop_params_;
  std::vector<int> resize_params_;
  int interp_;
};

using RandomResizedCropTaskInputPtr = std::shared_ptr<RandomResizedCropTaskInput>;

class RandomResizedCropTask : public internal::LockBasedRunnable {
 public:
  RandomResizedCropTask(std::vector<RandomResizedCropTaskInputPtr>::iterator first_input,
                        std::vector<NDArray>::iterator first_output,
                        int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<RandomResizedCropTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void RandomResizedCropTask::RunImpl() {
  std::vector<RandomResizedCropTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    RandomResizedCropTaskInputPtr random_resized_crop_task_input_ptr = (*input_it);

    cv::Mat org_image = NDArrayToOpencvMat(random_resized_crop_task_input_ptr->image_);
    cv::Mat crop_image, resize_image;
    MXCHECK(random_resized_crop_task_input_ptr->crop_params_.size() == 4)
        << "crop params sizes must be equals to 4 in CropTaskInput .";
    crop_image = org_image(cv::Rect(random_resized_crop_task_input_ptr->crop_params_[0],
                                    random_resized_crop_task_input_ptr->crop_params_[1],
                                    random_resized_crop_task_input_ptr->crop_params_[2],
                                    random_resized_crop_task_input_ptr->crop_params_[3]));
    cv::resize(crop_image,
               resize_image,
               cv::Size(random_resized_crop_task_input_ptr->resize_params_[1],
                        random_resized_crop_task_input_ptr->resize_params_[0]),
               1.0,
               1.0,
               random_resized_crop_task_input_ptr->interp_);

    (*output_it) = OpencvMatToNDArray(resize_image);
  }
};

class VisionRandomResizedCropOpCPU : public VisionBaseOpCPU {
 public:
  VisionRandomResizedCropOpCPU(const List& scale, const List& ratio, const Any& session_info)
      : VisionBaseOpCPU(session_info) {
    MXCHECK(scale.size() == 2)
        << "[BytedVisionRandomResizedCropOpCPU] scale size must be equals to 2 !";
    MXCHECK(ratio.size() == 2)
        << "[BytedVisionRandomResizedCropOpCPU] ratio size must be equals to 2 !";
    // parse scale
    for (const RTValue& item : scale) {
      scale_.emplace_back(item.As<double>());
    }
    // parse ratio
    for (const RTValue& item : ratio) {
      ratio_.emplace_back(item.As<double>());
    }
    task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  }

  RTValue process(const List& images,
                  const List& desired_height,
                  const List& desired_width,
                  const unicode_view& interpolation);
  std::vector<RandomResizedCropTaskInputPtr> build_inputs(const List& images,
                                                          const List& desired_height,
                                                          const List& desired_width,
                                                          int interp);

 private:
  std::vector<double> scale_;
  std::vector<double> ratio_;
  TaskManagerPtr task_manager_ptr = nullptr;
};

RTValue VisionRandomResizedCropOpCPU::process(const List& images,
                                              const List& desired_height,
                                              const List& desired_width,
                                              const unicode_view& interpolation) {
  cv::setNumThreads(0);
  int batch_size = images.size();
  MXCHECK(batch_size == desired_height.size() && batch_size == desired_width.size())
      << "The params sizes must be match in VisionRandomResizedCropOpCPU. ";
  int interp_flags = UnicodeToOpencvInterp(interpolation);
  if (interp_flags < 0) {
    MXCHECK(false) << "Invalid interp type for VisionRandomResizedCropOpCPU: " << interpolation;
  }
  std::vector<RandomResizedCropTaskInputPtr> random_resized_crop_task_inputs =
      build_inputs(images, desired_height, desired_width, interp_flags);
  List nd_ret;

  std::vector<NDArray> pad_task_outputs =
      task_manager_ptr->Execute<RandomResizedCropTask, RandomResizedCropTaskInputPtr, NDArray>(
          random_resized_crop_task_inputs, batch_size);

  nd_ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    nd_ret.append(std::move(pad_task_outputs[i]));
  }
  return nd_ret;
};

std::vector<RandomResizedCropTaskInputPtr> VisionRandomResizedCropOpCPU::build_inputs(
    const List& images, const List& desired_height, const List& desired_width, int interp) {
  std::vector<RandomResizedCropTaskInputPtr> random_resized_crop_task_input;
  int batch_size = images.size();
  random_resized_crop_task_input.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    bool fallback_to_center_crop = true;
    int x_pointer = 0, y_pointer = 0, new_width = 0, new_height = 0;
    auto nd_view = images[i].As<NDArray>();
    std::vector<int64_t> src_shape = nd_view.Shape();  // HWC
    int org_height = src_shape[0];
    int org_width = src_shape[1];
    std::vector<int> resize_param({desired_height[i].As<int>(), desired_width[i].As<int>()});

    int org_area = org_height * org_width;
    for (int random_idx = 0; random_idx < 10; ++random_idx) {
      int new_area = org_area * kernel_random_uniform(scale_[0], scale_[1]);
      double aspect_ratio = exp(kernel_random_uniform(log(ratio_[0]), log(ratio_[1])));

      new_height = (int)round(sqrt(new_area * aspect_ratio));
      new_width = (int)round(sqrt(new_area / aspect_ratio));

      if ((new_width > 0 and new_width <= org_width) and
          (new_height > 0 and new_height <= org_height)) {
        x_pointer = kernel_random_randint(0, org_width - new_width + 1);
        y_pointer = kernel_random_randint(0, org_height - new_height + 1);
        // 保证 x_pointer + new_width <= org_width and y_pointer + new_height <= org_height
        if ((x_pointer + new_width) > org_width) {
          new_width = org_width - x_pointer;
        }
        if ((y_pointer + new_height) > org_height) {
          new_height = org_height - y_pointer;
        }
        random_resized_crop_task_input.emplace_back(std::make_shared<RandomResizedCropTaskInput>(
            nd_view,
            std::vector<int>({x_pointer, y_pointer, new_width, new_height}),
            resize_param,
            interp));
        fallback_to_center_crop = false;
        break;
      }
    }
    if (!fallback_to_center_crop) {
      continue;
    }
    // fallback to center crop
    double in_ratio = double(org_width) / org_height;
    if (in_ratio < ratio_[0]) {
      new_width = org_width;
      new_height = (int)round(new_width / ratio_[0]);
    } else if (in_ratio > ratio_[1]) {
      new_height = org_height;
      new_width = (int)round(new_height * ratio_[1]);
    } else {
      new_width = org_width;
      new_height = org_height;
    }
    x_pointer = (org_width - new_width) / 2;
    y_pointer = (org_height - new_height) / 2;
    // 保证 x_pointer + new_width <= org_width and y_pointer + new_height <= org_height
    if ((x_pointer + new_width) > org_width) {
      new_width = org_width - x_pointer;
    }
    if ((y_pointer + new_height) > org_height) {
      new_height = org_height - x_pointer;
    }
    std::vector<int> crop_param({x_pointer, y_pointer, new_width, new_height});
    random_resized_crop_task_input.emplace_back(std::make_shared<RandomResizedCropTaskInput>(
        nd_view,
        std::vector<int>({x_pointer, y_pointer, new_width, new_height}),
        std::move(resize_param),
        interp));
  }
  return random_resized_crop_task_input;
};

class VisionRandomResizedCropGeneralOp : public VisionBaseOp {
 public:
  VisionRandomResizedCropGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionRandomResizedCropOp") {
  }
  ~VisionRandomResizedCropGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionRandomResizedCropGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionRandomResizedCropGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionRandomResizedCropGeneralOp*>(self)->process(args);
    });

MATX_REGISTER_NATIVE_OBJECT(VisionRandomResizedCropOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 3)
          << "[VisionRandomResizedCropOpCPU] Constructor Expect 3 arguments but get "
          << args.size();
      return std::make_shared<VisionRandomResizedCropOpCPU>(
          args[0].AsObjectView<List>().data(), args[1].AsObjectView<List>().data(), args[2]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 5) << "[VisionRandomResizedCropOpCPU] Expect 5 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionRandomResizedCropOpCPU*>(self)->process(
          args[0].AsObjectView<List>().data(),
          args[1].AsObjectView<List>().data(),
          args[2].AsObjectView<List>().data(),
          args[3].As<unicode_view>());
    });

}  // namespace ops
}  // namespace byted_matx_vision