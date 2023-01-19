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

#define PI 3.14159265

struct WarpAffineTaskInput {
  WarpAffineTaskInput(NDArray image,
                      int height,
                      int width,
                      cv::Mat matrix,
                      int border_type,
                      cv::Scalar border_value,
                      int interp)
      : image_(std::move(image)),
        height_(height),
        width_(width),
        matrix_(std::move(matrix)),
        border_type_(border_type),
        border_value_(std::move(border_value)),
        interp_(interp) {
  }

  NDArray image_;
  int height_;
  int width_;
  cv::Mat matrix_;
  int border_type_;
  cv::Scalar border_value_;
  int interp_;
};

using WarpAffineTaskInputPtr = std::shared_ptr<WarpAffineTaskInput>;

class WarpAffineTask : public internal::LockBasedRunnable {
 public:
  WarpAffineTask(std::vector<WarpAffineTaskInputPtr>::iterator first_input,
                 std::vector<NDArray>::iterator first_output,
                 int len)
      : input_it_(first_input), output_it_(first_output), len_(len) {
  }

 protected:
  void RunImpl() override;

 private:
  std::vector<WarpAffineTaskInputPtr>::iterator input_it_;
  std::vector<NDArray>::iterator output_it_;
  int len_;
};

void WarpAffineTask::RunImpl() {
  std::vector<WarpAffineTaskInputPtr>::iterator input_it = input_it_;
  std::vector<NDArray>::iterator output_it = output_it_;
  for (int i = 0; i < len_; ++i, ++input_it, ++output_it) {
    WarpAffineTaskInputPtr warp_affine_task_input_ptr = (*input_it);

    cv::Mat mat_src = NDArrayToOpencvMat(warp_affine_task_input_ptr->image_);
    cv::Mat mat_dst;
    cv::warpAffine(
        mat_src,
        mat_dst,
        warp_affine_task_input_ptr->matrix_,
        cv::Size(warp_affine_task_input_ptr->width_, warp_affine_task_input_ptr->height_),
        warp_affine_task_input_ptr->interp_,
        warp_affine_task_input_ptr->border_type_,
        warp_affine_task_input_ptr->border_value_);
    (*output_it) = OpencvMatToNDArray(mat_dst);
  }
};

class VisionWarpAffineOpCPU : public VisionBaseOpCPU {
 public:
  VisionWarpAffineOpCPU(const Any& session_info) : VisionBaseOpCPU(session_info) {
    task_manager_ptr = std::make_shared<TaskManager>(thread_pool_);
  };
  RTValue base(const List& images,
               const int* width,
               const int* height,
               const std::vector<cv::Mat> trans_matrix,
               const unicode_view& borderType,
               const Tuple& borderValue,
               const unicode_view& interpolation);

  RTValue affine_process(const List& images,
                         const List& dsizes,
                         const List& trans_matrix,
                         const unicode_view& borderType,
                         const Tuple& borderValue,
                         const unicode_view& interpolation);

  RTValue rotate_process(const List& images,
                         const List& dsizes,
                         const List& center,
                         const List& angle,
                         const List& scale,
                         const List& expand,
                         const unicode_view& borderType,
                         const Tuple& borderValue,
                         const unicode_view& interpolation);

 private:
  TaskManagerPtr task_manager_ptr = nullptr;
};

RTValue VisionWarpAffineOpCPU::base(const List& images,
                                    const int* width,
                                    const int* height,
                                    const std::vector<cv::Mat> trans_matrix,
                                    const unicode_view& borderType,
                                    const Tuple& borderValue,
                                    const unicode_view& interpolation) {
  int batch_size = images.size();
  int interp_flags = UnicodeToOpencvInterp(interpolation);
  int cv_border_type = UnicodePadTypesToCVBorderTypes(borderType);
  if (borderValue.size() != 1 && borderValue.size() != 3) {
    MXCHECK(false) << "The shape of border value should either be 1 or be 3.";
  }
  cv::Scalar border_scalar_value;
  if (borderValue.size() == 1) {
    border_scalar_value = cv::Scalar(borderValue[0].As<float>(), 0, 0);
  } else {
    border_scalar_value = cv::Scalar(
        borderValue[0].As<float>(), borderValue[1].As<float>(), borderValue[2].As<float>());
  }
  // construct warp_affine_task_input
  std::vector<WarpAffineTaskInputPtr> warp_affine_task_inputs;
  warp_affine_task_inputs.reserve(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    auto nd_view = images[i].As<NDArray>();
    warp_affine_task_inputs.emplace_back(std::make_shared<WarpAffineTaskInput>(nd_view,
                                                                               height[i],
                                                                               width[i],
                                                                               trans_matrix[i],
                                                                               cv_border_type,
                                                                               border_scalar_value,
                                                                               interp_flags));
  }

  List ret;

  std::vector<NDArray> warp_affine_task_outputs =
      task_manager_ptr->Execute<WarpAffineTask, WarpAffineTaskInputPtr, NDArray>(
          warp_affine_task_inputs, batch_size);
  ret.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    ret.append(std::move(warp_affine_task_outputs[i]));
  }
  return ret;
}

RTValue VisionWarpAffineOpCPU::affine_process(const List& images,
                                              const List& dsizes_list,
                                              const List& trans_matrix,
                                              const unicode_view& borderType,
                                              const Tuple& borderValue,
                                              const unicode_view& interpolation) {
  int batch_size = images.size();
  std::vector<cv::Mat> mat_matrix;
  mat_matrix.reserve(batch_size);
  float matrix_array[6 * batch_size];
  int width[batch_size], height[batch_size];
  for (int i = 0; i < batch_size; i++) {
    float* cur_matrix_array = matrix_array + i * 6;
    auto matrix_view = trans_matrix[i].AsObjectView<List>();
    const List& cur_matrix = matrix_view.data();
    for (int row = 0; row < 2; row++) {
      auto row_view = cur_matrix[row].AsObjectView<List>();
      const List& cur_row = row_view.data();
      for (int col = 0; col < 3; col++) {
        float matrix_item = cur_row[col].As<float>();
        cur_matrix_array[row * 3 + col] = matrix_item;
      }
    }
    cv::Mat cur_mat_matrix(2, 3, CV_32FC1, (void*)cur_matrix_array);
    mat_matrix.emplace_back(std::move(cur_mat_matrix));
    auto dsize_view = dsizes_list[i].AsObjectView<List>();
    const List& cur_dsize = dsize_view.data();
    width[i] = cur_dsize[1].As<int>();
    height[i] = cur_dsize[0].As<int>();
  }
  return base(images, width, height, mat_matrix, borderType, borderValue, interpolation);
}

RTValue VisionWarpAffineOpCPU::rotate_process(const List& images,
                                              const List& dsizes_list,
                                              const List& center,
                                              const List& angle,
                                              const List& scale,
                                              const List& expand,
                                              const unicode_view& borderType,
                                              const Tuple& borderValue,
                                              const unicode_view& interpolation) {
  int batch_size = images.size();
  std::vector<cv::Mat> mat_matrix;
  mat_matrix.reserve(batch_size);
  int width[batch_size], height[batch_size];
  int matrix_idx = 0;
  for (int i = 0; i < batch_size; i++) {
    double tmp_angle = angle[i].As<double>();
    double tmp_scale = scale[i].As<double>();
    auto center_view = center[i].AsObjectView<List>();
    const List& tmp_center_list = center_view.data();  // [cty, ctx]
    cv::Point tmp_center(tmp_center_list[1].As<float>(), tmp_center_list[0].As<float>());
    cv::Mat rot_mat = cv::getRotationMatrix2D(tmp_center, tmp_angle, tmp_scale);
    mat_matrix.emplace_back(std::move(rot_mat));
    auto dsize_view = dsizes_list[i].AsObjectView<List>();
    const List& cur_dsize = dsize_view.data();
    width[i] = cur_dsize[1].As<int>();
    height[i] = cur_dsize[0].As<int>();
  }

  int i = 0;
  for (const RTValue& nd_elem : images) {
    auto view_elem = nd_elem.AsObjectView<NDArray>();
    const NDArray& elem = view_elem.data();
    bool to_expand = expand[i].As<bool>();
    if (to_expand) {
      std::vector<int64_t> src_shape = elem.Shape();
      double tmp_angle = angle[i].As<double>();
      double sin_alpha = sin(tmp_angle * PI / 180);
      double cos_alpha = cos(tmp_angle * PI / 180);
      double tmp_scale = scale[i].As<double>();
      int old_h = src_shape[0];
      int old_w = src_shape[1];
      int new_h, new_w;
      new_h = old_h * std::abs(cos_alpha) + old_w * std::abs(sin_alpha);
      new_w = old_h * std::abs(sin_alpha) + old_w * std::abs(cos_alpha);
      height[i] = (int)(new_h * tmp_scale);
      width[i] = (int)(new_w * tmp_scale);

      double old_ch = (old_h - 1) / 2.0;
      double old_cw = (old_w - 1) / 2.0;
      double new_ch = (new_h - 1) / 2.0;
      double new_cw = (new_w - 1) / 2.0;
      mat_matrix[i].at<double>(0, 2) =
          (-cos_alpha * old_cw - sin_alpha * old_ch + new_cw) * tmp_scale;
      mat_matrix[i].at<double>(1, 2) =
          (sin_alpha * old_cw - cos_alpha * old_ch + new_ch) * tmp_scale;
    }
    i += 1;
  }
  return base(images, width, height, mat_matrix, borderType, borderValue, interpolation);
}

class VisionWarpAffineGeneralOp : public VisionBaseOp {
 public:
  VisionWarpAffineGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionWarpAffineOp") {
  }
  ~VisionWarpAffineGeneralOp() = default;
};

class VisionRotateGeneralOp : public VisionBaseOp {
 public:
  VisionRotateGeneralOp(PyArgs args) : VisionBaseOp(args, "VisionRotateOp") {
  }
  ~VisionRotateGeneralOp() = default;
};

MATX_REGISTER_NATIVE_OBJECT(VisionWarpAffineOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionWarpAffineOpCPU] Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionWarpAffineOpCPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 7)
                            << "[VisionWarpAffineOpCPU][func: process] Expect 7 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionWarpAffineOpCPU*>(self)->affine_process(
                            args[0].AsObjectView<List>().data(),
                            args[1].AsObjectView<List>().data(),
                            args[2].AsObjectView<List>().data(),
                            args[3].As<unicode_view>(),
                            args[4].AsObjectView<Tuple>().data(),
                            args[5].As<unicode_view>());
                      });

using VisionRotateOpCPU = VisionWarpAffineOpCPU;
MATX_REGISTER_NATIVE_OBJECT(VisionRotateOpCPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionRotateOpCPU] Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionRotateOpCPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 10)
                            << "[VisionRotateOpCPU][func: process] Expect 10 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionWarpAffineOpCPU*>(self)->rotate_process(
                            args[0].AsObjectView<List>().data(),
                            args[1].AsObjectView<List>().data(),
                            args[2].AsObjectView<List>().data(),
                            args[3].AsObjectView<List>().data(),
                            args[4].AsObjectView<List>().data(),
                            args[5].AsObjectView<List>().data(),
                            args[6].As<unicode_view>(),
                            args[7].AsObjectView<Tuple>().data(),
                            args[8].As<unicode_view>());
                      });

MATX_REGISTER_NATIVE_OBJECT(VisionWarpAffineGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionWarpAffineGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionWarpAffineGeneralOp*>(self)->process(args);
    });

MATX_REGISTER_NATIVE_OBJECT(VisionRotateGeneralOp)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      return std::make_shared<VisionRotateGeneralOp>(args);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      return reinterpret_cast<VisionRotateGeneralOp*>(self)->process(args);
    });

}  // namespace ops
}  // namespace byted_matx_vision
