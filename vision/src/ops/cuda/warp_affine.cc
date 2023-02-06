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

#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/py_args.h>
#include <mutex>
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/container/tuple_ref.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/runtime_value.h"
#include "utils/cuda/cuda_op_helper.h"
#include "utils/cuda/cuda_type_helper.h"
#include "utils/opencv_util.h"
#include "utils/pad_types.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

#define PI 3.14159265

using namespace matxscript::runtime;

class VisionWarpAffineOpGPU : public VisionBaseImageOpGPU<cuda_op::WarpAffineVarShape> {
 public:
  VisionWarpAffineOpGPU(const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::WarpAffineVarShape>(session_info){};
  RTValue base(const List& images,
               const int* dsizes,
               const float* trans_matrix,
               const unicode_view& borderType,
               const Tuple& borderValue,
               const unicode_view& interpolation,
               int sync);

  RTValue affine_process(const List& images,
                         const List& dsizes,
                         const List& affine_matrix,
                         const unicode_view& borderType,
                         const Tuple& borderValue,
                         const unicode_view& interpolation,
                         int sync);

  RTValue rotate_process(const List& images,
                         const List& dsizes,
                         const List& center,
                         const List& angle,
                         const List& scale,
                         const List& expand,
                         const unicode_view& borderType,
                         const Tuple& borderValue,
                         const unicode_view& interpolation,
                         int sync);
};

RTValue VisionWarpAffineOpGPU::base(const List& arg_images,
                                    const int* dsizes,
                                    const float* trans_matrix,
                                    const unicode_view& borderType,
                                    const Tuple& borderValue,
                                    const unicode_view& interpolation,
                                    int sync) {
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());
  int interp_flags = UnicodeToOpencvInterp(interpolation);
  int cv_border_type = UnicodePadTypesToCVBorderTypes(borderType);

  if (borderValue.size() != 1 && borderValue.size() != 3) {
    MXCHECK(false) << "The shape of border value should either be 1 or be 3.";
  }

  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  auto finish_event_mutex = std::make_shared<std::mutex>();
  auto not_finish = std::make_shared<bool>(true);

  int batch_size = images.size();
  size_t op_buffer_size = op_->calBufferSize(batch_size);
  std::shared_ptr<void> cpu_buffer_ptr(malloc(op_buffer_size), free);
  void* gpu_workspace = cuda_api_->Alloc(ctx_, op_buffer_size);

  const void* input_ptr[batch_size];
  void* output_ptr[batch_size];
  cv::Size cv_dsize[batch_size];
  cuda_op::DataShape input_shape[batch_size];
  int channel = 0;
  DataType nd_data_type;
  List res;

  int i = 0;
  for (const RTValue& nd_elem : images) {
    auto view_elem = nd_elem.AsObjectView<NDArray>();
    const NDArray& elem = view_elem.data();
    input_ptr[i] = (void*)(elem->data);
    std::vector<int64_t> src_shape = elem.Shape();

    if (i == 0) {
      channel = src_shape[2];
      nd_data_type = elem.DataType();
    } else {
      if (channel != src_shape[2]) {
        MXCHECK(false) << "Invalid input. The output shape should be equal";
      }
      if (nd_data_type != elem.DataType()) {
        MXCHECK(false) << "The inputs must have same data type";
      }
    }
    input_shape[i].N = 1;
    input_shape[i].C = channel;
    input_shape[i].H = src_shape[0];
    input_shape[i].W = src_shape[1];
    std::vector<int64_t> out_shape;  // HWC
    int cur_dsize_w = dsizes[i * 2 + 1];
    int cur_dsize_h = dsizes[i * 2];
    out_shape = {cur_dsize_h, cur_dsize_w, channel};
    size_t output_buffer_size = CalculateOutputBufferSize(out_shape, nd_data_type);

    NDArray dst_arr = MakeNDArrayWithWorkSpace(
        ctx_,
        cuda_api_,
        output_buffer_size,
        out_shape,
        nd_data_type,
        [finish_event,
         finish_event_mutex,
         not_finish,
         elem,
         cpu_buffer_ptr,
         local_device_api = this->cuda_api_,
         local_device_id = this->device_id_,
         gpu_workspace]() {
          std::lock_guard<std::mutex> lock(*finish_event_mutex);
          if (*not_finish) {
            DLContext local_ctx;
            local_ctx.device_id = local_device_id;
            local_ctx.device_type = DLDeviceType::kDLCUDA;
            CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
            CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
            local_device_api->Free(local_ctx, gpu_workspace);
            *not_finish = false;
          }
        },
        0,
        nullptr);
    res.push_back(dst_arr);
    output_ptr[i] = (void*)(dst_arr->data);
    cv_dsize[i] = cv::Size(cur_dsize_w, cur_dsize_h);
    i += 1;
  }
  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);

  cv::Scalar scalar_value;
  if (borderValue.size() == 1) {
    scalar_value = cv::Scalar(borderValue[0].As<float>(), 0, 0);
  } else {
    scalar_value = cv::Scalar(
        borderValue[0].As<float>(), borderValue[1].As<float>(), borderValue[2].As<float>());
  }

  op_->infer(input_ptr,
             output_ptr,
             gpu_workspace,
             (void*)cpu_buffer_ptr.get(),
             batch_size,
             op_buffer_size,
             cv_dsize,
             trans_matrix,
             interp_flags,
             cv_border_type,
             scalar_value,
             input_shape,
             cuda_op::kNHWC,
             op_data_type,
             cu_stream);

  // record stop event on the stream
  CHECK_CUDA_CALL(cudaEventRecord(finish_event, cu_stream));
  CUDA_EVENT_SYNC_IF_DEBUG(finish_event);
  CUDA_STREAM_SYNC_IF_DEBUG(cu_stream);
  CUDA_DEVICE_SYNC_IF_DEBUG();
  if (sync != VISION_SYNC_MODE::ASYNC) {
    CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
    if (sync == VISION_SYNC_MODE::SYNC_CPU) {
      return to_cpu(res, getStream());
    } else {
      return res;
    }
  }
  return res;
}

RTValue VisionWarpAffineOpGPU::affine_process(const List& images,
                                              const List& dsizes_list,
                                              const List& affine_matrix,
                                              const unicode_view& borderType,
                                              const Tuple& borderValue,
                                              const unicode_view& interpolation,
                                              int sync) {
  int batch_size = images.size();
  float trans_matrix[6 * batch_size];
  int dsizes[batch_size * 2];
  for (int i = 0; i < batch_size; i++) {
    auto matrix_view = affine_matrix[i].AsObjectView<List>();
    const List& cur_affine_matrix = matrix_view.data();
    for (int row = 0; row < 2; row++) {
      auto row_view = cur_affine_matrix[row].AsObjectView<List>();
      const List& cur_row = row_view.data();
      for (int col = 0; col < 3; col++) {
        float matrix_item = cur_row[col].As<float>();
        trans_matrix[i * 6 + row * 3 + col] = matrix_item;
      }
    }
    auto dsize_view = dsizes_list[i].AsObjectView<List>();
    const List& cur_dsize = dsize_view.data();
    int cur_dsize_w = cur_dsize[1].As<int>();
    int cur_dsize_h = cur_dsize[0].As<int>();
    dsizes[i * 2] = cur_dsize_h;
    dsizes[i * 2 + 1] = cur_dsize_w;
  }
  return base(images, dsizes, trans_matrix, borderType, borderValue, interpolation, sync);
}

RTValue VisionWarpAffineOpGPU::rotate_process(const List& images,
                                              const List& dsizes_list,
                                              const List& center,
                                              const List& angle,
                                              const List& scale,
                                              const List& expand,
                                              const unicode_view& borderType,
                                              const Tuple& borderValue,
                                              const unicode_view& interpolation,
                                              int sync) {
  int batch_size = images.size();
  float trans_matrix[6 * batch_size];
  int dsizes[batch_size * 2];
  int matrix_idx = 0;
  for (int i = 0; i < batch_size; i++) {
    double tmp_angle = angle[i].As<double>();
    double tmp_scale = scale[i].As<double>();
    auto center_view = center[i].AsObjectView<List>();
    const List& tmp_center_list = center_view.data();  // [cty, ctx]
    cv::Point tmp_center(tmp_center_list[1].As<float>(), tmp_center_list[0].As<float>());
    cv::Mat rotMat = cv::getRotationMatrix2D(tmp_center, tmp_angle, tmp_scale);
    for (int row = 0; row < 2; row++) {
      for (int col = 0; col < 3; col++) {
        trans_matrix[matrix_idx] = (float)(rotMat.at<double>(row, col));
        matrix_idx += 1;
      }
    }
    auto dsize_view = dsizes_list[i].AsObjectView<List>();
    const List& cur_dsize = dsize_view.data();
    int cur_dsize_w = cur_dsize[1].As<int>();
    int cur_dsize_h = cur_dsize[0].As<int>();
    dsizes[i * 2] = cur_dsize_h;
    dsizes[i * 2 + 1] = cur_dsize_w;
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
      dsizes[i * 2] = (int)(new_h * tmp_scale);
      dsizes[i * 2 + 1] = (int)(new_w * tmp_scale);

      double old_ch = (old_h - 1) / 2.0;
      double old_cw = (old_w - 1) / 2.0;
      double new_ch = (new_h - 1) / 2.0;
      double new_cw = (new_w - 1) / 2.0;
      trans_matrix[6 * i + 2] = (-cos_alpha * old_cw - sin_alpha * old_ch + new_cw) * tmp_scale;
      trans_matrix[6 * i + 5] = (sin_alpha * old_cw - cos_alpha * old_ch + new_ch) * tmp_scale;
    }
    i += 1;
  }
  return base(images, dsizes, trans_matrix, borderType, borderValue, interpolation, sync);
}

MATX_REGISTER_NATIVE_OBJECT(VisionWarpAffineOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionWarpAffineOpGPU] Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionWarpAffineOpGPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 7)
                            << "[VisionWarpAffineOpGPU][func: process] Expect 7 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionWarpAffineOpGPU*>(self)->affine_process(
                            args[0].AsObjectView<List>().data(),
                            args[1].AsObjectView<List>().data(),
                            args[2].AsObjectView<List>().data(),
                            args[3].As<unicode_view>(),
                            args[4].AsObjectView<Tuple>().data(),
                            args[5].As<unicode_view>(),
                            args[6].As<int>());
                      });

using VisionRotateOpGPU = VisionWarpAffineOpGPU;
MATX_REGISTER_NATIVE_OBJECT(VisionRotateOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionRotateOpGPU] Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionRotateOpGPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 10)
                            << "[VisionRotateOpGPU][func: process] Expect 10 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionWarpAffineOpGPU*>(self)->rotate_process(
                            args[0].AsObjectView<List>().data(),
                            args[1].AsObjectView<List>().data(),
                            args[2].AsObjectView<List>().data(),
                            args[3].AsObjectView<List>().data(),
                            args[4].AsObjectView<List>().data(),
                            args[5].AsObjectView<List>().data(),
                            args[6].As<unicode_view>(),
                            args[7].AsObjectView<Tuple>().data(),
                            args[8].As<unicode_view>(),
                            args[9].As<int>());
                      });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision