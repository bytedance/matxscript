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

#include <opencv2/opencv.hpp>
#include <opencv_cuda.h>
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/runtime_value.h"
#include "ops/base/vision_base_op.h"
#include "utils/cuda/cuda_op_helper.h"
#include "utils/cuda/cuda_type_helper.h"
#include "utils/opencv_util.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace ::matxscript::runtime;

class VisionRandomResizedCropOpGPU : public VisionBaseOpGPU {
 public:
  VisionRandomResizedCropOpGPU(const List& scale, const List& ratio, const Any& session_info)
      : VisionBaseOpGPU(session_info) {
    MXCHECK(scale.size() == 2)
        << "[BytedVisionRandomResizedCropOpGPU] scale size must be equals to 2 !";
    MXCHECK(ratio.size() == 2)
        << "[BytedVisionRandomResizedCropOpGPU] ratio size must be equals to 2 !";

    max_input_shape_.N = 1;
    max_input_shape_.C = 3;
    max_input_shape_.H = 1024;
    max_input_shape_.W = 1024;

    op_ = std::make_shared<cuda_op::RandomResizedCropVarShape>(max_input_shape_,
                                                               max_output_shape_,
                                                               scale[0].As<double>(),
                                                               scale[1].As<double>(),
                                                               ratio[0].As<double>(),
                                                               ratio[1].As<double>());
  }

  ~VisionRandomResizedCropOpGPU() = default;
  RTValue process(const List& images,
                  const List& desired_height,
                  const List& desired_width,
                  const unicode_view& interpolation,
                  int sync);

 protected:
  std::shared_ptr<cuda_op::RandomResizedCropVarShape> op_ = nullptr;
  cuda_op::DataShape max_input_shape_, max_output_shape_;
};

RTValue VisionRandomResizedCropOpGPU::process(const List& arg_images,
                                              const List& desired_height,
                                              const List& desired_width,
                                              const unicode_view& interpolation,
                                              int sync) {
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());
  int batch_size = images.size();
  int cv_interpolation_flags = UnicodeToOpencvInterp(interpolation);
  if (cv_interpolation_flags < 0) {
    MXCHECK(false) << "Invalid interp type for VisionRandomResizedCropOpCPU: " << interpolation;
  }

  // calculate buffer
  size_t op_buffer_size = op_->calBufferSize(batch_size);
  std::shared_ptr<void> cpu_buffer_ptr(malloc(op_buffer_size), free);
  void* gpu_workspace = cuda_api_->Alloc(ctx_, op_buffer_size);

  // create event
  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  auto finish_event_mutex = std::make_shared<std::mutex>();
  auto not_finish = std::make_shared<bool>(true);

  // construct outputs and inputs
  const void* input_ptr[batch_size];
  void* output_ptr[batch_size];

  cuda_op::DataShape input_shape[batch_size];
  int channel = 0;
  DataType nd_data_type;
  List res;
  cv::Size dsize[batch_size];
  double fx[batch_size];
  double fy[batch_size];
  cv::Rect roi[batch_size];

  int i = 0;
  for (const RTValue& elem_nd : images) {
    auto view_elem = elem_nd.AsObjectView<NDArray>();
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

    int tmp_x, tmp_y, tmp_width, tmp_height;
    op_->getCropParams(input_shape[i], &tmp_y, &tmp_x, &tmp_height, &tmp_width);
    roi[i].x = tmp_x;
    roi[i].y = tmp_y;
    roi[i].width = tmp_width;
    roi[i].height = tmp_height;

    std::vector<int64_t> out_shape = {
        desired_height[i].As<int>(), desired_width[i].As<int>(), channel};
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
    dsize[i] = cv::Size(desired_width[i].As<int>(), desired_height[i].As<int>());
    fx[i] = 1.0;
    fy[i] = 1.0;
    output_ptr[i] = (void*)(dst_arr->data);
    i += 1;
  }

  // call kernel
  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);

  op_->infer(input_ptr,
             output_ptr,
             gpu_workspace,
             (void*)cpu_buffer_ptr.get(),
             batch_size,
             op_buffer_size,
             dsize,
             fx,
             fy,
             roi,
             cv_interpolation_flags,
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
};

MATX_REGISTER_NATIVE_OBJECT(VisionRandomResizedCropOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 3)
          << "[VisionRandomResizedCropOpGPU] Constructor Expect 3 arguments but get "
          << args.size();
      return std::make_shared<VisionRandomResizedCropOpGPU>(
          args[0].AsObjectView<List>().data(), args[1].AsObjectView<List>().data(), args[2]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 5) << "[VisionRandomResizedCropOpGPU] Expect 5 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionRandomResizedCropOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(),
          args[1].AsObjectView<List>().data(),
          args[2].AsObjectView<List>().data(),
          args[3].As<unicode_view>(),
          args[4].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision