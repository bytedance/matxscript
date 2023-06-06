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

#include <cv_cuda.h>
#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/py_args.h>
#include <mutex>
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/runtime_value.h"
#include "utils/cuda/cuda_op_helper.h"
#include "utils/cuda/cuda_type_helper.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

class VisionCropOpGPU : public VisionBaseImageOpGPU<cuda_op::CropVarShape> {
 public:
  VisionCropOpGPU(const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::CropVarShape>(session_info) {
  }
  RTValue process(const List& arg_images,
                  const List& x,
                  const List& y,
                  const List& widths,
                  const List& heights,
                  int sync);
};

RTValue VisionCropOpGPU::process(const List& arg_images,
                                 const List& x,
                                 const List& y,
                                 const List& widths,
                                 const List& heights,
                                 int sync) {
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());
  int batch_size = images.size();

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
  cv::Rect roi[batch_size];
  int channel = 0;
  DataType nd_data_type;
  List res;

  int i = 0;
  for (const RTValue& elem : images) {
    auto view_elem = elem.AsObjectView<NDArray>();
    const NDArray& elem_data = view_elem.data();
    input_ptr[i] = (void*)(elem_data->data);
    std::vector<int64_t> src_shape = elem_data.Shape();

    if (i == 0) {
      channel = src_shape[2];
      nd_data_type = elem_data.DataType();
    } else {
      if (channel != src_shape[2]) {
        MXCHECK(false) << "Invalid input. The output shape should be equal";
      }
      if (nd_data_type != elem_data.DataType()) {
        MXCHECK(false) << "The inputs must have same data type";
      }
    }

    input_shape[i].N = 1;
    input_shape[i].C = channel;
    input_shape[i].H = src_shape[0];
    input_shape[i].W = src_shape[1];
    int tmp_x = x[i].As<int>();
    int tmp_y = y[i].As<int>();
    int tmp_width = widths[i].As<int>();
    int tmp_height = heights[i].As<int>();
    MXCHECK(0 <= tmp_x && 0 <= tmp_width && tmp_x + tmp_width <= src_shape[1])
        << "X + Width should be less than or equal to image width, but get : " << tmp_x + tmp_width
        << ", origin image width: " << src_shape[1];

    MXCHECK(0 <= tmp_y && 0 <= tmp_height && tmp_y + tmp_height <= src_shape[0])
        << "Y + Height should be less than or equal to image height, but get : "
        << tmp_y + tmp_height << ", origin image height: " << src_shape[0];

    roi[i].x = tmp_x;
    roi[i].y = tmp_y;
    roi[i].width = tmp_width;
    roi[i].height = tmp_height;
    std::vector<int64_t> out_shape = {tmp_height, tmp_width, channel};
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
         elem_data,
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
             roi,
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

MATX_REGISTER_NATIVE_OBJECT(VisionCropOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[BytedVisionCropOpGPU] Constructor Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionCropOpGPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 6) << "[BytedVisionCropOpGPU] Expect 6 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionCropOpGPU*>(self)->process(args[0].AsObjectView<List>().data(),
                                                               args[1].AsObjectView<List>().data(),
                                                               args[2].AsObjectView<List>().data(),
                                                               args[3].AsObjectView<List>().data(),
                                                               args[4].AsObjectView<List>().data(),
                                                               args[5].As<int>());
    });
}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision