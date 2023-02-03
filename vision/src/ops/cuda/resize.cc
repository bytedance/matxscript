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
#include "matxscript/runtime/container/unicode.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/runtime_value.h"
#include "utils/cuda/cuda_op_helper.h"
#include "utils/cuda/cuda_type_helper.h"
#include "utils/opencv_util.h"
#include "utils/type_helper.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

class VisionResizeOpGPU : public VisionBaseOpGPU {
 public:
  VisionResizeOpGPU(const Any& session_info);
  RTValue process(const List& images,
                  const List& height,
                  const List& width,
                  const unicode_view& interp,
                  int sync);

 private:
  size_t calPillowResizeBufferSize(const List& images,
                                   const List& desired_height,
                                   const List& desired_width);
  int max_batch_size_ = 1;
  std::shared_ptr<cuda_op::ResizeVarShape> resize_op_;
  std::shared_ptr<cuda_op::PillowResizeVarShape> pillow_resize_op_;
  cuda_op::DataShape max_input_shape_, max_output_shape_;
};

VisionResizeOpGPU::VisionResizeOpGPU(const Any& session_info) : VisionBaseOpGPU(session_info) {
  max_input_shape_.N = max_batch_size_;
  max_input_shape_.C = 3;
  max_input_shape_.H = 1024;
  max_input_shape_.W = 1024;
  resize_op_ = std::make_shared<cuda_op::ResizeVarShape>(max_input_shape_, max_output_shape_);
  pillow_resize_op_ =
      std::make_shared<cuda_op::PillowResizeVarShape>(max_input_shape_, max_output_shape_);
}

size_t VisionResizeOpGPU::calPillowResizeBufferSize(const List& images,
                                                    const List& desired_height,
                                                    const List& desired_width) {
  int batch_size = images.size();
  int max_input_width = 0, max_input_height = 0;
  int max_output_width = 0, max_output_height = 0;
  int i = 0;
  for (const RTValue& nd_elem : images) {
    auto view_elem = nd_elem.AsObjectView<NDArray>();
    const NDArray& elem = view_elem.data();
    std::vector<int64_t> src_shape = elem.Shape();
    if (max_input_width < src_shape[1])
      max_input_width = src_shape[1];
    if (max_input_height < src_shape[0])
      max_input_height = src_shape[0];
    int tmp_output_width = desired_width[i].As<int>();
    int tmp_output_height = desired_height[i].As<int>();
    if (max_output_width < tmp_output_width)
      max_output_width = tmp_output_width;
    if (max_output_height < tmp_output_height)
      max_output_height = tmp_output_height;
    i += 1;
  }
  max_input_shape_.H = max_input_height;
  max_input_shape_.W = max_input_width;
  max_output_shape_.H = max_output_height;
  max_output_shape_.W = max_output_width;

  size_t op_buffer_size =
      pillow_resize_op_->calBufferSize(max_input_shape_, max_output_shape_, batch_size);
  return op_buffer_size;
}

RTValue VisionResizeOpGPU::process(const List& arg_images,
                                   const List& desired_height,
                                   const List& desired_width,
                                   const unicode_view& interpolation,
                                   int sync) {
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());
  int interp_flags = UnicodeToOpencvInterp(interpolation);
  bool use_pillow_resize = interp_flags < 0;

  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  auto finish_event_mutex = std::make_shared<std::mutex>();
  auto not_finish = std::make_shared<bool>(true);

  int batch_size = images.size();
  MXCHECK_EQ(desired_height.size(), batch_size)
      << "argument desired_height should be equal to batch size";
  MXCHECK_EQ(desired_width.size(), batch_size)
      << "argument desired_width should be equal to batch size";
  size_t op_buffer_size;
  if (use_pillow_resize) {
    op_buffer_size = calPillowResizeBufferSize(images, desired_height, desired_width);
  } else {
    op_buffer_size = resize_op_->calBufferSize(batch_size);
  }
  std::shared_ptr<void> cpu_buffer_ptr(malloc(op_buffer_size), free);
  void* gpu_workspace = cuda_api_->Alloc(ctx_, op_buffer_size);

  const void* input_ptr[batch_size];
  void* output_ptr[batch_size];

  cv::Size dsize[batch_size];
  double fx[batch_size];
  double fy[batch_size];

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

    int tmp_width = desired_width[i].As<int>();
    int tmp_height = desired_height[i].As<int>();

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
    dsize[i] = cv::Size(tmp_width, tmp_height);
    fx[i] = 1.0;
    fy[i] = 1.0;
    i += 1;
  }

  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);

  if (use_pillow_resize) {
    pillow_resize_op_->infer(input_ptr,
                             output_ptr,
                             gpu_workspace,
                             (void*)cpu_buffer_ptr.get(),
                             batch_size,
                             op_buffer_size,
                             dsize,
                             -interp_flags - 1,
                             input_shape,
                             cuda_op::kNHWC,
                             op_data_type,
                             cu_stream);
  } else {
    resize_op_->infer(input_ptr,
                      output_ptr,
                      gpu_workspace,
                      (void*)cpu_buffer_ptr.get(),
                      batch_size,
                      op_buffer_size,
                      dsize,
                      fx,
                      fy,
                      interp_flags,
                      input_shape,
                      cuda_op::kNHWC,
                      op_data_type,
                      cu_stream);
  }

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

MATX_REGISTER_NATIVE_OBJECT(VisionResizeOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionResizeOpGPU] Expect 1 argument but get " << args.size();
      return std::make_shared<VisionResizeOpGPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 5) << "[VisionResizeOpGPU][func: process] Expect 5 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionResizeOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(),
          args[1].AsObjectView<List>().data(),
          args[2].AsObjectView<List>().data(),
          args[3].As<unicode_view>(),
          args[4].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision
