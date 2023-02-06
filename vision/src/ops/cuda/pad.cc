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
#include <vector>
#include "cv_cuda.h"
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/container/tuple_ref.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/global_type_index.h"
#include "matxscript/runtime/runtime_value.h"
#include "utils/cuda/cuda_op_helper.h"
#include "utils/cuda/cuda_type_helper.h"
#include "utils/pad_types.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

class VisionPadOpGPU : public VisionBaseImageOpGPU<cuda_op::CopyMakeBorderVarShape> {
 public:
  VisionPadOpGPU(const Tuple& pad_values, const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::CopyMakeBorderVarShape>(session_info) {
    if (pad_values.size() != 3 || pad_values[0].type_code() != TypeIndex::kRuntimeInteger) {
      MXTHROW << "[BytedVisionPadCPU]: pad_values size must equals to three !";
    }
    for (const RTValue& pad_value : pad_values) {
      pad_values_.emplace_back(pad_value.As<int>());
    }
  };
  RTValue process(const List& images,
                  const List& top_pads,
                  const List& bottom_pads,
                  const List& left_pads,
                  const List& right_pads,
                  const unicode_view& border_type,
                  int sync);

 private:
  std::vector<int> pad_values_;  // CHW
};

RTValue VisionPadOpGPU::process(const List& arg_images,
                                const List& top_pads,
                                const List& bottom_pads,
                                const List& left_pads,
                                const List& right_pads,
                                const unicode_view& border_type,
                                int sync) {
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());

  int batch_size = images.size();
  int cv_border_type = UnicodePadTypesToCVBorderTypes(border_type);

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
  cv::Scalar scalar_value(pad_values_[0], pad_values_[1], pad_values_[2]);
  std::vector<int> top_vec, bottom_vec, left_vec, right_vec;
  top_vec.reserve(batch_size);
  bottom_vec.reserve(batch_size);
  left_vec.reserve(batch_size);
  right_vec.reserve(batch_size);

  const void* input_ptr[batch_size];
  void* output_ptr[batch_size];

  cuda_op::DataShape input_shape[batch_size];
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
    int cur_top = top_pads[i].As<int>();
    int cur_bottom = bottom_pads[i].As<int>();
    int cur_left = left_pads[i].As<int>();
    int cur_right = right_pads[i].As<int>();
    top_vec.emplace_back(cur_top);
    bottom_vec.emplace_back(cur_bottom);
    left_vec.emplace_back(cur_left);
    right_vec.emplace_back(cur_right);

    std::vector<int64_t> out_shape{
        src_shape[0] + cur_top + cur_bottom, src_shape[1] + cur_left + cur_right, src_shape[2]};
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
             top_vec.data(),
             bottom_vec.data(),
             left_vec.data(),
             right_vec.data(),
             false,
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
};

MATX_REGISTER_NATIVE_OBJECT(VisionPadOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 2) << "[BytedVisionPadGPU] Constructor Expect 2 arguments but get "
                                 << args.size();
      return std::make_shared<VisionPadOpGPU>(args[0].AsObjectView<Tuple>().data(), args[1]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 7) << "[BytedVisionPadGPU] Expect 7 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionPadOpGPU*>(self)->process(args[0].AsObjectView<List>().data(),
                                                              args[1].AsObjectView<List>().data(),
                                                              args[2].AsObjectView<List>().data(),
                                                              args[3].AsObjectView<List>().data(),
                                                              args[4].AsObjectView<List>().data(),
                                                              args[5].As<unicode_view>(),
                                                              args[6].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision