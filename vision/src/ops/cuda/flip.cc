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

class VisionFlipOpGPU : public VisionBaseImageOpGPU<cuda_op::FlipOrCopyVarShape> {
 public:
  VisionFlipOpGPU(const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::FlipOrCopyVarShape>(session_info){};
  RTValue process(const List& images, const Any& code, int sync);

 private:
  std::vector<int> get_code_vec(const Any& code, int num);
};

std::vector<int> VisionFlipOpGPU::get_code_vec(const Any& code, int num) {
  if (code.type_code() == TypeIndex::kRuntimeList) {
    auto view = code.AsObjectViewNoCheck<List>();
    const List& code_list = view.data();
    std::vector<int> ret_code_list;
    ret_code_list.reserve(code_list.size());
    for (const RTValue& item : code_list) {
      ret_code_list.push_back(item.As<int>());
    }
    return ret_code_list;
  } else {
    return std::vector<int>(num, code.As<int>());
  }
}

RTValue VisionFlipOpGPU::process(const List& arg_images, const Any& code, int sync) {
  // TODO: check if necessary
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());

  // parse input
  int batch_size = images.size();
  std::vector<int> flip_code_vec = get_code_vec(code, batch_size);

  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  auto finish_event_mutex = std::make_shared<std::mutex>();
  auto not_finish = std::make_shared<bool>(true);

  size_t op_buffer_size = op_->calBufferSize(batch_size);
  std::shared_ptr<void> cpu_buffer_ptr(malloc(op_buffer_size), free);
  void* gpu_workspace = cuda_api_->Alloc(ctx_, op_buffer_size);

  const void* input_ptr[batch_size];
  void* output_ptr[batch_size];

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
    size_t output_buffer_size = CalculateOutputBufferSize(src_shape, nd_data_type);

    NDArray dst_arr = MakeNDArrayWithWorkSpace(
        ctx_,
        cuda_api_,
        output_buffer_size,
        src_shape,
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
    i += 1;
  }
  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);

  op_->infer(input_ptr,
             output_ptr,
             gpu_workspace,
             (void*)cpu_buffer_ptr.get(),
             batch_size,
             op_buffer_size,
             flip_code_vec.data(),
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

MATX_REGISTER_NATIVE_OBJECT(VisionFlipOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionFlipOpGPU] Expect 1 arguments but get " << args.size();
      return std::make_shared<VisionFlipOpGPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 3) << "[VisionFlipOpGPU][func: process] Expect 3 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionFlipOpGPU*>(self)->process(
          args[0].AsObjectView<List>().data(), args[1], args[2].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision