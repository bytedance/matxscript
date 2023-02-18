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

#include <opencv_cuda.h>
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/runtime_value.h"
#include "utils/cuda/cuda_op_helper.h"
#include "utils/cuda/cuda_type_helper.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace ::matxscript::runtime;

class VisionStackOpGPU : public VisionBaseImageOpGPU<cuda_op::Stack> {
 public:
  VisionStackOpGPU(const Any& session_info) : VisionBaseImageOpGPU<cuda_op::Stack>(session_info){};
  RTValue process(const List& images, int sync);
};

RTValue VisionStackOpGPU::process(const List& auto_images, int sync) {
  check_and_set_device(device_id_);
  auto images = check_copy(auto_images, ctx_, getStream());
  int64_t batch_size = images.size();
  std::vector<int64_t> src_shape;
  DataType nd_data_type;
  size_t op_buffer_size = op_->calBufferSize(batch_size);
  std::shared_ptr<void> cpu_buffer_ptr(malloc(op_buffer_size), free);
  const void* input_ptr[batch_size];
  int i = 0;
  for (const RTValue& rt_elem : images) {
    auto view_elem = rt_elem.AsObjectView<NDArray>();
    const NDArray& elem = view_elem.data();
    if (i == 0) {
      src_shape = elem.Shape();
      MXCHECK_EQ(src_shape.size(), 3)
          << "Invalid input data shape. The dim of each input element shoud be 3.";
      nd_data_type = elem.DataType();
    } else {
      if (src_shape != elem.Shape()) {
        MXCHECK(false) << "The inputs must have same shape";
      }
      if (nd_data_type != elem.DataType()) {
        MXCHECK(false) << "The inputs must have same data type";
      }
    }
    input_ptr[i] = (void*)(elem->data);
    i += 1;
  }
  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);
  // create event for nd_array destructor
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));
  cudaStream_t cu_stream = getStream();

  cuda_op::DataShape input_shape;
  input_shape.N = 1;
  input_shape.C = src_shape[2];
  input_shape.H = src_shape[0];
  input_shape.W = src_shape[1];

  std::vector<int64_t> out_shape;
  out_shape = {batch_size, src_shape[0], src_shape[1], src_shape[2]};

  size_t output_buffer_size = CalculateOutputBufferSize(out_shape, nd_data_type);

  void* q;

  NDArray dst_arr = MakeNDArrayWithWorkSpace(
      ctx_,
      cuda_api_,
      output_buffer_size,
      out_shape,
      nd_data_type,
      [finish_event, images, cpu_buffer_ptr]() {
        CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
        CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
      },
      op_buffer_size,
      &q);

  void* p = dst_arr->data;

  op_->infer(input_ptr,
             &p,
             q,
             (void*)cpu_buffer_ptr.get(),
             batch_size,
             op_buffer_size,
             input_shape,
             cuda_op::kNHWC,
             op_data_type,
             cu_stream);
  CHECK_CUDA_CALL(cudaEventRecord(finish_event, cu_stream));
  CUDA_EVENT_SYNC_IF_DEBUG(finish_event);
  CUDA_STREAM_SYNC_IF_DEBUG(cu_stream);
  CUDA_DEVICE_SYNC_IF_DEBUG();
  if (sync != VISION_SYNC_MODE::ASYNC) {
    CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
    if (sync == VISION_SYNC_MODE::SYNC_CPU) {
      return to_cpu(dst_arr, getStream());
    } else {
      return dst_arr;
    }
  }
  return dst_arr;
}

MATX_REGISTER_NATIVE_OBJECT(VisionStackOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionStackOpGPU] Expect 1 arguments but get " << args.size();
      return std::make_shared<VisionStackOpGPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 2) << "[VisionStackOpGPU][func: process] Expect 2 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionStackOpGPU*>(self)->process(args[0].AsObjectView<List>().data(),
                                                                args[1].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision