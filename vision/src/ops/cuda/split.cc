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
#include "utils/pad_types.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

class VisionSplitOpGPU : public VisionBaseImageOpGPU<cuda_op::Split> {
 public:
  VisionSplitOpGPU(const Any& session_info) : VisionBaseImageOpGPU<cuda_op::Split>(session_info){};
  RTValue process(const NDArray& image, int sync);
};

RTValue VisionSplitOpGPU::process(const NDArray& arg_image, int sync) {
  // TODO: check if necessary
  check_and_set_device(device_id_);
  auto image = check_copy(arg_image, ctx_, getStream());

  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  auto finish_event_mutex = std::make_shared<std::mutex>();
  auto not_finish = std::make_shared<bool>(true);

  std::vector<int64_t> src_shape = image.Shape();
  std::vector<int64_t> out_shape;
  cuda_op::DataShape input_shape;
  if (src_shape.size() == 3) {
    input_shape.N = 1;
    input_shape.C = src_shape[2];
    input_shape.H = src_shape[0];
    input_shape.W = src_shape[1];
    out_shape = {src_shape[0], src_shape[1]};
  } else {
    input_shape.N = src_shape[0];
    input_shape.C = src_shape[3];
    input_shape.H = src_shape[1];
    input_shape.W = src_shape[2];
    out_shape = {src_shape[0], src_shape[1], src_shape[2]};
  }

  DataType nd_data_type = image.DataType();
  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);
  size_t output_buffer_size = CalculateOutputBufferSize(out_shape, nd_data_type);

  List res;
  void* output_ptr[input_shape.C];
  for (int i = 0; i < input_shape.C; i++) {
    NDArray dst_arr = MakeNDArrayWithWorkSpace(
        ctx_,
        cuda_api_,
        output_buffer_size,
        out_shape,
        nd_data_type,
        [finish_event_mutex, not_finish, finish_event, image]() {
          std::lock_guard<std::mutex> lock(*finish_event_mutex);
          if (*not_finish) {
            CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
            CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
            *not_finish = false;
          }
        },
        0,
        nullptr);
    res.push_back(dst_arr);
    output_ptr[i] = dst_arr->data;
  }

  op_->infer(
      &(image->data), output_ptr, nullptr, input_shape, cuda_op::kNHWC, op_data_type, cu_stream);

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

MATX_REGISTER_NATIVE_OBJECT(VisionSplitOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionSplitOpGPU] Expect 1 arguments but get " << args.size();
      return std::make_shared<VisionSplitOpGPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 2) << "[VisionSplitOpGPU][func: process] Expect 2 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionSplitOpGPU*>(self)->process(
          args[0].AsObjectView<NDArray>().data(), args[1].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision
