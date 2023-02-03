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
#include "utils/pad_types.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;
using matxscript::runtime::DataType;

class VisionSumOrMeanOpGPU : public VisionBaseImageOpGPU<cuda_op::SumOrMeanVarShape> {
 public:
  VisionSumOrMeanOpGPU(const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::SumOrMeanVarShape>(session_info){};
  RTValue process(const List& images, bool per_channel, bool get_mean, int sync);
};

RTValue VisionSumOrMeanOpGPU::process(const List& arg_images,
                                      bool per_channel,
                                      bool get_mean,
                                      int sync) {
  // TODO: check if necessary
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());

  // parse input
  int batch_size = images.size();

  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  auto tmp_view = images[0].AsObjectView<NDArray>();
  const NDArray& tmp_ndarray = tmp_view.data();
  std::vector<int64_t> tmp_shape = tmp_ndarray.Shape();
  int channel;
  if (tmp_shape.size() == 2)
    channel = 1;
  else if (tmp_shape.size() == 3)
    channel = tmp_shape[2];
  else
    MXCHECK(false) << "Invalid input. The image shape should either has 2 dim or 3 dim.";
  size_t op_buffer_size = op_->calBufferSize(channel, batch_size);

  std::shared_ptr<void> cpu_buffer_ptr(malloc(op_buffer_size), free);
  void* gpu_workspace;

  const void* input_ptr[batch_size];
  void* output_ptr[batch_size];

  cuda_op::DataShape input_shape[batch_size];
  DataType nd_data_type;

  int i = 0;
  for (const RTValue& nd_elem : images) {
    auto view_elem = nd_elem.AsObjectView<NDArray>();
    const NDArray& elem = view_elem.data();
    input_ptr[i] = (void*)(elem->data);
    std::vector<int64_t> src_shape = elem.Shape();

    if (i == 0) {
      nd_data_type = elem.DataType();
    } else {
      if (src_shape.size() < 2 || src_shape.size() > 3)
        MXCHECK(false) << "Invalid input. The image shape should either has 2 dim or 3 dim.";
      if (src_shape.size() == 3 && channel != src_shape[2]) {
        MXCHECK(false) << "Invalid input. The input channel should be equal";
      }
      if (nd_data_type != elem.DataType()) {
        MXCHECK(false) << "The inputs must have same data type";
      }
    }

    input_shape[i].N = 1;
    input_shape[i].C = channel;
    input_shape[i].H = src_shape[0];
    input_shape[i].W = src_shape[1];
    i += 1;
  }

  DLDataType t;
  t.lanes = 1;
  t.bits = 64;
  t.code = kDLFloat;
  DataType target_data_type(t);

  int output_channel_size = per_channel ? channel : 1;
  std::vector<int64_t> out_shape = {batch_size, output_channel_size};

  size_t output_buffer_size = CalculateOutputBufferSize(out_shape, target_data_type);

  NDArray res = MakeNDArrayWithWorkSpace(
      ctx_,
      cuda_api_,
      output_buffer_size,
      out_shape,
      target_data_type,
      [finish_event, images, cpu_buffer_ptr]() {
        CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
        CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
      },
      op_buffer_size,
      &gpu_workspace);

  void* cur_ptr = (void*)(res->data);
  for (int i = 0; i < batch_size; i++) {
    output_ptr[i] = cur_ptr;
    cur_ptr += output_channel_size * sizeof(double);
  }

  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);

  op_->infer(input_ptr,
             output_ptr,
             gpu_workspace,
             (void*)cpu_buffer_ptr.get(),
             batch_size,
             op_buffer_size,
             per_channel,
             get_mean,
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

MATX_REGISTER_NATIVE_OBJECT(VisionSumOrMeanOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionSumOrMeanOpGPU] Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionSumOrMeanOpGPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 4)
                            << "[VisionSumOrMeanOpGPU][func: process] Expect 4 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionSumOrMeanOpGPU*>(self)->process(
                            args[0].AsObjectView<List>().data(),
                            args[1].As<bool>(),
                            args[2].As<bool>(),
                            args[3].As<int>());
                      });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision
