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

class VisionMixupImagesOpGPU : public VisionBaseImageOpGPU<cuda_op::MixupImagesVarShape> {
 public:
  VisionMixupImagesOpGPU(const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::MixupImagesVarShape>(session_info){};
  RTValue process(
      const List& images1, const List& images2, const List& factor1, const List& factor2, int sync);
};

RTValue VisionMixupImagesOpGPU::process(const List& arg_images1,
                                        const List& arg_images2,
                                        const List& factor1,
                                        const List& factor2,
                                        int sync) {
  // TODO: check if necessary
  check_and_set_device(device_id_);
  auto images1 = check_copy(arg_images1, ctx_, getStream());
  auto images2 = check_copy(arg_images2, ctx_, getStream());

  // parse input
  int batch_size = images1.size();
  MXCHECK(images2.size() == batch_size)
      << "Invalid input. The addend and the augend images should have the same size.";

  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  auto finish_event_mutex = std::make_shared<std::mutex>();
  auto not_finish = std::make_shared<bool>(true);

  size_t op_buffer_size = op_->calBufferSize(batch_size);

  std::shared_ptr<void> cpu_buffer_ptr(malloc(op_buffer_size), free);
  void* gpu_workspace = cuda_api_->Alloc(ctx_, op_buffer_size);

  const void* input_ptr[2 * batch_size];
  void* output_ptr[batch_size];

  cuda_op::DataShape input_shape[2 * batch_size];
  int channel1 = 0;
  int channel2 = 0;
  DataType nd_data_type;
  List res;
  float factor1_[batch_size];
  float factor2_[batch_size];

  int i = 0;
  for (const RTValue& nd_elem1 : images1) {
    auto view_elem1 = nd_elem1.AsObjectView<NDArray>();
    const NDArray& elem1 = view_elem1.data();
    input_ptr[2 * i] = (void*)(elem1->data);
    std::vector<int64_t> src_shape1 = elem1.Shape();

    auto view_elem2 = images2[i].AsObjectView<NDArray>();
    const NDArray& elem2 = view_elem2.data();
    input_ptr[2 * i + 1] = (void*)(elem2->data);
    std::vector<int64_t> src_shape2 = elem2.Shape();
    if (src_shape2.size() == 2) {
      src_shape2.push_back(1);
    }

    if (i == 0) {
      channel1 = src_shape1[2];
      channel2 = src_shape2[2];
      nd_data_type = elem1.DataType();
    } else {
      if (channel1 != src_shape1[2]) {
        MXCHECK(false) << "Invalid input. The channel size for augend images should be equal";
      }
      if (nd_data_type != elem1.DataType()) {
        MXCHECK(false) << "Invalid input. The inputs must have same data type";
      }
      if (channel2 != src_shape2[2]) {
        MXCHECK(false) << "Invalid input. The channel size for addend images should be equal";
      }
    }
    if (nd_data_type != elem2.DataType()) {
      MXCHECK(false)
          << "Invalid input. The addend and the augend images should have the same data type";
    }

    input_shape[2 * i].N = 1;
    input_shape[2 * i].C = channel1;
    input_shape[2 * i].H = src_shape1[0];
    input_shape[2 * i].W = src_shape1[1];

    input_shape[2 * i + 1].N = 1;
    input_shape[2 * i + 1].C = channel2;
    input_shape[2 * i + 1].H = src_shape2[0];
    input_shape[2 * i + 1].W = src_shape2[1];

    factor1_[i] = factor1[i].As<float>();
    factor2_[i] = factor2[i].As<float>();

    size_t output_buffer_size = CalculateOutputBufferSize(src_shape1, nd_data_type);

    NDArray dst_arr = MakeNDArrayWithWorkSpace(
        ctx_,
        cuda_api_,
        output_buffer_size,
        src_shape1,
        nd_data_type,
        [finish_event,
         finish_event_mutex,
         not_finish,
         elem1,
         elem2,
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
             factor1_,
             factor2_,
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

MATX_REGISTER_NATIVE_OBJECT(VisionMixupImagesOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionMixupImagesOpGPU] Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionMixupImagesOpGPU>(args[0]);
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 5)
                            << "[VisionMixupImagesOpGPU][func: process] Expect 5 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionMixupImagesOpGPU*>(self)->process(
                            args[0].AsObjectView<List>().data(),
                            args[1].AsObjectView<List>().data(),
                            args[2].AsObjectView<List>().data(),
                            args[3].AsObjectView<List>().data(),
                            args[4].As<int>());
                      });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision
