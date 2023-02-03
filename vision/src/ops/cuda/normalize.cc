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
#include "utils/type_helper.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

class VisionNormalizeOpGPU : public VisionBaseImageOpGPU<cuda_op::NormalizeVarShape> {
 public:
  VisionNormalizeOpGPU(const Any& session_info,
                       const List& mean,
                       const List& std,
                       float global_shift,
                       float global_scale,
                       const unicode_view& fmt);
  RTValue process(const List& images, int sync);

 private:
  float global_scale_;
  float global_shift_;
  float epsilon_ = 0.0;
  bool scale_is_stddev_ = true;
  int channel_size_;
  void* mean_cuda_ptr;
  void* std_cuda_ptr;
  int cv_depth_type;
};

VisionNormalizeOpGPU::VisionNormalizeOpGPU(const Any& session_info,
                                           const List& list_mean,
                                           const List& list_std,
                                           float global_shift,
                                           float global_scale,
                                           const unicode_view& rtype)
    : VisionBaseImageOpGPU<cuda_op::NormalizeVarShape>(session_info) {
  if (list_mean.size() != list_std.size()) {
    MXTHROW << "The size of mean and std should be equal";
  }
  channel_size_ = list_mean.size();
  std::vector<float> tmp_mean;
  std::vector<float> tmp_std;
  for (int i = 0; i < channel_size_; i++) {
    tmp_mean.push_back(list_mean[i].As<float>());
    tmp_std.push_back(list_std[i].As<float>());
  }

  mean_cuda_ptr = cuda_api_->Alloc(ctx_, channel_size_ * sizeof(float));
  cudaMemcpy(mean_cuda_ptr, tmp_mean.data(), channel_size_ * sizeof(float), cudaMemcpyHostToDevice);
  std_cuda_ptr = cuda_api_->Alloc(ctx_, channel_size_ * sizeof(float));
  cudaMemcpy(std_cuda_ptr, tmp_std.data(), channel_size_ * sizeof(float), cudaMemcpyHostToDevice);
  cv_depth_type = UnicodeTypeToOpencvDepth(rtype);
  global_scale_ = global_scale;
  global_shift_ = global_shift;
}

RTValue VisionNormalizeOpGPU::process(const List& arg_image_ndarray, int sync) {
  // TODO: check if necessary
  check_and_set_device(device_id_);
  auto image_ndarray = check_copy(arg_image_ndarray, ctx_, getStream());
  int batch_size = image_ndarray.size();

  DataType target_data_type(OpencvDepthToDLDataType(cv_depth_type));
  cuda_op::DataType target_op_data_type = DLDataTypeToOpencvCudaType(target_data_type);

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
  for (const RTValue& nd_elem : image_ndarray) {
    auto view_elem = nd_elem.AsObjectView<NDArray>();
    const NDArray& elem = view_elem.data();
    input_ptr[i] = (void*)(elem->data);
    std::vector<int64_t> src_shape = elem.Shape();

    if (i == 0) {
      channel = src_shape[2];
      MXCHECK_EQ(channel_size_, channel) << "The size of mean should be equal to the channel size";
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
    size_t output_buffer_size = CalculateOutputBufferSize(src_shape, target_data_type);

    NDArray dst_arr = MakeNDArrayWithWorkSpace(
        ctx_,
        cuda_api_,
        output_buffer_size,
        src_shape,
        target_data_type,
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
             mean_cuda_ptr,
             std_cuda_ptr,
             input_shape,
             channel_size_,
             channel_size_,
             scale_is_stddev_,
             global_scale_,
             global_shift_,
             epsilon_,
             cuda_op::kNHWC,
             op_data_type,
             target_op_data_type,
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

MATX_REGISTER_NATIVE_OBJECT(VisionNormalizeOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 6) << "[VisionNormalizeOpGPU] Expect 6 arguments but get "
                                << args.size();
      return std::make_shared<VisionNormalizeOpGPU>(args[5],
                                                    args[0].AsObjectView<List>().data(),
                                                    args[1].AsObjectView<List>().data(),
                                                    args[2].As<float>(),
                                                    args[3].As<float>(),
                                                    args[4].As<unicode_view>());
    })
    .RegisterFunction("process",
                      [](void* self, PyArgs args) -> RTValue {
                        MXCHECK_EQ(args.size(), 2)
                            << "[VisionNormalizeOpGPU][func: process] Expect 2 arguments but get "
                            << args.size();
                        return reinterpret_cast<VisionNormalizeOpGPU*>(self)->process(
                            args[0].AsObjectView<List>().data(), args[1].As<int>());
                      });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision