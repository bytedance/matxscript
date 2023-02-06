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

class VisionCastOpGPU : public VisionBaseImageOpGPU<cuda_op::ConvertTo> {
 public:
  VisionCastOpGPU(const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::ConvertTo>(session_info){};
  RTValue process(
      const NDArray& images, const unicode_view& dtype, double alpha, double beta, int sync);
};

RTValue VisionCastOpGPU::process(
    const NDArray& arg_images, const unicode_view& dtype, double alpha, double beta, int sync) {
  // TODO: check if necessary
  check_and_set_device(device_id_);
  auto input_images = check_copy(arg_images, ctx_, getStream());

  std::vector<int64_t> src_shape = input_images.Shape();
  cuda_op::DataShape input_shape;
  cuda_op::DataFormat data_fmt;
  if (src_shape.size() == 3) {
    input_shape.N = 1;
    input_shape.C = src_shape[2];
    input_shape.H = src_shape[0];
    input_shape.W = src_shape[1];
    data_fmt = cuda_op::kHWC;
  } else if (src_shape.size() == 4) {
    input_shape.N = src_shape[0];
    input_shape.C = src_shape[3];
    input_shape.H = src_shape[1];
    input_shape.W = src_shape[2];
    data_fmt = cuda_op::kNHWC;
  } else {
    MXCHECK(false) << "Invalid data format. input data should be either HWC or NHWC";
  }

  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  DataType nd_data_type = input_images.DataType();
  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);

  int cv_depth_type = UnicodeTypeToOpencvDepth(dtype);
  DataType target_data_type(OpencvDepthToDLDataType(cv_depth_type));
  cuda_op::DataType target_op_data_type = DLDataTypeToOpencvCudaType(target_data_type);

  size_t output_buffer_size = CalculateOutputBufferSize(src_shape, target_data_type);

  NDArray dst_arr = MakeNDArrayWithWorkSpace(
      ctx_,
      cuda_api_,
      output_buffer_size,
      src_shape,
      target_data_type,
      [finish_event, input_images]() {
        CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
        CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
      },
      0,
      nullptr);

  void* p = dst_arr->data;
  op_->infer(&(input_images->data),
             &p,
             nullptr,
             target_op_data_type,
             alpha,
             beta,
             input_shape,
             data_fmt,
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
      return to_cpu(dst_arr, getStream());
    } else {
      return dst_arr;
    }
  }
  return dst_arr;
}

MATX_REGISTER_NATIVE_OBJECT(VisionCastOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK(args.size() == 1) << "[VisionCastOpGPU] Expect 1 argument but get " << args.size();
      return std::make_shared<VisionCastOpGPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 5) << "[VisionCastOpGPU][func: process] Expect 5 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionCastOpGPU*>(self)->process(
          args[0].AsObjectView<NDArray>().data(),
          args[1].As<unicode_view>(),
          args[2].As<double>(),
          args[3].As<double>(),
          args[4].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision
