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

#include "cv_cuda.h"
#include "matxscript/runtime/container/list_ref.h"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/container/unicode_view.h"
#include "matxscript/runtime/logging.h"
#include "matxscript/runtime/native_object_registry.h"
#include "matxscript/runtime/runtime_value.h"
#include "utils/cuda/cuda_op_helper.h"
#include "utils/cuda/cuda_type_helper.h"
#include "vision_base_op_gpu.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace ::matxscript::runtime;

class VisionTransposeOpGPU : public VisionBaseImageOpGPU<cuda_op::Reformat> {
 public:
  VisionTransposeOpGPU(const Any& session_info)
      : VisionBaseImageOpGPU<cuda_op::Reformat>(session_info){};
  RTValue process(const NDArray& images,
                  const unicode_view& src_fmt,
                  const unicode_view& dst_fmt,
                  int sync);
};

RTValue VisionTransposeOpGPU::process(const NDArray& arg_images,
                                      const unicode_view& src_fmt,
                                      const unicode_view& dst_fmt,
                                      int sync) {
  check_and_set_device(device_id_);
  auto images = check_copy(arg_images, ctx_, getStream());
  int cuda_src_fmt = UnicodeTODataFormat(src_fmt);
  int cuda_dst_fmt = UnicodeTODataFormat(dst_fmt);
  MXCHECK((cuda_src_fmt != cuda_op::DataFormat::kNCHW) or
          (cuda_src_fmt != cuda_op::DataFormat::kNHWC))
      << "[VisionTransposeOpGPU] src_fmt must be NCHW or NHWC, but get: " << src_fmt;
  MXCHECK((cuda_dst_fmt != cuda_op::DataFormat::kNCHW) or
          (cuda_dst_fmt != cuda_op::DataFormat::kNHWC))
      << "[VisionTransposeOpGPU] dst_fmt must be NCHW or NHWC, bug get: " << dst_fmt;

  std::vector<int64_t> src_shape = images.Shape();

  DataType nd_data_type = images.DataType();
  cuda_op::DataType op_data_type = DLDataTypeToOpencvCudaType(nd_data_type);

  // create event
  cudaStream_t cu_stream = getStream();
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));

  // create event for nd_array destructor
  cuda_op::DataShape input_shape;
  if (cuda_src_fmt == cuda_op::DataFormat::kNCHW) {
    input_shape.N = src_shape[0];
    input_shape.C = src_shape[1];
    input_shape.H = src_shape[2];
    input_shape.W = src_shape[3];
  } else {
    input_shape.N = src_shape[0];
    input_shape.C = src_shape[3];
    input_shape.H = src_shape[1];
    input_shape.W = src_shape[2];
  }

  std::vector<int64_t> out_shape;

  if (cuda_dst_fmt == cuda_op::DataFormat::kNCHW) {
    out_shape = {input_shape.N, input_shape.C, input_shape.H, input_shape.W};
  } else {
    out_shape = {input_shape.N, input_shape.H, input_shape.W, input_shape.C};
  }

  size_t output_buffer_size = CalculateOutputBufferSize(out_shape, nd_data_type);
  NDArray dst_arr = MakeNDArrayWithWorkSpace(
      ctx_,
      cuda_api_,
      output_buffer_size,
      out_shape,
      nd_data_type,
      [finish_event, images]() {
        CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
        CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
      },
      0,
      nullptr);
  void* p = dst_arr->data;

  cuda_op::DataFormat kernel_src_fmt = static_cast<cuda_op::DataFormat>(cuda_src_fmt);
  cuda_op::DataFormat kernel_dst_fmt = static_cast<cuda_op::DataFormat>(cuda_dst_fmt);
  op_->infer(&(images->data),
             &p,
             nullptr,
             input_shape,
             kernel_src_fmt,
             kernel_dst_fmt,
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
};

MATX_REGISTER_NATIVE_OBJECT(VisionTransposeOpGPU)
    .SetConstructor([](PyArgs args) -> std::shared_ptr<void> {
      MXCHECK_EQ(args.size(), 1) << "[VisionTransposeOpGPU] Constructor Expect 1 arguments but get "
                                 << args.size();
      return std::make_shared<VisionTransposeOpGPU>(args[0]);
    })
    .RegisterFunction("process", [](void* self, PyArgs args) -> RTValue {
      MXCHECK_EQ(args.size(), 4) << "[VisionTransposeOpGPU] Expect 4 arguments but get "
                                 << args.size();
      return reinterpret_cast<VisionTransposeOpGPU*>(self)->process(
          args[0].AsObjectView<NDArray>().data(),
          args[1].As<unicode_view>(),
          args[2].As<unicode_view>(),
          args[3].As<int>());
    });

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision