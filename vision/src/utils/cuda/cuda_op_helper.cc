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

#include "cuda_op_helper.h"

#include <matxscript/runtime/logging.h>
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/dlpack.h"
#include "utils/cuda/config.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using matxscript::runtime::DataType;
using matxscript::runtime::DeviceAPI;
using matxscript::runtime::List;
using matxscript::runtime::NDArray;
using matxscript::runtime::OpaqueObject;
using matxscript::runtime::StreamInfo;

struct DLManagedTensorContext {
  DeviceAPI* cuda_api;
  std::function<void()> sync_func;
};

static void DefaultDeleter(struct DLManagedTensor* managed_tensor) {
  if (managed_tensor) {
    auto* ctx = reinterpret_cast<DLManagedTensorContext*>(managed_tensor->manager_ctx);
    CHECK_CUDA_CALL(cudaSetDevice(managed_tensor->dl_tensor.device.device_id));
    ctx->sync_func();
    ctx->cuda_api->Free(managed_tensor->dl_tensor.device, managed_tensor->dl_tensor.data);
    delete ctx;
    delete managed_tensor;
  }
}

size_t CalculateOutputBufferSize(std::vector<int64_t>& image_shape, const DataType& data_type) {
  size_t shape_size = image_shape.size();
  size_t output_buffer_size = 0;
  switch (shape_size) {
    case 4:
      output_buffer_size = image_shape[0] * image_shape[1] * image_shape[2] * image_shape[3] *
                           ((data_type.bits() * data_type.lanes() + 7) / 8);
      return output_buffer_size;
      break;
    case 3:
      output_buffer_size = image_shape[0] * image_shape[1] * image_shape[2] *
                           ((data_type.bits() * data_type.lanes() + 7) / 8);
      return output_buffer_size;
      break;
    case 2:
      output_buffer_size =
          image_shape[0] * image_shape[1] * ((data_type.bits() * data_type.lanes() + 7) / 8);
      return output_buffer_size;
      break;

    default:
      MXCHECK(false) << "Not support output shape: " << shape_size
                     << " Only support image shape size: [2, 3, 4]";
      break;
  }
}

NDArray MakeNDArrayWithWorkSpace(const DLDevice& ctx,
                                 DeviceAPI* cuda_api,
                                 size_t buffer_size,
                                 std::vector<int64_t>& image_shape,
                                 const DataType& data_type,
                                 std::function<void()> sync_func,
                                 size_t extra_size,
                                 void** extra_workspace) {
  auto align_buffer_size = (buffer_size + (256 - 1)) / 256 * 256;
  auto real_size = align_buffer_size + extra_size;
  DLTensor dl_tensor;
  dl_tensor.data = cuda_api->Alloc(ctx, real_size);
  // cudaMemset(dl_tensor.data, 1, real_size);
  dl_tensor.device = ctx;
  dl_tensor.byte_offset = 0;
  dl_tensor.dtype = data_type;
  dl_tensor.strides = nullptr;
  dl_tensor.ndim = static_cast<int>(image_shape.size());
  dl_tensor.shape = matxscript::runtime::BeginPtr(image_shape);

  // TODO: use placement new
  auto* dl_man_tensor = new DLManagedTensor;
  dl_man_tensor->dl_tensor = dl_tensor;
  dl_man_tensor->manager_ctx = new DLManagedTensorContext{cuda_api, std::move(sync_func)};
  dl_man_tensor->deleter = DefaultDeleter;

  // assign extra workspace
  if (extra_size && extra_workspace) {
    *extra_workspace = reinterpret_cast<unsigned char*>(dl_tensor.data) + align_buffer_size;
  }

  NDArray dst_arr = NDArray::FromDLPack(dl_man_tensor);
  return dst_arr;
}

MATXScriptStreamHandle GetCudaStreamFromOpaqueObject(const OpaqueObject& opaque_object) {
  if (!opaque_object.defined()) {
    // default is null stream
    return nullptr;
  }
  StreamInfo* stream_info = reinterpret_cast<StreamInfo*>(opaque_object.GetOpaquePtr());
  return reinterpret_cast<MATXScriptStreamHandle>(stream_info->device_stream);
}

void check_and_set_device(int device_id) {
  int current_device = -1;
  CHECK_CUDA_CALL(cudaGetDevice(&current_device));
  if (current_device != device_id) {
    CHECK_CUDA_CALL(cudaSetDevice(device_id));
  }
}

NDArray to_cpu(const NDArray& image, MATXScriptStreamHandle stream) {
  MXCHECK(image.IsContiguous()) << "[sync to cpu]: ndarray should be contiguous";
  NDArray to = NDArray::Empty(image.Shape(), image->dtype, DLDevice{kDLCPU, 0});
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));
  DeviceAPI::Get(image->device)
      ->CopyDataFromTo(image->data,
                       static_cast<size_t>(image->byte_offset),
                       to->data,
                       static_cast<size_t>(to->byte_offset),
                       image.DataSize(),
                       image->device,
                       to->device,
                       image->dtype,
                       stream);
  CHECK_CUDA_CALL(cudaEventRecord(finish_event, static_cast<cudaStream_t>(stream)));
  CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
  CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
  return to;
}

List to_cpu(const List& images, MATXScriptStreamHandle stream) {
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));
  List ret;
  for (auto& item : images) {
    auto view = item.AsObjectView<NDArray>();
    const NDArray& image = view.data();
    MXCHECK(image.IsContiguous()) << "[sync to cpu]: ndarray should be contiguous";
    NDArray to = NDArray::Empty(image.Shape(), image->dtype, DLDevice{kDLCPU, 0});
    DeviceAPI::Get(image->device)
        ->CopyDataFromTo(image->data,
                         static_cast<size_t>(image->byte_offset),
                         to->data,
                         static_cast<size_t>(to->byte_offset),
                         image.DataSize(),
                         image->device,
                         to->device,
                         image->dtype,
                         stream);
    ret.append(to);
  }
  CHECK_CUDA_CALL(cudaEventRecord(finish_event, static_cast<cudaStream_t>(stream)));
  CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
  CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
  return ret;
}

List check_copy(const List& images, DLDevice ctx, MATXScriptStreamHandle stream) {
  int cpu_ctx_num = 0;
  for (auto& item : images) {
    auto t_ctx = item.AsObjectView<NDArray>().data()->device;
    if (t_ctx.device_type == kDLCPU) {
      cpu_ctx_num += 1;
      continue;
    } else if (t_ctx.device_type != ctx.device_type || t_ctx.device_id != ctx.device_id) {
      MXTHROW << "input ndarray is expected on device " << ctx.device_id << ", but get "
              << t_ctx.device_id;
    }
  }

  if (cpu_ctx_num == 0) {
    return images;
  }

  if (cpu_ctx_num != images.size()) {
    MXTHROW << "cpu and gpu mixed input";
  }

  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));
  List ret;
  for (auto& item : images) {
    auto view = item.AsObjectView<NDArray>();
    const NDArray& image = view.data();
    MXCHECK(image.IsContiguous()) << "[copy to gpu]: ndarray should be contiguous";
    NDArray to = NDArray::Empty(image.Shape(), image->dtype, ctx);
    DeviceAPI::Get(ctx)->CopyDataFromTo(image->data,
                                        static_cast<size_t>(image->byte_offset),
                                        to->data,
                                        static_cast<size_t>(to->byte_offset),
                                        image.DataSize(),
                                        image->device,
                                        to->device,
                                        image->dtype,
                                        stream);
    ret.append(to);
  }
  CHECK_CUDA_CALL(cudaEventRecord(finish_event, static_cast<cudaStream_t>(stream)));
  CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
  CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
  return ret;
}

NDArray check_copy(const NDArray& image, DLDevice ctx, MATXScriptStreamHandle stream) {
  auto in_ctx = image->device;
  if (in_ctx.device_type != kDLCPU) {
    MXCHECK(in_ctx.device_id == ctx.device_id and in_ctx.device_type == ctx.device_type)
        << "input ndarray is expected on device " << ctx.device_id << ", but get "
        << in_ctx.device_id;

    return image;
  }
  cudaEvent_t finish_event;
  CHECK_CUDA_CALL(cudaEventCreate(&finish_event));
  NDArray to = NDArray::Empty(image.Shape(), image->dtype, ctx);
  DeviceAPI::Get(ctx)->CopyDataFromTo(image->data,
                                      static_cast<size_t>(image->byte_offset),
                                      to->data,
                                      static_cast<size_t>(to->byte_offset),
                                      image.DataSize(),
                                      image->device,
                                      to->device,
                                      image->dtype,
                                      stream);
  CHECK_CUDA_CALL(cudaEventRecord(finish_event, static_cast<cudaStream_t>(stream)));
  CHECK_CUDA_CALL(cudaEventSynchronize(finish_event));
  CHECK_CUDA_CALL(cudaEventDestroy(finish_event));
  return to;
}

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision
