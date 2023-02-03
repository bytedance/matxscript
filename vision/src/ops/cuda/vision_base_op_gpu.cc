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

#include "vision_base_op_gpu.h"

#include "matxscript/pipeline/attributes.h"
#include "matxscript/runtime/threadpool/i_thread_pool.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

using namespace matxscript::runtime;

cudaStream_t VisionBaseOpGPU::getStream() {

  //cudaStream_t t = static_cast<cudaStream_t>(cuda_api_->GetDefaultComputeStream(ctx_));
  //std::stringstream ss;
  //ss<<"byted vision thread id"<<std::this_thread::get_id()<<std::endl;
  //ss<<"byted vision stream: "<<t<<std::endl; 
  //std::cout<<ss.str();
  return static_cast<cudaStream_t>(cuda_api_->GetCurrentThreadStream(ctx_));
}

VisionBaseOpGPU::VisionBaseOpGPU(const Any& session_info) {
  auto view = session_info.AsObjectView<Dict>();
  const Dict& info = view.data();
  int session_id = info["session_device_id"].As<int>();
  if (session_id == NONE_DEVICE) {
    device_id_ = info["device_id"].As<int>();
  } else {
    device_id_ = session_id;
  }
  CHECK_CUDA_CALL(cudaSetDevice(device_id_));
  ctx_.device_id = device_id_;
  ctx_.device_type = DLDeviceType::kDLCUDA;
  cuda_api_ = matxscript::runtime::DeviceAPI::Get(ctx_);
  void* pool = info["thread_pool"].As<void*>();
  thread_pool_ = static_cast<internal::IThreadPool*>(pool);
}

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision