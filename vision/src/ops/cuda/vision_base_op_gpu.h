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

#pragma once
#include <cuda_runtime.h>
#include <cv_cuda.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/threadpool/i_thread_pool.h>
#include <utils/cuda/config.h>
#include "driver_types.h"

namespace byted_matx_vision {
namespace ops {
namespace cuda {

class VisionBaseOpGPU {
 public:
  VisionBaseOpGPU(const matxscript::runtime::Any& session_info);
  ~VisionBaseOpGPU() = default;

 protected:
  cudaStream_t getStream();
  int device_id_;
  DLDevice ctx_;
  matxscript::runtime::DeviceAPI* cuda_api_;
  ::matxscript::runtime::internal::IThreadPool* thread_pool_;
};

template <typename T>
class VisionBaseImageOpGPU : public VisionBaseOpGPU {
 public:
  VisionBaseImageOpGPU(const matxscript::runtime::Any& session_info)
      : VisionBaseOpGPU(session_info) {
    max_input_shape_.N = max_batch_size_;
    max_input_shape_.C = 3;
    max_input_shape_.H = 1024;
    max_input_shape_.W = 1024;
    op_ = std::make_shared<T>(max_input_shape_, max_output_shape_);
  }
  ~VisionBaseImageOpGPU() = default;

 protected:
  int max_batch_size_ = 1;
  std::shared_ptr<T> op_;
  cuda_op::DataShape max_input_shape_, max_output_shape_;
};

}  // namespace cuda
}  // namespace ops
}  // namespace byted_matx_vision