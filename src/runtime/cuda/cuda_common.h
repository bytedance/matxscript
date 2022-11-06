// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
 *
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

/*!
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef MATXSCRIPT_RUNTIME_CUDA_CUDA_COMMON_H_
#define MATXSCRIPT_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cuda_runtime.h>

#include <string>

#include "core/device/cuda/cuda_allocator.h"
#include "core/framework/allocator.h"
#include "core/framework/arena.h"
#include "core/framework/bfc_arena.h"

namespace matxscript {
namespace runtime {

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      MXLOG(FATAL) << "CUDAError: " #x " failed with error: " << msg;   \
    }                                                                   \
  }

#define CUDA_CALL(func)                                        \
  {                                                            \
    cudaError_t e = (func);                                    \
    MXCHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                  \
  }

/*! \brief simple lock workspace */
class CUDAGlobalEntry {
 public:
  /*! \brief cuda stream map*/
  static thread_local std::shared_ptr<void> thread_local_stream;
  /*! \brief constructor */
  CUDAGlobalEntry();
  // get the workspace
  static CUDAGlobalEntry* Get();
};
thread_local std::shared_ptr<void> CUDAGlobalEntry::thread_local_stream =
    0;  // 0 presents the default cuda stream
}  // namespace runtime
}  // namespace matxscript
#endif  // MATXSCRIPT_RUNTIME_CUDA_CUDA_COMMON_H_
