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
#include <matxscript/runtime/logging.h>

#define CHECK_CUDA_DRIVER_CALL(x)                                       \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      MXTHROW << "CUDAError: " #x " failed with error: " << msg;        \
    }                                                                   \
  }

#define CHECK_CUDA_CALL(cudaFunction)                                         \
  {                                                                           \
    cudaError_t e = (cudaFunction);                                           \
    MXCHECK(e == cudaSuccess) << "CUDA_CALL: " << #cudaFunction               \
                              << ", ErrorMessage: " << cudaGetErrorString(e); \
  }

#if defined(CUDA_DEBUG_DEVICE_SYNC)
#define CUDA_DEVICE_SYNC_IF_DEBUG() \
  { CHECK_CUDA_CALL(cudaDeviceSynchronize()); }
#else
#define CUDA_DEVICE_SYNC_IF_DEBUG()
#endif

#if defined(CUDA_DEBUG_STREAM_SYNC)
#define CUDA_STREAM_SYNC_IF_DEBUG(cu_stream) \
  { CHECK_CUDA_CALL(cudaStreamSynchronize(cu_stream)); }
#else
#define CUDA_STREAM_SYNC_IF_DEBUG(...)
#endif

#if defined(CUDA_DEBUG_EVENT_SYNC)
#define CUDA_EVENT_SYNC_IF_DEBUG(e) \
  { CHECK_CUDA_CALL(cudaEventSynchronize(e)); }
#else
#define CUDA_EVENT_SYNC_IF_DEBUG(...)
#endif
