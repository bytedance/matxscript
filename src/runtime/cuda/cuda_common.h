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
#pragma once

#include <cuda_runtime.h>

#include <string>

#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {
namespace cuda {

#define MATXSCRIPT_CUDA_DRIVER_CALL(x)                                  \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      MXLOG(FATAL) << "CUDAError: " #x " failed with error: " << msg;   \
    }                                                                   \
  }

#define MATXSCRIPT_CUDA_CALL(func)                             \
  {                                                            \
    cudaError_t e = (func);                                    \
    MXCHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                  \
  }

// Intentionally ignore a CUDA error
#define MATXSCRIPT_CUDA_IGNORE_ERROR(EXPR)           \
  do {                                               \
    const cudaError_t __err = EXPR;                  \
    if (MATXSCRIPT_UNLIKELY(__err != cudaSuccess)) { \
      cudaError_t error_unused = cudaGetLastError(); \
      (void)error_unused;                            \
    }                                                \
  } while (0)

}  // namespace cuda
}  // namespace runtime
}  // namespace matxscript
