// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from PyTorch.
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
#include "cuda_functions.h"

#include "cuda_common.h"

#include <limits>

namespace matxscript {
namespace runtime {
namespace cuda {

namespace {
// returns -1 on failure
int32_t driver_version() {
  int driver_version = -1;
  MATXSCRIPT_CUDA_IGNORE_ERROR(cudaDriverGetVersion(&driver_version));
  return driver_version;
}

int device_count_impl(bool fail_if_no_driver) {
  int count;
  auto err = cudaGetDeviceCount(&count);
  if (err == cudaSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much CUDA
  // code in this regime.)
  cudaError_t last_err MATXSCRIPT_ATTRIBUTE_UNUSED = cudaGetLastError();
  switch (err) {
    case cudaErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case cudaErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // No CUDA driver means no devices
          count = 0;
          break;
        }
        MXTHROW << "Found no NVIDIA driver on your system. Please check that you "
                   "have an NVIDIA GPU and installed a driver from "
                   "http://www.nvidia.com/Download/index.aspx";
      } else {
        MXTHROW << "The NVIDIA driver on your system is too old (found version " << version
                << "). Please update your GPU driver by downloading and installing "
                   "a new version from the URL: "
                   "http://www.nvidia.com/Download/index.aspx Alternatively, go to: "
                   "https://pytorch.org to install a PyTorch version that has been "
                   "compiled with your version of the CUDA driver.";
      }
    } break;
    case cudaErrorInitializationError:
      MXTHROW << "CUDA driver initialization failed, you might not "
                 "have a CUDA gpu.";
      break;
    case cudaErrorUnknown:
      MXTHROW << "CUDA unknown error - this may be due to an "
                 "incorrectly set up environment, e.g. changing env "
                 "variable CUDA_VISIBLE_DEVICES after program start. "
                 "Setting the available devices to be zero.";
      break;
#if MATXSCRIPT_SANITIZE_ADDRESS
    case cudaErrorMemoryAllocation:
      // In ASAN mode, we know that a cudaErrorMemoryAllocation error will
      // pop up if compiled with NVCC (clang-cuda is fine)
      MXTHROW << "Got 'out of memory' error while trying to initialize CUDA. "
                 "CUDA with nvcc does not work well with ASAN and it's probably "
                 "the reason. We will simply shut down CUDA support. If you "
                 "would like to use GPUs, turn off ASAN.";
      break;
#endif  // MATXSCRIPT_SANITIZE_ADDRESS
    default:
      MXTHROW << "Unexpected error from cudaGetDeviceCount(). Did you run "
                 "some cuda functions before calling NumCudaDevices() "
                 "that might have already set an error? Error "
              << err << ": " << cudaGetErrorString(err);
  }
  return count;
}
}  // namespace

int device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      if (result <= std::numeric_limits<int>::max()) {
        MXTHROW << "Too many CUDA devices, DeviceIndex overflowed";
      }
      return result;
    } catch (const std::exception& ex) {
      // We don't want to fail, but still log the warning
      MXLOG(WARNING) << "CUDA initialization: " << ex.what();
      return 0;
    }
  }();
  return static_cast<int>(count);
}

int device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // Zero gpus doesn't produce a warning in `device_count` but we fail here
  MXCHECK(count) << "No CUDA GPUs are available";
  return static_cast<int>(count);
}

int current_device() {
  int cur_device;
  MATXSCRIPT_CUDA_CALL(cudaGetDevice(&cur_device));
  return static_cast<int>(cur_device);
}

void set_device(int device) {
  MATXSCRIPT_CUDA_CALL(cudaSetDevice(static_cast<int>(device)));
}

}  // namespace cuda
}  // namespace runtime
}  // namespace matxscript
