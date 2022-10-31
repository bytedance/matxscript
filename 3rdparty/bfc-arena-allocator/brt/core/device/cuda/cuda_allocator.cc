// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modifications Copyright (c) ByteDance.

#include "core/device/cuda/cuda_allocator.h"

#include "core/common/common.h"

#include <cuda_runtime.h>

#define CUDA_CALL(func)                                         \
  {                                                             \
    cudaError_t e = (func);                                     \
    if (!(e == cudaSuccess || e == cudaErrorCudartUnloading)) { \
      std::stringstream fmsg;                                   \
      fmsg << "CUDA: " << cudaGetErrorString(e);                \
      throw ::brt::BrtException(BRT_WHERE, fmsg.str());         \
    }                                                           \
  }

namespace brt {

void CUDAAllocator::CheckDevice(bool throw_when_fail) const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    BRT_ENFORCE(current_device == Info().id);
  }
#endif

  BRT_UNUSED_PARAMETER(throw_when_fail);
}

void CUDAAllocator::SetDevice(bool throw_when_fail) const {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    int allocator_device_id = Info().id;
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }

  BRT_UNUSED_PARAMETER(throw_when_fail);
}

void* CUDAAllocator::Alloc(size_t size) {
  SetDevice(true);
  CheckDevice(true);
  void* p = nullptr;
  if (size > 0) {
    // BFCArena was updated recently to handle the exception and adjust the request size
    CUDA_CALL(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  SetDevice(false);
  CheckDevice(false);  // ignore CUDA failure when free
  cudaFree(p);         // do not throw error since it's OK for cudaFree to fail during shutdown
}

void* CUDAExternalAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    p = alloc_(size);
    // review(codemzs): BRT_ENFORCE does not seem appropiate.
    BRT_ENFORCE(p != nullptr);
  }

  return p;
}

void CUDAExternalAllocator::Free(void* p) {
  free_(p);
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    cudaMallocHost((void**)&p, size);
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  cudaFreeHost(p);
}

}  // namespace brt
