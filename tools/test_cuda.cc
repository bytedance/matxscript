// Copyright 2022 ByteDance Ltd. and/or its affiliates.
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <sstream>

#define CUDA_CALL(func)                                         \
  {                                                             \
    cudaError_t e = (func);                                     \
    if (!(e == cudaSuccess || e == cudaErrorCudartUnloading)) { \
      std::stringstream fmsg;                                   \
      fmsg << "CUDA: " << cudaGetErrorString(e);                \
      throw std::runtime_error(fmsg.str());                     \
    }                                                           \
  }

void SetDevice(int allocator_device_id) {
  int current_device;
  auto cuda_err = cudaGetDevice(&current_device);
  if (cuda_err == cudaSuccess) {
    if (current_device != allocator_device_id) {
      cuda_err = cudaSetDevice(allocator_device_id);
    }
  }
}

void* Alloc(size_t size, int allocator_device_id) {
  SetDevice(allocator_device_id);
  void* p = nullptr;
  if (size > 0) {
    // BFCArena was updated recently to handle the exception and adjust the request size
    CUDA_CALL(cudaMalloc((void**)&p, size));
  }
  return p;
}

void Free(void* p, int allocator_device_id) {
  SetDevice(allocator_device_id);
  CUDA_CALL(cudaFree(p));  // do not throw error since it's OK for cudaFree to fail during shutdown
}

int main(int argc, char* argv[]) {
  int dev = 0;
  int size = 1024;
  if (argc == 2) {
    dev = atoi(argv[1]);
  } else if (argc == 3) {
    size = atoi(argv[2]);
  }
  SetDevice(dev);
  void* p = brt::Alloc(size, dev);
  std::cout << "size: " << size << ", addr: " << p << std::endl;
  Free(p, dev);
  return 0;
}
