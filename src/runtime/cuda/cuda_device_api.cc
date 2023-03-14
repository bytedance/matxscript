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
#include <cstring>
#include <mutex>

#include <cuda.h>
#include <cuda_runtime.h>

#include "core/device/cuda/cuda_allocator.h"
#include "core/framework/allocator.h"
#include "core/framework/arena.h"
#include "core/framework/bfc_arena.h"

#include "cuda_common.h"
#include "cuda_functions.h"

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/dlpack.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {
namespace cuda {

static void* create_stream(int device_id) {
  MATXSCRIPT_CUDA_CALL(cudaSetDevice(device_id));
  cudaStream_t retval;
  MATXSCRIPT_CUDA_CALL(cudaStreamCreate(&retval));
  return static_cast<MATXScriptStreamHandle>(retval);
}

static void free_stream(int device_id, void* stream) {
  MATXSCRIPT_CUDA_CALL(cudaSetDevice(device_id));
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  MATXSCRIPT_CUDA_CALL(cudaStreamDestroy(cu_stream));
}

// Global stream state and constants
static std::once_flag init_flag;
static int num_gpus = -1;

// Thread-local current streams
static thread_local std::unique_ptr<std::shared_ptr<void>[]> current_streams = nullptr;

// Populates global values.
// Warning: this function must only be called once!
static void initGlobalStreamState() {
  num_gpus = device_count();
}

// Init front-end to ensure initialization only occurs once
static std::unique_ptr<std::shared_ptr<void>[]> createDefaultCUDAStreams() {
  // Inits default streams (once, globally)
  std::call_once(init_flag, initGlobalStreamState);

  int prev_device = current_device();
  // Inits current streams (thread local) to default streams
  auto streams = std::make_unique<std::shared_ptr<void>[]>(num_gpus);
  for (int i = 0; i < num_gpus; ++i) {
    streams[i] =
        std::shared_ptr<void>(create_stream(i), [i](void* stream) { free_stream(i, stream); });
  }

  MATXSCRIPT_CUDA_CALL(cudaSetDevice(prev_device));
  return streams;
}

static const std::unique_ptr<std::shared_ptr<void>[]>& getDefaultCUDAStreams() {
  // Default streams
  static std::unique_ptr<std::shared_ptr<void>[]> default_streams = createDefaultCUDAStreams();
  return default_streams;
}

// Init front-end to ensure initialization only occurs once
static void initCUDAStreamsOnce() {
  // Inits default streams (once, globally)
  std::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams = std::make_unique<std::shared_ptr<void>[]>(num_gpus);
  for (int i = 0; i < num_gpus; ++i) {
    current_streams[i] = getDefaultCUDAStreams()[i];
  }
}

// Helper to verify the GPU index is valid
static inline void check_gpu(int device_index) {
  MATXSCRIPT_ASSERT(device_index >= 0 && device_index < num_gpus);
}

class CUDADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(MATXScriptDevice device) final {
    MATXSCRIPT_CUDA_CALL(cudaSetDevice(device.device_id));
  }

  void GetAttr(MATXScriptDevice device, DeviceAttrKind kind, RTValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        value = (cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device.device_id) ==
                 cudaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device.device_id));
        break;
      }
      case kWarpSize: {
        MATXSCRIPT_CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, device.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, device.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, device.device_id));
        os << value << ".";
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, device.device_id));
        os << value;
        *rv = String(os.str());
        return;
      }
      case kDeviceName: {
        String name(256, '\0');
        MATXSCRIPT_CUDA_DRIVER_CALL(cuDeviceGetName(&name[0], name.size(), device.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, device.device_id));
        break;
      }
      case kMultiProcessorCount: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&dims[0], cudaDevAttrMaxBlockDimX, device.device_id));
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&dims[1], cudaDevAttrMaxBlockDimY, device.device_id));
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&dims[2], cudaDevAttrMaxBlockDimZ, device.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = String(ss.str());
        return;
      }
      case kMaxRegistersPerBlock: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, device.device_id));
        break;
      }
      case kGcnArch:
        return;
      case kApiVersion: {
        *rv = CUDA_VERSION;
        return;
      }
    }
    *rv = value;
  }

  void* Alloc(MATXScriptDevice device, size_t nbytes) final {
    void* ret;
    if (device.device_type == kDLCUDAHost) {
      if (static_cast<size_t>(device.device_id) >= cudaPinnedBFCAllocators.size() ||
          cudaPinnedBFCAllocators[device.device_id] == nullptr) {
        InitPinAllocator(device);
      }
      ret = cudaPinnedBFCAllocators[device.device_id]->Alloc(nbytes);
    } else {
      if (static_cast<size_t>(device.device_id) >= cudaBFCAllocators.size() ||
          cudaBFCAllocators[device.device_id] == nullptr) {
        InitCudaAllocator(device);
      }
      ret = cudaBFCAllocators[device.device_id]->Alloc(nbytes);
    }
    return ret;
  }

  void* Alloc(MATXScriptDevice device,
              size_t nbytes,
              size_t alignment,
              DLDataType type_hint) final {
    MXCHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    return Alloc(device, nbytes);
  }

  void* AllocRaw(MATXScriptDevice device,
                 size_t nbytes,
                 size_t alignment,
                 DLDataType type_hint) final {
    MXCHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    if (device.device_type == kDLCUDAHost) {
      MATXSCRIPT_CUDA_CALL(cudaMallocHost(&ret, nbytes));
    } else {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(device.device_id));
      MATXSCRIPT_CUDA_CALL(cudaMalloc(&ret, nbytes));
    }
    return ret;
  }

  void Free(MATXScriptDevice device, void* ptr) final {
    if (device.device_type == kDLCUDAHost) {
      MXCHECK(static_cast<size_t>(device.device_id) < cudaPinnedBFCAllocators.size() &&
              cudaPinnedBFCAllocators[device.device_id] != nullptr);
      cudaPinnedBFCAllocators[device.device_id]->Free(ptr);
    } else {
      MXCHECK(static_cast<size_t>(device.device_id) < cudaBFCAllocators.size() &&
              cudaBFCAllocators[device.device_id] != nullptr);
      cudaBFCAllocators[device.device_id]->Free(ptr);
    }
  }

  void FreeRaw(MATXScriptDevice device, void* ptr) final {
    if (device.device_type == kDLCUDAHost) {
      MATXSCRIPT_CUDA_CALL(cudaFreeHost(ptr));
    } else {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(device.device_id));
      MATXSCRIPT_CUDA_CALL(cudaFree(ptr));
    }
  }

  ~CUDADeviceAPI() {
  }

 protected:
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      MATXScriptDevice device_from,
                      MATXScriptDevice device_to,
                      DLDataType type_hint,
                      MATXScriptStreamHandle stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (device_from.device_type == kDLCUDAHost) {
      device_from.device_type = kDLCPU;
    }

    if (device_to.device_type == kDLCUDAHost) {
      device_to.device_type = kDLCPU;
    }

    // In case there is a copy from host mem to host mem */
    if (device_to.device_type == kDLCPU && device_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (device_from.device_type == kDLCUDA && device_to.device_type == kDLCUDA) {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(device_from.device_id));
      if (device_from.device_id == device_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, device_to.device_id, from, device_from.device_id, size, cu_stream);
      }
    } else if (device_from.device_type == kDLCUDA && device_to.device_type == kDLCPU) {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(device_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (device_from.device_type == kDLCPU && device_to.device_type == kDLCUDA) {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(device_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      MXLOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  MATXScriptStreamHandle CreateStream(MATXScriptDevice device) {
    return create_stream(device.device_id);
  }

  void FreeStream(MATXScriptDevice device, MATXScriptStreamHandle stream) {
    return free_stream(device.device_id, stream);
  }

  MATXScriptStreamHandle GetDefaultStream(MATXScriptDevice device) final {
    initCUDAStreamsOnce();
    auto device_index = device.device_id;
    if (device_index == -1) {
      device_index = current_device();
    }
    check_gpu(device_index);
    return getDefaultCUDAStreams()[device_index].get();
  }

  MATXScriptStreamHandle GetCurrentThreadStream(MATXScriptDevice device) final {
    initCUDAStreamsOnce();
    auto device_index = device.device_id;
    if (device_index == -1) {
      device_index = current_device();
    }
    check_gpu(device_index);
    return current_streams[device_index].get();
  }

  std::shared_ptr<void> GetSharedCurrentThreadStream(MATXScriptDevice device) final {
    initCUDAStreamsOnce();
    auto device_index = device.device_id;
    if (device_index == -1) {
      device_index = current_device();
    }
    check_gpu(device_index);
    return current_streams[device_index];
  }

  void SetCurrentThreadStream(MATXScriptDevice device, std::shared_ptr<void> stream) final {
    initCUDAStreamsOnce();
    current_streams[device.device_id] = std::move(stream);
  }

  void StreamSync(MATXScriptDevice device, MATXScriptStreamHandle stream) final {
    MATXSCRIPT_CUDA_CALL(cudaSetDevice(device.device_id));
    MATXSCRIPT_CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void CreateEventSync(MATXScriptStreamHandle stream) final {
    cudaEvent_t finish_event;
    MATXSCRIPT_CUDA_CALL(cudaEventCreate(&finish_event))
    MATXSCRIPT_CUDA_CALL(cudaEventRecord(finish_event, static_cast<cudaStream_t>(stream)))
    MATXSCRIPT_CUDA_CALL(cudaEventSynchronize(finish_event));
    MATXSCRIPT_CUDA_CALL(cudaEventDestroy(finish_event));
  }

  void SyncStreamFromTo(MATXScriptDevice device,
                        MATXScriptStreamHandle event_src,
                        MATXScriptStreamHandle event_dst) {
    MATXSCRIPT_CUDA_CALL(cudaSetDevice(device.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    MATXSCRIPT_CUDA_CALL(cudaEventCreate(&evt));
    MATXSCRIPT_CUDA_CALL(cudaEventRecord(evt, src_stream));
    MATXSCRIPT_CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
    MATXSCRIPT_CUDA_CALL(cudaEventDestroy(evt));
  }

  static CUDADeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CUDADeviceAPI();
    return inst;
  }

 private:
  MATXScriptStreamHandle GetStream(MATXScriptDevice device,
                                   std::vector<MATXScriptStreamHandle>& pool) {
    std::lock_guard<std::mutex> lock(streamAllocMutex_);
    if (static_cast<size_t>(device.device_id) >= pool.size()) {
      pool.resize(device.device_id + 1, nullptr);
    }
    if (pool[device.device_id] == nullptr) {
      pool[device.device_id] = CreateStream(device);
    }
    return pool[device.device_id];
  }

  static void GPUCopy(
      const void* from, void* to, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    if (stream != nullptr) {
      MATXSCRIPT_CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      MATXSCRIPT_CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }

  void InitCudaAllocator(MATXScriptDevice device) {
    std::lock_guard<std::mutex> lock(cudaAllocMutex_);
    if (static_cast<size_t>(device.device_id) >= cudaBFCAllocators.size()) {
      cudaBFCAllocators.resize(device.device_id + 1, nullptr);
    }
    if (cudaBFCAllocators[device.device_id] == nullptr) {
      cudaBFCAllocators[device.device_id] = new brt::BFCArena(
          std::unique_ptr<brt::IAllocator>(new brt::CUDAAllocator(device.device_id, "cuda")),
          1ULL << 35);
    }
  }

  void InitPinAllocator(MATXScriptDevice device) {
    std::lock_guard<std::mutex> lock(pinAllocMutex_);
    if (static_cast<size_t>(device.device_id) >= cudaPinnedBFCAllocators.size()) {
      cudaPinnedBFCAllocators.resize(device.device_id + 1, nullptr);
    }
    if (cudaPinnedBFCAllocators[device.device_id] == nullptr) {
      cudaPinnedBFCAllocators[device.device_id] =
          new brt::BFCArena(std::unique_ptr<brt::IAllocator>(
                                new brt::CUDAPinnedAllocator(device.device_id, "cuda_pin")),
                            1ULL << 33);
    }
  }

  std::vector<brt::BFCArena*> cudaBFCAllocators;
  std::vector<brt::BFCArena*> cudaPinnedBFCAllocators;
  std::mutex cudaAllocMutex_;
  std::mutex pinAllocMutex_;
  std::mutex streamAllocMutex_;
};

MATXSCRIPT_REGISTER_GLOBAL("device_api.gpu").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  return static_cast<void*>(ptr);
});

MATXSCRIPT_REGISTER_GLOBAL("device_api.cuda").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  return static_cast<void*>(ptr);
});

MATXSCRIPT_REGISTER_GLOBAL("device_api.cpu_pinned").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  return static_cast<void*>(ptr);
});

MATXSCRIPT_REGISTER_GLOBAL("device_api.cuda_host").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  return static_cast<void*>(ptr);
});

}  // namespace cuda
}  // namespace runtime
}  // namespace matxscript
