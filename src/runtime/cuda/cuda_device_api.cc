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

  // Inits current streams (thread local) to default streams
  auto streams = std::make_unique<std::shared_ptr<void>[]>(num_gpus);
  for (int i = 0; i < num_gpus; ++i) {
    streams[i] =
        std::shared_ptr<void>(create_stream(i), [i](void* stream) { free_stream(i, stream); });
  }
  return streams;
}

// Default streams
static std::unique_ptr<std::shared_ptr<void>[]> default_streams = createDefaultCUDAStreams();

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
    current_streams[i] = default_streams[i];
  }
}

// Helper to verify the GPU index is valid
static inline void check_gpu(int device_index) {
  MATXSCRIPT_ASSERT(device_index >= 0 && device_index < num_gpus);
}

class CUDADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(MATXScriptContext ctx) final {
    MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx.device_id));
  }

  void GetAttr(MATXScriptContext ctx, DeviceAttrKind kind, RTValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        value = (cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id) ==
                 cudaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        MATXSCRIPT_CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, ctx.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = String(os.str());
        return;
      }
      case kDeviceName: {
        String name(256, '\0');
        MATXSCRIPT_CUDA_DRIVER_CALL(cuDeviceGetName(&name[0], name.size(), ctx.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        MATXSCRIPT_CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, ctx.device_id));
        break;
      }
      case kMultiProcessorCount: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, ctx.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&dims[0], cudaDevAttrMaxBlockDimX, ctx.device_id));
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&dims[1], cudaDevAttrMaxBlockDimY, ctx.device_id));
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&dims[2], cudaDevAttrMaxBlockDimZ, ctx.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = String(ss.str());
        return;
      }
      case kMaxRegistersPerBlock: {
        MATXSCRIPT_CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, ctx.device_id));
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

  void* Alloc(MATXScriptContext ctx, size_t nbytes) final {
    void* ret;
    if (ctx.device_type == kDLCPUPinned) {
      if (static_cast<size_t>(ctx.device_id) >= cudaPinnedBFCAllocators.size() ||
          cudaPinnedBFCAllocators[ctx.device_id] == nullptr) {
        InitPinAllocator(ctx);
      }
      ret = cudaPinnedBFCAllocators[ctx.device_id]->Alloc(nbytes);
    } else {
      if (static_cast<size_t>(ctx.device_id) >= cudaBFCAllocators.size() ||
          cudaBFCAllocators[ctx.device_id] == nullptr) {
        InitCudaAllocator(ctx);
      }
      ret = cudaBFCAllocators[ctx.device_id]->Alloc(nbytes);
    }
    return ret;
  }

  void* Alloc(MATXScriptContext ctx, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    MXCHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    return Alloc(ctx, nbytes);
  }

  void* AllocRaw(MATXScriptContext ctx,
                 size_t nbytes,
                 size_t alignment,
                 DLDataType type_hint) final {
    MXCHECK_EQ(256 % alignment, 0U) << "CUDA space is aligned at 256 bytes";
    void* ret;
    if (ctx.device_type == kDLCPUPinned) {
      MATXSCRIPT_CUDA_CALL(cudaMallocHost(&ret, nbytes));
    } else {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx.device_id));
      MATXSCRIPT_CUDA_CALL(cudaMalloc(&ret, nbytes));
    }
    return ret;
  }

  void Free(MATXScriptContext ctx, void* ptr) final {
    if (ctx.device_type == kDLCPUPinned) {
      MXCHECK(static_cast<size_t>(ctx.device_id) < cudaPinnedBFCAllocators.size() &&
              cudaPinnedBFCAllocators[ctx.device_id] != nullptr);
      cudaPinnedBFCAllocators[ctx.device_id]->Free(ptr);
    } else {
      MXCHECK(static_cast<size_t>(ctx.device_id) < cudaBFCAllocators.size() &&
              cudaBFCAllocators[ctx.device_id] != nullptr);
      cudaBFCAllocators[ctx.device_id]->Free(ptr);
    }
  }

  void FreeRaw(MATXScriptContext ctx, void* ptr) final {
    if (ctx.device_type == kDLCPUPinned) {
      MATXSCRIPT_CUDA_CALL(cudaFreeHost(ptr));
    } else {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx.device_id));
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
                      MATXScriptContext ctx_from,
                      MATXScriptContext ctx_to,
                      DLDataType type_hint,
                      MATXScriptStreamHandle stream) final {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    from = static_cast<const char*>(from) + from_offset;
    to = static_cast<char*>(to) + to_offset;

    if (ctx_from.device_type == kDLCPUPinned) {
      ctx_from.device_type = kDLCPU;
    }

    if (ctx_to.device_type == kDLCPUPinned) {
      ctx_to.device_type = kDLCPU;
    }

    // In case there is a copy from host mem to host mem */
    if (ctx_to.device_type == kDLCPU && ctx_from.device_type == kDLCPU) {
      memcpy(to, from, size);
      return;
    }

    if (ctx_from.device_type == kDLGPU && ctx_to.device_type == kDLGPU) {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id, size, cu_stream);
      }
    } else if (ctx_from.device_type == kDLGPU && ctx_to.device_type == kDLCPU) {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLGPU) {
      MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      MXLOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  MATXScriptStreamHandle CreateStream(MATXScriptContext ctx) {
    return create_stream(ctx.device_id);
  }

  void FreeStream(MATXScriptContext ctx, MATXScriptStreamHandle stream) {
    return free_stream(ctx.device_id, stream);
  }

  MATXScriptStreamHandle GetDefaultStream(MATXScriptContext ctx) final {
    initCUDAStreamsOnce();
    auto device_index = ctx.device_id;
    if (device_index == -1) {
      device_index = current_device();
    }
    check_gpu(device_index);
    return default_streams[device_index].get();
  }

  MATXScriptStreamHandle GetCurrentThreadStream(MATXScriptContext ctx) final {
    initCUDAStreamsOnce();
    auto device_index = ctx.device_id;
    if (device_index == -1) {
      device_index = current_device();
    }
    check_gpu(device_index);
    return current_streams[device_index].get();
  }

  std::shared_ptr<void> GetSharedCurrentThreadStream(MATXScriptContext ctx) final {
    initCUDAStreamsOnce();
    auto device_index = ctx.device_id;
    if (device_index == -1) {
      device_index = current_device();
    }
    check_gpu(device_index);
    return current_streams[device_index];
  }

  void SetCurrentThreadStream(MATXScriptContext ctx, std::shared_ptr<void> stream) final {
    initCUDAStreamsOnce();
    current_streams[ctx.device_id] = std::move(stream);
  }

  void StreamSync(MATXScriptContext ctx, MATXScriptStreamHandle stream) final {
    MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx.device_id));
    MATXSCRIPT_CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void CreateEventSync(MATXScriptStreamHandle stream) final {
    cudaEvent_t finish_event;
    MATXSCRIPT_CUDA_CALL(cudaEventCreate(&finish_event))
    MATXSCRIPT_CUDA_CALL(cudaEventRecord(finish_event, static_cast<cudaStream_t>(stream)))
    MATXSCRIPT_CUDA_CALL(cudaEventSynchronize(finish_event));
    MATXSCRIPT_CUDA_CALL(cudaEventDestroy(finish_event));
  }

  void SyncStreamFromTo(MATXScriptContext ctx,
                        MATXScriptStreamHandle event_src,
                        MATXScriptStreamHandle event_dst) {
    MATXSCRIPT_CUDA_CALL(cudaSetDevice(ctx.device_id));
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
  MATXScriptStreamHandle GetStream(MATXScriptContext ctx,
                                   std::vector<MATXScriptStreamHandle>& pool) {
    std::lock_guard<std::mutex> lock(streamAllocMutex_);
    if (static_cast<size_t>(ctx.device_id) >= pool.size()) {
      pool.resize(ctx.device_id + 1, nullptr);
    }
    if (pool[ctx.device_id] == nullptr) {
      pool[ctx.device_id] = CreateStream(ctx);
    }
    return pool[ctx.device_id];
  }

  static void GPUCopy(
      const void* from, void* to, size_t size, cudaMemcpyKind kind, cudaStream_t stream) {
    if (stream != nullptr) {
      MATXSCRIPT_CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      MATXSCRIPT_CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
  }

  void InitCudaAllocator(MATXScriptContext ctx) {
    std::lock_guard<std::mutex> lock(cudaAllocMutex_);
    if (static_cast<size_t>(ctx.device_id) >= cudaBFCAllocators.size()) {
      cudaBFCAllocators.resize(ctx.device_id + 1, nullptr);
    }
    if (cudaBFCAllocators[ctx.device_id] == nullptr) {
      cudaBFCAllocators[ctx.device_id] = new brt::BFCArena(
          std::unique_ptr<brt::IAllocator>(new brt::CUDAAllocator(ctx.device_id, "cuda")),
          1ULL << 35);
    }
  }

  void InitPinAllocator(MATXScriptContext ctx) {
    std::lock_guard<std::mutex> lock(pinAllocMutex_);
    if (static_cast<size_t>(ctx.device_id) >= cudaPinnedBFCAllocators.size()) {
      cudaPinnedBFCAllocators.resize(ctx.device_id + 1, nullptr);
    }
    if (cudaPinnedBFCAllocators[ctx.device_id] == nullptr) {
      cudaPinnedBFCAllocators[ctx.device_id] = new brt::BFCArena(
          std::unique_ptr<brt::IAllocator>(new brt::CUDAPinnedAllocator(ctx.device_id, "cuda_pin")),
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

MATXSCRIPT_REGISTER_GLOBAL("device_api.cpu_pinned").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  return static_cast<void*>(ptr);
});

}  // namespace cuda
}  // namespace runtime
}  // namespace matxscript
