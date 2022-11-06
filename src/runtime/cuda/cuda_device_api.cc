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

#include "cuda_common.h"

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/dlpack.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

class CUDADeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(MATXScriptContext ctx) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
  }
  void GetAttr(MATXScriptContext ctx, DeviceAttrKind kind, RTValue* rv) final {
    int value = 0;
    switch (kind) {
      case kExist:
        value = (cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id) ==
                 cudaSuccess);
        break;
      case kMaxThreadsPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, ctx.device_id));
        break;
      }
      case kWarpSize: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, ctx.device_id));
        break;
      }
      case kMaxSharedMemoryPerBlock: {
        CUDA_CALL(
            cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, ctx.device_id));
        break;
      }
      case kComputeVersion: {
        std::ostringstream os;
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, ctx.device_id));
        os << value << ".";
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, ctx.device_id));
        os << value;
        *rv = String(os.str());
        return;
      }
      case kDeviceName: {
        String name(256, '\0');
        CUDA_DRIVER_CALL(cuDeviceGetName(&name[0], name.size(), ctx.device_id));
        name.resize(strlen(name.c_str()));
        *rv = std::move(name);
        return;
      }
      case kMaxClockRate: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, ctx.device_id));
        break;
      }
      case kMultiProcessorCount: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, ctx.device_id));
        break;
      }
      case kMaxThreadDimensions: {
        int dims[3];
        CUDA_CALL(cudaDeviceGetAttribute(&dims[0], cudaDevAttrMaxBlockDimX, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(&dims[1], cudaDevAttrMaxBlockDimY, ctx.device_id));
        CUDA_CALL(cudaDeviceGetAttribute(&dims[2], cudaDevAttrMaxBlockDimZ, ctx.device_id));

        std::stringstream ss;  // use json string to return multiple int values;
        ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
        *rv = String(ss.str());
        return;
      }
      case kMaxRegistersPerBlock: {
        CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, ctx.device_id));
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
      CUDA_CALL(cudaMallocHost(&ret, nbytes));
    } else {
      CUDA_CALL(cudaSetDevice(ctx.device_id));
      CUDA_CALL(cudaMalloc(&ret, nbytes));
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
      CUDA_CALL(cudaFreeHost(ptr));
    } else {
      CUDA_CALL(cudaSetDevice(ctx.device_id));
      CUDA_CALL(cudaFree(ptr));
    }
  }

  ~CUDADeviceAPI() {
    std::lock_guard<std::mutex> lock(streamAllocMutex_);
    for (int i = 0; i < cudaDefaultStreams.size(); ++i) {
      if (cudaDefaultStreams[i] != nullptr) {
        FreeStream(MATXScriptContext{kDLGPU, i}, cudaDefaultStreams[i]);
      }
    }
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
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      if (ctx_from.device_id == ctx_to.device_id) {
        GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
      } else {
        cudaMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id, size, cu_stream);
      }
    } else if (ctx_from.device_type == kDLGPU && ctx_to.device_type == kDLCPU) {
      CUDA_CALL(cudaSetDevice(ctx_from.device_id));
      GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLGPU) {
      CUDA_CALL(cudaSetDevice(ctx_to.device_id));
      GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
      MXLOG(FATAL) << "expect copy from/to GPU or between GPU";
    }
  }

 public:
  MATXScriptStreamHandle CreateStream(MATXScriptContext ctx) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t retval;
    CUDA_CALL(cudaStreamCreate(&retval));
    return static_cast<MATXScriptStreamHandle>(retval);
  }

  void FreeStream(MATXScriptContext ctx, MATXScriptStreamHandle stream) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    CUDA_CALL(cudaStreamDestroy(cu_stream));
  }

  void SyncStreamFromTo(MATXScriptContext ctx,
                        MATXScriptStreamHandle event_src,
                        MATXScriptStreamHandle event_dst) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStream_t src_stream = static_cast<cudaStream_t>(event_src);
    cudaStream_t dst_stream = static_cast<cudaStream_t>(event_dst);
    cudaEvent_t evt;
    CUDA_CALL(cudaEventCreate(&evt));
    CUDA_CALL(cudaEventRecord(evt, src_stream));
    CUDA_CALL(cudaStreamWaitEvent(dst_stream, evt, 0));
    CUDA_CALL(cudaEventDestroy(evt));
  }

  void StreamSync(MATXScriptContext ctx, MATXScriptStreamHandle stream) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void DefaultComputeStreamSync(MATXScriptContext ctx) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    MATXScriptStreamHandle stream = GetDefaultComputeStream(ctx);
    CUDA_CALL(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
  }

  void SetStreamForCurrentThread(MATXScriptContext ctx, std::shared_ptr<void> stream) final {
    CUDAGlobalEntry::thread_local_stream = stream;
  }

  void ResetStreamForCurrentThread(MATXScriptContext ctx) final {
    CUDAGlobalEntry::thread_local_stream.reset();
  }

  static CUDADeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CUDADeviceAPI();
    return inst;
  }

  MATXScriptStreamHandle GetDefaultComputeStream(MATXScriptContext ctx) {
    if (CUDAGlobalEntry::thread_local_stream != nullptr) {
      return *std::static_pointer_cast<MATXScriptStreamHandle>(
          CUDAGlobalEntry::thread_local_stream);
    }
    return GetStream(ctx, cudaDefaultStreams);
  }

  MATXScriptStreamHandle GetDefaultIOStreamH2D(MATXScriptContext ctx) {
    if (CUDAGlobalEntry::thread_local_stream != nullptr) {
      return *std::static_pointer_cast<MATXScriptStreamHandle>(
          CUDAGlobalEntry::thread_local_stream);
    }
    return GetStream(ctx, cudaDefaultH2DStreams);
  }

  MATXScriptStreamHandle GetDefaultIOStreamD2H(MATXScriptContext ctx) {
    if (CUDAGlobalEntry::thread_local_stream != nullptr) {
      return *std::static_pointer_cast<MATXScriptStreamHandle>(
          CUDAGlobalEntry::thread_local_stream);
    }
    return GetStream(ctx, cudaDefaultD2HStreams);
  }

  void CreateEventSync(MATXScriptStreamHandle stream) final {
    cudaEvent_t finish_event;
    CUDA_CALL(cudaEventCreate(&finish_event))
    CUDA_CALL(cudaEventRecord(finish_event, static_cast<cudaStream_t>(stream)))
    CUDA_CALL(cudaEventSynchronize(finish_event));
    CUDA_CALL(cudaEventDestroy(finish_event));
  }

  MATXScriptStreamHandle GetCopyFromStream(MATXScriptContext ctx,
                                           MATXScriptContext from_ctx) final {
    if (from_ctx.device_type == kDLCPU) {
      return GetDefaultIOStreamH2D(ctx);
    }
    return nullptr;
  }

  MATXScriptStreamHandle GetCopyToStream(MATXScriptContext ctx, MATXScriptContext to_ctx) final {
    if (to_ctx.device_type == kDLCPU) {
      return GetDefaultIOStreamD2H(ctx);
    }
    return nullptr;
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
      CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
      CUDA_CALL(cudaMemcpy(to, from, size, kind));
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
  std::vector<MATXScriptStreamHandle> cudaDefaultStreams;
  std::vector<MATXScriptStreamHandle> cudaDefaultH2DStreams;
  std::vector<MATXScriptStreamHandle> cudaDefaultD2HStreams;
  std::mutex cudaAllocMutex_;
  std::mutex pinAllocMutex_;
  std::mutex streamAllocMutex_;
};

CUDAGlobalEntry::CUDAGlobalEntry() {
  CUDADeviceAPI::Global();
}

CUDAGlobalEntry* CUDAGlobalEntry::Get() {
  static CUDAGlobalEntry* cuda_entry = new CUDAGlobalEntry();
  return cuda_entry;
}

MATXSCRIPT_REGISTER_GLOBAL("device_api.gpu").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  return static_cast<void*>(ptr);
});

MATXSCRIPT_REGISTER_GLOBAL("device_api.cpu_pinned").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CUDADeviceAPI::Global();
  return static_cast<void*>(ptr);
});

}  // namespace runtime
}  // namespace matxscript
