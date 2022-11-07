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
 * \file cpu_device_api.cc
 */
#include <matxscript/runtime/device_api.h>

#include <cstdlib>
#include <cstring>

#include "core/framework/allocator.h"
#include "core/framework/arena.h"
#include "core/framework/bfc_arena.h"

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/dlpack.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace matxscript {
namespace runtime {

class CPUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(MATXScriptContext ctx) final {
  }
  void GetAttr(MATXScriptContext ctx, DeviceAttrKind kind, RTValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* Alloc(MATXScriptContext ctx, size_t nbytes) final {
    MXCHECK(cpuBFCAllocator != nullptr);
    void* ptr = cpuBFCAllocator->Alloc(nbytes);
    return ptr;
  }

  void* Alloc(MATXScriptContext ctx, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    return Alloc(ctx, nbytes);
  }

  void* AllocRaw(MATXScriptContext ctx,
                 size_t nbytes,
                 size_t alignment,
                 DLDataType type_hint) final {
    void* ptr;
#if _MSC_VER
    ptr = _aligned_malloc(nbytes, alignment);
    if (ptr == nullptr)
      throw std::bad_alloc();
#elif defined(__ANDROID__) && __ANDROID_API__ < 17
    ptr = memalign(alignment, nbytes);
    if (ptr == nullptr)
      throw std::bad_alloc();
#else
    // posix_memalign is available in android ndk since __ANDROID_API__ >= 17
    int ret = posix_memalign(&ptr, alignment, nbytes);
    if (ret != 0)
      throw std::bad_alloc();
#endif
    return ptr;
  }

  void Free(MATXScriptContext ctx, void* ptr) final {
    MXCHECK(cpuBFCAllocator != nullptr);
    cpuBFCAllocator->Free(ptr);
  }

  void FreeRaw(MATXScriptContext ctx, void* ptr) final {
#if _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      MATXScriptContext ctx_from,
                      MATXScriptContext ctx_to,
                      DLDataType type_hint,
                      MATXScriptStreamHandle stream) final {
    std::memcpy(
        static_cast<char*>(to) + to_offset, static_cast<const char*>(from) + from_offset, size);
  }

  MATXScriptStreamHandle CreateStream(MATXScriptContext ctx) final {
    return nullptr;
  }

  void FreeStream(MATXScriptContext ctx, MATXScriptStreamHandle stream) final {
  }

  MATXScriptStreamHandle GetDefaultStream(MATXScriptContext ctx) final {
    return nullptr;
  }

  MATXScriptStreamHandle GetCurrentThreadStream(MATXScriptContext ctx) final {
    return nullptr;
  }

  std::shared_ptr<void> GetSharedCurrentThreadStream(MATXScriptContext ctx) final {
    return nullptr;
  }

  void SetCurrentThreadStream(MATXScriptContext ctx, std::shared_ptr<void> stream) final {
  }

  void StreamSync(MATXScriptContext ctx, MATXScriptStreamHandle stream) final {
  }

  void CreateEventSync(MATXScriptStreamHandle stream) final {
  }

  void SyncStreamFromTo(MATXScriptContext ctx,
                        MATXScriptStreamHandle event_src,
                        MATXScriptStreamHandle event_dst) final {
  }

  static CPUDeviceAPI* Global() {
    // NOTE: explicitly use new to avoid exit-time destruction of global state
    // Global state will be recycled by OS as the process exits.
    static auto* inst = new CPUDeviceAPI();
    return inst;
  }

 private:
  brt::BFCArena* cpuBFCAllocator =
      new brt::BFCArena(std::unique_ptr<brt::IAllocator>(new brt::CPUAllocator()), 1ULL << 32);
  ;
};

struct CPUGlobalEntry {
  CPUGlobalEntry() {
    CPUDeviceAPI::Global();
  }
};

MATXSCRIPT_REGISTER_GLOBAL("device_api.cpu").set_body([](PyArgs args) -> RTValue {
  DeviceAPI* ptr = CPUDeviceAPI::Global();
  return static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace matxscript
