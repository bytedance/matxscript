// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modifications Copyright (c) ByteDance.

#pragma once

#include "core/framework/allocator.h"

namespace brt {

class CUDAAllocator : public IAllocator {
 public:
  CUDAAllocator(int device_id, const char* name)
      : IAllocator(BrtMemoryInfo(
            name, BrtAllocatorType::BrtDeviceAllocator, device_id, BrtMemTypeDefault)) {
  }
  void* Alloc(size_t size) override;
  void Free(void* p) override;
  void SetDevice(bool throw_when_fail) const override;

 private:
  void CheckDevice(bool throw_when_fail) const;
};

class CUDAExternalAllocator : public CUDAAllocator {
  typedef void* (*ExternalAlloc)(size_t size);
  typedef void (*ExternalFree)(void* p);

 public:
  CUDAExternalAllocator(int device_id, const char* name, void* alloc, void* free)
      : CUDAAllocator(device_id, name) {
    alloc_ = reinterpret_cast<ExternalAlloc>(alloc);
    free_ = reinterpret_cast<ExternalFree>(free);
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;

 private:
  ExternalAlloc alloc_;
  ExternalFree free_;
};

// TODO: add a default constructor
class CUDAPinnedAllocator : public IAllocator {
 public:
  CUDAPinnedAllocator(int device_id, const char* name)
      : IAllocator(BrtMemoryInfo(
            name, BrtAllocatorType::BrtDeviceAllocator, device_id, BrtMemTypeCPUOutput)) {
  }

  void* Alloc(size_t size) override;
  void Free(void* p) override;
};
}  // namespace brt
