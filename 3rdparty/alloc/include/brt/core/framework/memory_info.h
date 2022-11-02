// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modifications Copyright (c) ByteDance.

#pragma once

#include <string.h>
#include <sstream>
#include <string>

/**
 * memory types for allocator, exec provider specific types should be extended in each provider
 * Whenever this struct is updated, please also update the MakeKey function in
 * onnxruntime/core/framework/execution_provider.cc
 */
typedef enum BrtMemType {
  BrtMemTypeCPUInput = -2,  // Any CPU memory used by non-CPU execution provider
  BrtMemTypeCPUOutput =
      -1,  // CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED
  BrtMemTypeCPU = BrtMemTypeCPUOutput,  // temporary CPU accessible memory allocated by non-CPU
                                        // execution provider, i.e. CUDA_PINNED
  BrtMemTypeDefault = 0,                // the default allocator for execution provider
} BrtMemType;

typedef enum BrtAllocatorType {
  Invalid = -1,
  BrtDeviceAllocator = 0,
  BrtArenaAllocator = 1
} BrtAllocatorType;

struct BrtMemoryInfo {
  BrtMemoryInfo() = default;  // to allow default construction of Tensor

  // use string for name, so we could have customized allocator in execution provider.
  const char* name = nullptr;
  int id = -1;
  BrtMemType mem_type = BrtMemTypeDefault;
  BrtAllocatorType alloc_type = Invalid;

  constexpr BrtMemoryInfo(const char* name_,
                          BrtAllocatorType type_,
                          int id_ = 0,
                          BrtMemType mem_type_ = BrtMemTypeDefault)
#if ((defined(__GNUC__) && __GNUC__ > 4) || defined(__clang__))
      // this causes a spurious error in CentOS gcc 4.8 build so disable if GCC version < 5
      __attribute__((nonnull))
#endif
      : name(name_), id(id_), mem_type(mem_type_), alloc_type(type_) {
  }

  // To make OrtMemoryInfo become a valid key in std map
  bool operator<(const BrtMemoryInfo& other) const {
    if (alloc_type != other.alloc_type)
      return alloc_type < other.alloc_type;
    if (mem_type != other.mem_type)
      return mem_type < other.mem_type;
    if (id != other.id)
      return id < other.id;

    return strcmp(name, other.name) < 0;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "BrtMemoryInfo:["
         << "name:" << name << " id:" << id << " BrtMemType:" << mem_type
         << " BrtAllocatorType:" << alloc_type << "]";
    return ostr.str();
  }
};

inline bool operator==(const BrtMemoryInfo& left, const BrtMemoryInfo& other) {
  return left.mem_type == other.mem_type && left.alloc_type == other.alloc_type &&
         left.id == other.id && strcmp(left.name, other.name) == 0;
}

inline bool operator!=(const BrtMemoryInfo& lhs, const BrtMemoryInfo& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, const BrtMemoryInfo& info);
