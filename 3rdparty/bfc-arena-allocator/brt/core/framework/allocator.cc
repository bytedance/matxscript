// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// ===========================================================================
// Modifications Copyright (c) ByteDance.

#include "core/framework/allocator.h"
#include <cstdlib>
#include <sstream>

namespace brt {

// private helper for calculation so SafeInt usage doesn't bleed into the public allocator.h header
bool IAllocator::CalcMemSizeForArrayWithAlignment(size_t nmemb,
                                                  size_t size,
                                                  size_t alignment,
                                                  size_t* out) noexcept {
  bool ok = true;

  if (alignment == 0) {
    *out = nmemb;
  } else {
    size_t alignment_mask = alignment - 1;
    *out = (nmemb + alignment_mask) & ~static_cast<size_t>(alignment_mask);
  }

  return ok;
}

namespace {
void* DefaultAlloc(size_t size) {
  if (size <= 0)
    return nullptr;
  void* p;
  size_t alignment = 32;
#if _MSC_VER
  p = _aligned_malloc(size, alignment);
  if (p == nullptr)
    abort();
#elif defined(_LIBCPP_SGX_CONFIG)
  p = memalign(alignment, size);
  if (p == nullptr)
    abort();
#else
  int ret = posix_memalign(&p, alignment, size);
  if (ret != 0)
    abort();
#endif
  return p;
}

void DefaultFree(void* p) {
#if _MSC_VER
  _aligned_free(p);
#else
  free(p);
#endif
}
}  // namespace

void* CPUAllocator::Alloc(size_t size) {
  return DefaultAlloc(size);
}

void CPUAllocator::Free(void* p) {
  DefaultFree(p);
}
}  // namespace brt

std::ostream& operator<<(std::ostream& out, const BrtMemoryInfo& info) {
  return (out << info.ToString());
}
