// Copyright 2022 ByteDance Ltd. and/or its affiliates.
// Taken from https://github.com/abseil/abseil-cpp/blob/master/absl/base/internal/endian.h
//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

// The following guarantees declaration of the byte swap functions
#ifdef _MSC_VER
#include <stdlib.h>  // NOLINT(build/include)
#elif defined(__FreeBSD__)
#include <sys/endian.h>
#elif defined(__GLIBC__)
#include <byteswap.h>  // IWYU pragma: export
#endif

#include <cstdint>

namespace matxscript {
namespace runtime {
namespace base_internal {

// Use compiler byte-swapping intrinsics if they are available.  32-bit
// and 64-bit versions are available in Clang and GCC as of GCC 4.3.0.
// The 16-bit version is available in Clang and GCC only as of GCC 4.8.0.
// For simplicity, we enable them all only for GCC 4.8.0 or later.
#if defined(__clang__) || \
    (defined(__GNUC__) && ((__GNUC__ == 4 && __GNUC_MINOR__ >= 8) || __GNUC__ >= 5))
inline uint64_t gbswap_64(uint64_t host_int) {
  return __builtin_bswap64(host_int);
}
inline uint32_t gbswap_32(uint32_t host_int) {
  return __builtin_bswap32(host_int);
}
inline uint16_t gbswap_16(uint16_t host_int) {
  return __builtin_bswap16(host_int);
}

#elif defined(_MSC_VER)
inline uint64_t gbswap_64(uint64_t host_int) {
  return _byteswap_uint64(host_int);
}
inline uint32_t gbswap_32(uint32_t host_int) {
  return _byteswap_ulong(host_int);
}
inline uint16_t gbswap_16(uint16_t host_int) {
  return _byteswap_ushort(host_int);
}

#else
inline uint64_t gbswap_64(uint64_t host_int) {
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__APPLE__)
  // Adapted from /usr/include/byteswap.h.  Not available on Mac.
  if (__builtin_constant_p(host_int)) {
    return __bswap_constant_64(host_int);
  } else {
    uint64_t result;
    __asm__("bswap %0" : "=r"(result) : "0"(host_int));
    return result;
  }
#elif defined(__GLIBC__)
  return bswap_64(host_int);
#else
  return (((host_int & uint64_t{0xFF}) << 56) | ((host_int & uint64_t{0xFF00}) << 40) |
          ((host_int & uint64_t{0xFF0000}) << 24) | ((host_int & uint64_t{0xFF000000}) << 8) |
          ((host_int & uint64_t{0xFF00000000}) >> 8) |
          ((host_int & uint64_t{0xFF0000000000}) >> 24) |
          ((host_int & uint64_t{0xFF000000000000}) >> 40) |
          ((host_int & uint64_t{0xFF00000000000000}) >> 56));
#endif  // bswap_64
}

inline uint32_t gbswap_32(uint32_t host_int) {
#if defined(__GLIBC__)
  return bswap_32(host_int);
#else
  return (((host_int & uint32_t{0xFF}) << 24) | ((host_int & uint32_t{0xFF00}) << 8) |
          ((host_int & uint32_t{0xFF0000}) >> 8) | ((host_int & uint32_t{0xFF000000}) >> 24));
#endif
}

inline uint16_t gbswap_16(uint16_t host_int) {
#if defined(__GLIBC__)
  return bswap_16(host_int);
#else
  return (((host_int & uint16_t{0xFF}) << 8) | ((host_int & uint16_t{0xFF00}) >> 8));
#endif
}

#endif  // intrinsics available

}  // namespace base_internal
}  // namespace runtime
}  // namespace matxscript
