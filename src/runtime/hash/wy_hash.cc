// Copyright 2022 ByteDance Ltd. and/or its affiliates.
// Taken from https://github.com/abseil/abseil-cpp/blob/master/absl/hash/internal/wyhash.cc
//
// Copyright 2020 The Abseil Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <matxscript/runtime/hash/base/unaligned_access.h>
#include <matxscript/runtime/hash/wy_hash.h>

namespace matxscript {
namespace runtime {
namespace hash_internal {

#ifdef MATXSCRIPT_HAVE_INTRINSIC_INT128

inline uint64_t Uint128Low64(uint128 v) noexcept {
#if defined(MATXSCRIPT_IS_LITTLE_ENDIAN)
  return static_cast<uint64_t>(v & ~uint64_t{0});
#elif defined(MATXSCRIPT_IS_BIG_ENDIAN)
  return static_cast<uint64_t>(v >> 64);
#endif
}

inline uint64_t Uint128High64(uint128 v) noexcept {
#if defined(MATXSCRIPT_IS_LITTLE_ENDIAN)
  return static_cast<uint64_t>(v >> 64);
#elif defined(MATXSCRIPT_IS_BIG_ENDIAN)
  return static_cast<uint64_t>(v & ~uint64_t{0});
#endif
}

uint64_t WyhashMix(uint64_t v0, uint64_t v1) noexcept {
  uint128 p = v0;
  p *= v1;
  return Uint128Low64(p) ^ Uint128High64(p);
}

uint64_t Wyhash(const void* data, size_t len, uint64_t seed, const uint64_t salt[]) noexcept {
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  uint64_t starting_length = static_cast<uint64_t>(len);
  uint64_t current_state = seed ^ salt[0];

  if (len > 64) {
    // If we have more than 64 bytes, we're going to handle chunks of 64
    // bytes at a time. We're going to build up two separate hash states
    // which we will then hash together.
    uint64_t duplicated_state = current_state;

    do {
      uint64_t a = base_internal::UnalignedLoad64(ptr);
      uint64_t b = base_internal::UnalignedLoad64(ptr + 8);
      uint64_t c = base_internal::UnalignedLoad64(ptr + 16);
      uint64_t d = base_internal::UnalignedLoad64(ptr + 24);
      uint64_t e = base_internal::UnalignedLoad64(ptr + 32);
      uint64_t f = base_internal::UnalignedLoad64(ptr + 40);
      uint64_t g = base_internal::UnalignedLoad64(ptr + 48);
      uint64_t h = base_internal::UnalignedLoad64(ptr + 56);

      uint64_t cs0 = WyhashMix(a ^ salt[1], b ^ current_state);
      uint64_t cs1 = WyhashMix(c ^ salt[2], d ^ current_state);
      current_state = (cs0 ^ cs1);

      uint64_t ds0 = WyhashMix(e ^ salt[3], f ^ duplicated_state);
      uint64_t ds1 = WyhashMix(g ^ salt[4], h ^ duplicated_state);
      duplicated_state = (ds0 ^ ds1);

      ptr += 64;
      len -= 64;
    } while (len > 64);

    current_state = current_state ^ duplicated_state;
  }

  // We now have a data `ptr` with at most 64 bytes and the current state
  // of the hashing state machine stored in current_state.
  while (len > 16) {
    uint64_t a = base_internal::UnalignedLoad64(ptr);
    uint64_t b = base_internal::UnalignedLoad64(ptr + 8);

    current_state = WyhashMix(a ^ salt[1], b ^ current_state);

    ptr += 16;
    len -= 16;
  }

  // We now have a data `ptr` with at most 16 bytes.
  uint64_t a = 0;
  uint64_t b = 0;
  if (len > 8) {
    // When we have at least 9 and at most 16 bytes, set A to the first 64
    // bits of the input and B to the last 64 bits of the input. Yes, they will
    // overlap in the middle if we are working with less than the full 16
    // bytes.
    a = base_internal::UnalignedLoad64(ptr);
    b = base_internal::UnalignedLoad64(ptr + len - 8);
  } else if (len > 3) {
    // If we have at least 4 and at most 8 bytes, set A to the first 32
    // bits and B to the last 32 bits.
    a = base_internal::UnalignedLoad32(ptr);
    b = base_internal::UnalignedLoad32(ptr + len - 4);
  } else if (len > 0) {
    // If we have at least 1 and at most 3 bytes, read all of the provided
    // bits into A, with some adjustments.
    a = ((ptr[0] << 16) | (ptr[len >> 1] << 8) | ptr[len - 1]);
    b = 0;
  } else {
    a = 0;
    b = 0;
  }

  uint64_t w = WyhashMix(a ^ salt[1], b ^ current_state);
  uint64_t z = salt[1] ^ starting_length;
  return WyhashMix(w, z);
}

#endif

}  // namespace hash_internal
}  // namespace runtime
}  // namespace matxscript
