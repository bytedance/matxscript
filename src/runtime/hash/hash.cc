// Copyright 2022 ByteDance Ltd. and/or its affiliates.
// Taken from https://github.com/abseil/abseil-cpp/blob/master/absl/hash/internal/hash.h
//
// Copyright 2018 The Abseil Authors.
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
#include <matxscript/runtime/hash/base/config.h>
#include <matxscript/runtime/hash/base/unaligned_access.h>
#include <matxscript/runtime/hash/city_hash.h>
#include <matxscript/runtime/hash/hash.h>
#include <matxscript/runtime/hash/wy_hash.h>

namespace matxscript {
namespace runtime {

template <bool B, typename T, typename F>
using conditional_t = typename std::conditional<B, T, F>::type;

constexpr uint64_t kMul = sizeof(size_t) == 4 ? uint64_t{0xcc9e2d51} : uint64_t{0x9ddfea08eb382d69};

constexpr uint64_t kWyhashSalt[5] = {
    uint64_t{0x243F6A8885A308D3},
    uint64_t{0x13198A2E03707344},
    uint64_t{0xA4093822299F31D0},
    uint64_t{0x082EFA98EC4E6C89},
    uint64_t{0x452821E638D01377},
};

uint64_t Hasher::WyhashImpl(const unsigned char* data, size_t len) noexcept {
  return hash_internal::Wyhash(data, len, 0x7FFF71F49AF8UL, kWyhashSalt);
}

uint64_t Hasher::LargeImpl32(uint64_t state, const unsigned char* first, size_t len) noexcept {
  while (len >= PiecewiseChunkSize()) {
    state =
        Mix(state,
            hash_internal::CityHash32(reinterpret_cast<const char*>(first), PiecewiseChunkSize()));
    len -= PiecewiseChunkSize();
    first += PiecewiseChunkSize();
  }
  // Handle the remainder.
  return HashImpl(state, first, len, std::integral_constant<int, 4>{});
}

uint64_t Hasher::LargeImpl64(uint64_t state, const unsigned char* first, size_t len) noexcept {
  while (len >= PiecewiseChunkSize()) {
    state = Mix(state, Hash64(first, PiecewiseChunkSize()));
    len -= PiecewiseChunkSize();
    first += PiecewiseChunkSize();
  }
  // Handle the remainder.
  return HashImpl(state, first, len, std::integral_constant<int, 8>{});
}

// Reads 9 to 16 bytes from p.
// The least significant 8 bytes are in .first, the rest (zero padded) bytes
// are in .second.
std::pair<uint64_t, uint64_t> Hasher::Read9To16(const unsigned char* p, size_t len) noexcept {
  uint64_t low_mem = base_internal::UnalignedLoad64(p);
  uint64_t high_mem = base_internal::UnalignedLoad64(p + len - 8);
#ifdef MATXSCRIPT_IS_LITTLE_ENDIAN
  uint64_t most_significant = high_mem;
  uint64_t least_significant = low_mem;
#else
  uint64_t most_significant = low_mem;
  uint64_t least_significant = high_mem;
#endif
  return {least_significant, most_significant >> (128 - len * 8)};
}

// Reads 4 to 8 bytes from p. Zero pads to fill uint64_t.
uint64_t Hasher::Read4To8(const unsigned char* p, size_t len) noexcept {
  uint32_t low_mem = base_internal::UnalignedLoad32(p);
  uint32_t high_mem = base_internal::UnalignedLoad32(p + len - 4);
#ifdef MATXSCRIPT_IS_LITTLE_ENDIAN
  uint32_t most_significant = high_mem;
  uint32_t least_significant = low_mem;
#else
  uint32_t most_significant = low_mem;
  uint32_t least_significant = high_mem;
#endif
  return (static_cast<uint64_t>(most_significant) << (len - 4) * 8) | least_significant;
}

// Reads 1 to 3 bytes from p. Zero pads to fill uint32_t.
uint32_t Hasher::Read1To3(const unsigned char* p, size_t len) noexcept {
  unsigned char mem0 = p[0];
  unsigned char mem1 = p[len / 2];
  unsigned char mem2 = p[len - 1];
#ifdef MATXSCRIPT_IS_LITTLE_ENDIAN
  unsigned char significant2 = mem2;
  unsigned char significant1 = mem1;
  unsigned char significant0 = mem0;
#else
  unsigned char significant2 = mem0;
  unsigned char significant1 = mem1;
  unsigned char significant0 = mem2;
#endif
  return static_cast<uint32_t>(significant0 |                     //
                               (significant1 << (len / 2 * 8)) |  //
                               (significant2 << ((len - 1) * 8)));
}

uint64_t Hasher::Hash64(const unsigned char* data, size_t len) noexcept {
#ifdef MATXSCRIPT_HAVE_INTRINSIC_INT128
  return WyhashImpl(data, len);
#else
  return hash_internal::CityHash64(reinterpret_cast<const char*>(data), len);
#endif
}

uint64_t Hasher::Mix(uint64_t state, uint64_t v) noexcept {
#if defined(__aarch64__)
  // On AArch64, calculating a 128-bit product is inefficient, because it
  // requires a sequence of two instructions to calculate the upper and lower
  // halves of the result.
  using MultType = uint64_t;
#else
#ifndef MATXSCRIPT_HAVE_INTRINSIC_INT128
  return v;
#else
  typedef __uint128_t uint128;
  using MultType = conditional_t<sizeof(size_t) == 4, uint64_t, uint128>;
#endif
#endif
  // We do the addition in 64-bit space to make sure the 128-bit
  // multiplication is fast. If we were to do it as MultType the compiler has
  // to assume that the high word is non-zero and needs to perform 2
  // multiplications instead of one.
  MultType m = state + v;
  m *= kMul;
  return static_cast<uint64_t>(m ^ (m >> (sizeof(m) * 8 / 2)));
}

uint64_t Hasher::Hash(const unsigned char* data, size_t len) noexcept {
  uint64_t state = 0x7FFF71F49AF8;
  return HashImpl(state, data, len, std::integral_constant<int, sizeof(size_t)>{});
}

uint64_t Hasher::HashImpl(uint64_t state,
                          const unsigned char* first,
                          size_t len,
                          std::integral_constant<int, 4>
                          /* sizeof_size_t */) noexcept {
  // For large values we use CityHash, for small ones we just use a
  // multiplicative hash.
  uint64_t v;
  if (len > 8) {
    if (MATXSCRIPT_PREDICT_FALSE(len > PiecewiseChunkSize())) {
      return LargeImpl32(state, first, len);
    }
    v = hash_internal::CityHash32(reinterpret_cast<const char*>(first), len);
  } else if (len >= 4) {
    v = Read4To8(first, len);
  } else if (len > 0) {
    v = Read1To3(first, len);
  } else {
    // Empty ranges have no effect.
    return state;
  }
  return Mix(state, v);
}

uint64_t Hasher::HashImpl(uint64_t state,
                          const unsigned char* first,
                          size_t len,
                          std::integral_constant<int, 8>
                          /* sizeof_size_t */) noexcept {
  uint64_t v;
  if (len > 16) {
    if (MATXSCRIPT_PREDICT_FALSE(len > PiecewiseChunkSize())) {
      return LargeImpl64(state, first, len);
    }
    v = Hash64(first, len);
  } else if (len > 8) {
    auto p = Read9To16(first, len);
    state = Mix(state, p.first);
    v = p.second;
  } else if (len >= 4) {
    v = Read4To8(first, len);
  } else if (len > 0) {
    v = Read1To3(first, len);
  } else {
    // Empty ranges have no effect.
    return state;
  }
  return Mix(state, v);
}

}  // namespace runtime
}  // namespace matxscript
