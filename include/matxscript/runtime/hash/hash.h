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

#pragma once

#include <stdint.h>
#include <stdlib.h>  // for size_t.

namespace matxscript {
namespace runtime {

constexpr size_t PiecewiseChunkSize() noexcept {
  return 1024;
}

class Hasher {
 public:
  Hasher() = delete;

  Hasher(const Hasher& other) = delete;
  Hasher& operator=(const Hasher& other) = delete;
  Hasher(Hasher&& other) = delete;
  Hasher& operator=(Hasher&& other) = delete;

  static uint64_t Hash(const unsigned char* data, size_t len) noexcept;

 private:
  static uint64_t HashImpl(uint64_t state,
                           const unsigned char* first,
                           size_t len,
                           std::integral_constant<int, 4>
                           /* sizeof_size_t */) noexcept;

  static uint64_t HashImpl(uint64_t state,
                           const unsigned char* first,
                           size_t len,
                           std::integral_constant<int, 8>
                           /* sizeof_size_t */) noexcept;

  static uint64_t Hash64(const unsigned char* data, size_t len) noexcept;

  static uint64_t WyhashImpl(const unsigned char* data, size_t len) noexcept;

  static uint64_t LargeImpl32(uint64_t state, const unsigned char* first, size_t len) noexcept;
  static uint64_t LargeImpl64(uint64_t state, const unsigned char* first, size_t len) noexcept;

  static std::pair<uint64_t, uint64_t> Read9To16(const unsigned char* p, size_t len) noexcept;
  static uint64_t Read4To8(const unsigned char* p, size_t len) noexcept;
  static uint32_t Read1To3(const unsigned char* p, size_t len) noexcept;

  static uint64_t Mix(uint64_t state, uint64_t v) noexcept;
};

}  // namespace runtime
}  // namespace matxscript