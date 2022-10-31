// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "matxscript/runtime/runtime_port.h"

namespace matxscript {
namespace runtime {
namespace utf8_details {

class GreedyTableDecoder {
 public:
  using char8_t = unsigned char;
  using ptrdiff_t = std::ptrdiff_t;

 public:
  static size_t CountUnitSize(char8_t const* pSrc, char8_t const* pSrcEnd) noexcept;

  // Conversion to UTF-32 using greedy lookup table
  static ptrdiff_t Convert(char8_t const* pSrc, char8_t const* pSrcEnd, char32_t* pDst) noexcept;

 private:
  struct FirstUnitInfo {
    char8_t mFirstOctet;
    std::uint8_t size;
  };
  struct alignas(64) LookupTables {
    FirstUnitInfo maFirstUnitTable[128];
  };

 private:
  static LookupTables const smTables;

 private:
  static void SkipByGreedy_2(char8_t const*& pSrc) noexcept;
  static void AdvanceByGreedy_2(char8_t const*& pSrc, char32_t& cdpt) noexcept;

  static void SkipByGreedy_3(char8_t const*& pSrc) noexcept;
  static void AdvanceByGreedy_3(char8_t const*& pSrc, char32_t& cdpt) noexcept;

  static void SkipByGreedy_4(char8_t const*& pSrc) noexcept;
  static void AdvanceByGreedy_4(char8_t const*& pSrc, char32_t& cdpt) noexcept;
};

MATXSCRIPT_ALWAYS_INLINE void GreedyTableDecoder::SkipByGreedy_2(const char8_t*& pSrc) noexcept {
  if ((pSrc[0] & 0xC0) == 0x80) {
    pSrc += 1;
  }
}

MATXSCRIPT_ALWAYS_INLINE void GreedyTableDecoder::AdvanceByGreedy_2(const char8_t*& pSrc,
                                                                    char32_t& cdpt) noexcept {
  char32_t unit2;   //- The second UTF-8 code unit
  unit2 = pSrc[0];  //- Cache the second code unit

  //- Compute code point
  if ((unit2 & 0xC0) != 0x80) {
    cdpt = 0xFFFD;
  } else {
    cdpt = (cdpt << 6) | (unit2 & 0x3F);
    pSrc += 1;
  }
}

MATXSCRIPT_ALWAYS_INLINE void GreedyTableDecoder::SkipByGreedy_3(const char8_t*& pSrc) noexcept {
  if (((pSrc[0] & 0xC0) == 0x80) && ((pSrc[1] & 0xC0) == 0x80)) {
    pSrc += 2;
  }
}

MATXSCRIPT_ALWAYS_INLINE void GreedyTableDecoder::AdvanceByGreedy_3(const char8_t*& pSrc,
                                                                    char32_t& cdpt) noexcept {
  char32_t unit2;  //- The second UTF-8 code unit
  char32_t unit3;  //- The third UTF-8 code unit

  unit2 = pSrc[0];  //- Cache the second code unit
  unit3 = pSrc[1];  //- Cache the third code unit

  //- Compute code point
  if ((unit2 & 0xC0) != 0x80 || (unit3 & 0xC0) != 0x80) {
    cdpt = 0xFFFD;
  } else {
    cdpt = (cdpt << 12) | ((unit2 & 0x3F) << 6) | (unit3 & 0x3F);
    pSrc += 2;
  }
}

MATXSCRIPT_ALWAYS_INLINE void GreedyTableDecoder::SkipByGreedy_4(const char8_t*& pSrc) noexcept {
  if (((pSrc[0] & 0xC0) == 0x80) && ((pSrc[1] & 0xC0) == 0x80) && ((pSrc[2] & 0xC0) == 0x80)) {
    pSrc += 3;
  }
}

MATXSCRIPT_ALWAYS_INLINE void GreedyTableDecoder::AdvanceByGreedy_4(const char8_t*& pSrc,
                                                                    char32_t& cdpt) noexcept {
  char32_t unit2;  //- The second UTF-8 code unit
  char32_t unit3;  //- The third UTF-8 code unit
  char32_t unit4;  //- The fourth UTF-8 code unit

  unit2 = pSrc[0];  //- Cache the second code unit
  unit3 = pSrc[1];  //- Cache the third code unit
  unit4 = pSrc[2];  //- Cache the fourth code unit

  //- Compute code point
  if ((unit2 & 0xC0) != 0x80 || (unit3 & 0xC0) != 0x80 || (unit4 & 0xC0) != 0x80) {
    cdpt = 0xFFFD;
  } else {
    cdpt = (cdpt << 18) | ((unit2 & 0x3F) << 12) | ((unit3 & 0x3F) << 6) | (unit4 & 0x3F);
    pSrc += 3;
  }
}

}  // namespace utf8_details
}  // namespace runtime
}  // namespace matxscript
