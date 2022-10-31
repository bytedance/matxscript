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
#include <matxscript/runtime/utf8/utf8_greedy_table_decoder.h>

namespace matxscript {
namespace runtime {
namespace utf8_details {
// clang-format off

//- Static member data init.
//
GreedyTableDecoder::LookupTables const GreedyTableDecoder::smTables = {
    //- Initialize the maFirstUnitTable member array.  This array implements a lookup table that
    //  maps the first code unit of a sequence to: 1. a pre-masked value to start the computation
    //  of the resulting code point; and, 2. the total bytes num for this code unit.
    //
    {
        { 0x00, 0 },   //- 0x80
        { 0x01, 0 },   //- 0x81
        { 0x02, 0 },   //- 0x82
        { 0x03, 0 },   //- 0x83
        { 0x04, 0 },   //- 0x84
        { 0x05, 0 },   //- 0x85
        { 0x06, 0 },   //- 0x86
        { 0x07, 0 },   //- 0x87
        { 0x08, 0 },   //- 0x88
        { 0x09, 0 },   //- 0x89
        { 0x0A, 0 },   //- 0x8A
        { 0x0B, 0 },   //- 0x8B
        { 0x0C, 0 },   //- 0x8C
        { 0x0D, 0 },   //- 0x8D
        { 0x0E, 0 },   //- 0x8E
        { 0x0F, 0 },   //- 0x8F

        { 0x10, 0 },   //- 0x90
        { 0x11, 0 },   //- 0x91
        { 0x12, 0 },   //- 0x92
        { 0x13, 0 },   //- 0x93
        { 0x14, 0 },   //- 0x94
        { 0x15, 0 },   //- 0x95
        { 0x16, 0 },   //- 0x96
        { 0x17, 0 },   //- 0x97
        { 0x18, 0 },   //- 0x98
        { 0x19, 0 },   //- 0x99
        { 0x1A, 0 },   //- 0x9A
        { 0x1B, 0 },   //- 0x9B
        { 0x1C, 0 },   //- 0x9C
        { 0x1D, 0 },   //- 0x9D
        { 0x1E, 0 },   //- 0x9E
        { 0x1F, 0 },   //- 0x9F

        { 0x20, 0 },   //- 0xA0
        { 0x21, 0 },   //- 0xA1
        { 0x22, 0 },   //- 0xA2
        { 0x23, 0 },   //- 0xA3
        { 0x24, 0 },   //- 0xA4
        { 0x25, 0 },   //- 0xA5
        { 0x26, 0 },   //- 0xA6
        { 0x27, 0 },   //- 0xA7
        { 0x28, 0 },   //- 0xA8
        { 0x29, 0 },   //- 0xA9
        { 0x2A, 0 },   //- 0xAA
        { 0x2B, 0 },   //- 0xAB
        { 0x2C, 0 },   //- 0xAC
        { 0x2D, 0 },   //- 0xAD
        { 0x2E, 0 },   //- 0xAE
        { 0x2F, 0 },   //- 0xAF

        { 0x30, 0 },   //- 0xB0
        { 0x31, 0 },   //- 0xB1
        { 0x32, 0 },   //- 0xB2
        { 0x33, 0 },   //- 0xB3
        { 0x34, 0 },   //- 0xB4
        { 0x35, 0 },   //- 0xB5
        { 0x36, 0 },   //- 0xB6
        { 0x37, 0 },   //- 0xB7
        { 0x38, 0 },   //- 0xB8
        { 0x39, 0 },   //- 0xB9
        { 0x3A, 0 },   //- 0xBA
        { 0x3B, 0 },   //- 0xBB
        { 0x3C, 0 },   //- 0xBC
        { 0x3D, 0 },   //- 0xBD
        { 0x3E, 0 },   //- 0xBE
        { 0x3F, 0 },   //- 0xBF

        { 0xC0, 0 },   //- 0xC0
        { 0xC1, 0 },   //- 0xC1
        { 0x02, 2 },   //- 0xC2
        { 0x03, 2 },   //- 0xC3
        { 0x04, 2 },   //- 0xC4
        { 0x05, 2 },   //- 0xC5
        { 0x06, 2 },   //- 0xC6
        { 0x07, 2 },   //- 0xC7
        { 0x08, 2 },   //- 0xC8
        { 0x09, 2 },   //- 0xC9
        { 0x0A, 2 },   //- 0xCA
        { 0x0B, 2 },   //- 0xCB
        { 0x0C, 2 },   //- 0xCC
        { 0x0D, 2 },   //- 0xCD
        { 0x0E, 2 },   //- 0xCE
        { 0x0F, 2 },   //- 0xCF

        { 0x10, 2 },   //- 0xD0
        { 0x11, 2 },   //- 0xD1
        { 0x12, 2 },   //- 0xD2
        { 0x13, 2 },   //- 0xD3
        { 0x14, 2 },   //- 0xD4
        { 0x15, 2 },   //- 0xD5
        { 0x16, 2 },   //- 0xD6
        { 0x17, 2 },   //- 0xD7
        { 0x18, 2 },   //- 0xD8
        { 0x19, 2 },   //- 0xD9
        { 0x1A, 2 },   //- 0xDA
        { 0x1B, 2 },   //- 0xDB
        { 0x1C, 2 },   //- 0xDC
        { 0x1D, 2 },   //- 0xDD
        { 0x1E, 2 },   //- 0xDE
        { 0x1F, 2 },   //- 0xDF

        { 0x00, 3 },   //- 0xE0
        { 0x01, 3 },   //- 0xE1
        { 0x02, 3 },   //- 0xE2
        { 0x03, 3 },   //- 0xE3
        { 0x04, 3 },   //- 0xE4
        { 0x05, 3 },   //- 0xE5
        { 0x06, 3 },   //- 0xE6
        { 0x07, 3 },   //- 0xE7
        { 0x08, 3 },   //- 0xE8
        { 0x09, 3 },   //- 0xE9
        { 0x0A, 3 },   //- 0xEA
        { 0x0B, 3 },   //- 0xEB
        { 0x0C, 3 },   //- 0xEC
        { 0x0D, 3 },   //- 0xED
        { 0x0E, 3 },   //- 0xEE
        { 0x0F, 3 },   //- 0xEF

        { 0x00, 4 },   //- 0xF0
        { 0x01, 4 },   //- 0xF1
        { 0x02, 4 },   //- 0xF2
        { 0x03, 4 },   //- 0xF3
        { 0x04, 4 },   //- 0xF4
        { 0xF5, 0 },   //- 0xF5
        { 0xF6, 0 },   //- 0xF6
        { 0xF7, 0 },   //- 0xF7
        { 0xF8, 0 },   //- 0xF8
        { 0xF9, 0 },   //- 0xF9
        { 0xFA, 0 },   //- 0xFA
        { 0xFB, 0 },   //- 0xFB
        { 0xFC, 0 },   //- 0xFC
        { 0xFD, 0 },   //- 0xFD
        { 0xFE, 0 },   //- 0xFE
        { 0xFF, 0 },   //- 0xFF
    },
};

// clang-format on

MATXSCRIPT_ALIGN_FUNCTION std::ptrdiff_t GreedyTableDecoder::Convert(const char8_t* pSrc,
                                                                     const char8_t* pSrcEnd,
                                                                     char32_t* pDst) noexcept {
  char32_t* pDstOrig = pDst;
  char32_t cdpt;

  // range(head, tail-3), no checking
  pSrcEnd = pSrcEnd - 3;
  while (pSrc < pSrcEnd) {
    char32_t unit = *pSrc++;  //- Cache the first code unit
    if (unit < 0x80) {
      *pDst++ = unit;
    } else {
      FirstUnitInfo info = smTables.maFirstUnitTable[unit - 0x80];
      cdpt = info.mFirstOctet;
      switch (info.size) {
        case 2: {
          AdvanceByGreedy_2(pSrc, cdpt);
        } break;
        case 3: {
          AdvanceByGreedy_3(pSrc, cdpt);
        } break;
        case 4: {
          AdvanceByGreedy_4(pSrc, cdpt);
        } break;
        default: {
          // https://unicodebook.readthedocs.io/unicode_encodings.html
          // UTF-8 encoding supports longer byte sequences, up to 6 bytes,
          // but the biggest code point of Unicode 6.0 (U+10FFFF) only takes 4 bytes.
          cdpt = 0xFFFD;
        } break;
      }
      *pDst++ = cdpt;
    }
  }

  // range(tail-3, tail), add check
  pSrcEnd += 3;
  while (pSrc < pSrcEnd) {
    char32_t unit = *pSrc++;  //- Cache the first code unit
    if (unit < 0x80) {
      *pDst++ = unit;
    } else {
      FirstUnitInfo info = smTables.maFirstUnitTable[unit - 0x80];
      cdpt = info.mFirstOctet;
      if (pSrc + info.size - 1 > pSrcEnd) {
        *pDst++ = 0xFFFD;
        return pDst - pDstOrig;
      }
      switch (info.size) {
        case 2: {
          AdvanceByGreedy_2(pSrc, cdpt);
        } break;
        case 3: {
          AdvanceByGreedy_3(pSrc, cdpt);
        } break;
        case 4: {
          AdvanceByGreedy_4(pSrc, cdpt);
        } break;
        default: {
          // https://unicodebook.readthedocs.io/unicode_encodings.html
          // UTF-8 encoding supports longer byte sequences, up to 6 bytes,
          // but the biggest code point of Unicode 6.0 (U+10FFFF) only takes 4 bytes.
          cdpt = 0xFFFD;
        } break;
      }
      *pDst++ = cdpt;
    }
  }

  return pDst - pDstOrig;
}

size_t GreedyTableDecoder::CountUnitSize(const char8_t* pSrc, const char8_t* pSrcEnd) noexcept {
  size_t count = 0;

  // range(head, tail-3), no checking
  pSrcEnd = pSrcEnd - 3;
  while (pSrc < pSrcEnd) {
    char32_t unit = *pSrc++;  //- Cache the first code unit
    ++count;
    if (unit >= 0x80) {
      FirstUnitInfo info = smTables.maFirstUnitTable[unit - 0x80];
      switch (info.size) {
        case 2: {
          SkipByGreedy_2(pSrc);
        } break;
        case 3: {
          SkipByGreedy_3(pSrc);
        } break;
        case 4: {
          SkipByGreedy_4(pSrc);
        } break;
        default: {
          // https://unicodebook.readthedocs.io/unicode_encodings.html
          // UTF-8 encoding supports longer byte sequences, up to 6 bytes,
          // but the biggest code point of Unicode 6.0 (U+10FFFF) only takes 4 bytes.
        } break;
      }
    }
  }

  // range(tail-3, tail), add check
  pSrcEnd += 3;
  while (pSrc < pSrcEnd) {
    char32_t unit = *pSrc++;  //- Cache the first code unit
    ++count;
    if (unit >= 0x80) {
      FirstUnitInfo info = smTables.maFirstUnitTable[unit - 0x80];
      if (pSrc + info.size - 1 > pSrcEnd) {
        return count;
      }
      switch (info.size) {
        case 2: {
          SkipByGreedy_2(pSrc);
        } break;
        case 3: {
          SkipByGreedy_3(pSrc);
        } break;
        case 4: {
          SkipByGreedy_4(pSrc);
        } break;
        default: {
          // https://unicodebook.readthedocs.io/unicode_encodings.html
          // UTF-8 encoding supports longer byte sequences, up to 6 bytes,
          // but the biggest code point of Unicode 6.0 (U+10FFFF) only takes 4 bytes.
        } break;
      }
    }
  }

  return count;
}

}  // namespace utf8_details
}  // namespace runtime
}  // namespace matxscript
