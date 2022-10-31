// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The LookupTables originates from by https://github.com/BobSteagall/utf_utils.
 * =============================================================================
 *  Copyright (c) 2018 Bob Steagall and KEWB Computing, All Rights Reserved
 * =============================================================================
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
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "matxscript/runtime/runtime_port.h"

namespace matxscript {
namespace runtime {
namespace utf8_details {
//--------------------------------------------------------------------------------------------------
/// \brief  Traits style class to perform conversions from UTF-8 to UTF-32
///
/// \details
///     This traits-style class provides a demonstration of functions for converting strings
///     of UTF-8 code units to strings of UTF-32 code points, as well as transcoding UTF-8
///     into strings of UTF-16 code units.  Its focus is on converting _from_ UTF-8 as quickly
///     as possible, although it does include member functions for converting a UTF-32 code
///     point into sequences of UTF-8/UTF-16 code units.
///
///     It implements conversion from UTF-8 in three different, but related ways:
///       * using a purely DFA-based approach to recognizing valid sequences of UTF-8 code units;
///       * using the DFA-based approach with a short-circuit optimization for ASCII code units;
///
///     The member functions implement STL-style argument ordering, with source arguments on the
///     left and destination arguments on the right.  The string-to-string conversion member
///     functions are analogous to std::copy() in that the first two arguments define an input
///     range and the third argument defines the starting point of the output range.
//--------------------------------------------------------------------------------------------------
//
class DFADecoder {
 public:
  using char8_t = unsigned char;
  using ptrdiff_t = std::ptrdiff_t;

 public:
  //- Conversion to UTF-32/UTF-16 using small lookup table and masking operations on first code
  // unit.
  //
  static ptrdiff_t Convert(char8_t const* pSrc, char8_t const* pSrcEnd, char32_t* pDst) noexcept;

  //- Conversion that traces path through DFA, writing to stdout.
  //
  static ptrdiff_t ConvertWithTrace(char8_t const* pSrc,
                                    char8_t const* pSrcEnd,
                                    char32_t* pDst) noexcept;

 private:
  enum CharClass : uint8_t {
    ILL = 0,   //- C0..C1, F5..FF  ILLEGAL octets that should never appear in a UTF-8 sequence
               //
    ASC = 1,   //- 00..7F          ASCII leading byte range
               //
    CR1 = 2,   //- 80..8F          Continuation range 1
    CR2 = 3,   //- 90..9F          Continuation range 2
    CR3 = 4,   //- A0..BF          Continuation range 3
               //
    L2A = 5,   //- C2..DF          Leading byte range A / 2-byte sequence
               //
    L3A = 6,   //- E0              Leading byte range A / 3-byte sequence
    L3B = 7,   //- E1..EC, EE..EF  Leading byte range B / 3-byte sequence
    L3C = 8,   //- ED              Leading byte range C / 3-byte sequence
               //
    L4A = 9,   //- F0              Leading byte range A / 4-byte sequence
    L4B = 10,  //- F1..F3          Leading byte range B / 4-byte sequence
    L4C = 11,  //- F4              Leading byte range C / 4-byte sequence
  };

  enum State : uint8_t {
    BGN = 0,    //- Start
    ERR = 12,   //- Invalid sequence
                //
    CS1 = 24,   //- Continuation state 1
    CS2 = 36,   //- Continuation state 2
    CS3 = 48,   //- Continuation state 3
                //
    P3A = 60,   //- Partial 3-byte sequence state A
    P3B = 72,   //- Partial 3-byte sequence state B
                //
    P4A = 84,   //- Partial 4-byte sequence state A
    P4B = 96,   //- Partial 4-byte sequence state B
                //
    END = BGN,  //- Start and End are the same state!
    err = ERR,  //- For readability in the state transition table
  };

  struct FirstUnitInfo {
    char8_t mFirstOctet;
    State mNextState;
    std::uint8_t size;
  };

  struct alignas(128) LookupTables {
    FirstUnitInfo maFirstUnitTable[128];
    CharClass maOctetCategory[256];
    State maTransitions[108];
    std::uint8_t maFirstOctetMask[16];
  };

 private:
  static LookupTables const smTables;
  static char const* smClassNames[12];
  static char const* smStateNames[9];

 private:
  template <int32_t size>
  static void AdvanceWithTable(char8_t const*& pSrc, int32_t curr, char32_t& cdpt) noexcept;

  static void AdvanceWithTable_2(char8_t const*& pSrc, int32_t curr, char32_t& cdpt) noexcept;
  static void AdvanceWithTable_3(char8_t const*& pSrc, int32_t curr, char32_t& cdpt) noexcept;
  static void AdvanceWithTable_4(char8_t const*& pSrc, int32_t curr, char32_t& cdpt) noexcept;

  static State AdvanceWithTrace(char8_t const*& pSrc,
                                char8_t const* pSrcEnd,
                                char32_t& cdpt) noexcept;

  static void PrintStateData(State curr, CharClass type, uint32_t unit, State next);
};

//--------------------------------------------------------------------------------------------------
/// \brief  Converts a sequence of UTF-8 code units to a UTF-32 code point.
///
/// \details
///     This static member function reads input octets and uses them to traverse a DFA that
///     recognizes valid sequences of UTF-8 code units.  It is the heart of all non-ASCII
///     conversions in all member functions of this class.  This function uses the "small"
///     first-unit lookup table and the state machine table to traverse the DFA.
///
/// \param pSrc
///     A reference to a non-null pointer defining the beginning of the code unit input range.
/// \param pSrcEnd
///     A non-null past-the-end pointer defining the end of the code unit input range.
/// \param cdpt
///     A reference to the output code point.
///
/// \returns
///     An internal flag describing the current DFA state.
//--------------------------------------------------------------------------------------------------
//
template <int32_t size>
MATXSCRIPT_ALWAYS_INLINE void DFADecoder::AdvanceWithTable(char8_t const*& pSrc,
                                                           int32_t curr,
                                                           char32_t& cdpt) noexcept {
  char32_t unit;  //- The current UTF-8 code unit
  int32_t type;   //- The current code unit's character class

#pragma unroll(size)
  for (int32_t i = 1; i < size; ++i) {
    unit = *pSrc++;                              //- Cache the current code unit
    cdpt = (cdpt << 6) | (unit & 0x3F);          //- Adjust code point with continuation bits
    type = smTables.maOctetCategory[unit];       //- Look up the code unit's character class
    curr = smTables.maTransitions[curr + type];  //- Look up the next state
  }
  if (curr == ERR) {
    //    pSrc -= (size - 1);
    cdpt = 0xFFFD;
  }
}

MATXSCRIPT_ALWAYS_INLINE void DFADecoder::AdvanceWithTable_2(const char8_t*& pSrc,
                                                             int32_t curr,
                                                             char32_t& cdpt) noexcept {
  char32_t unit2;  //- The second UTF-8 code unit
  int32_t type;    //- The current code unit's character class

  unit2 = pSrc[0];  //- Cache the second code unit

  type = smTables.maOctetCategory[unit2];      //- Look up the code unit's character class
  curr = smTables.maTransitions[curr + type];  //- Look up the next state

  //- Compute code point
  if (curr == ERR) {
    cdpt = 0xFFFD;
  } else {
    cdpt = (cdpt << 6) | (unit2 & 0x3F);
    pSrc += 1;
  }
}

MATXSCRIPT_ALWAYS_INLINE void DFADecoder::AdvanceWithTable_3(const char8_t*& pSrc,
                                                             int32_t curr,
                                                             char32_t& cdpt) noexcept {
  char32_t unit2;  //- The second UTF-8 code unit
  char32_t unit3;  //- The third UTF-8 code unit
  int32_t type;    //- The current code unit's character class

  unit2 = pSrc[0];  //- Cache the second code unit
  unit3 = pSrc[1];  //- Cache the third code unit

  type = smTables.maOctetCategory[unit2];      //- Look up the code unit's character class
  curr = smTables.maTransitions[curr + type];  //- Look up the next state
  type = smTables.maOctetCategory[unit3];      //- Look up the code unit's character class
  curr = smTables.maTransitions[curr + type];  //- Look up the next state

  //- Compute code point
  if (curr == ERR) {
    cdpt = 0xFFFD;
  } else {
    cdpt = (cdpt << 12) | ((unit2 & 0x3F) << 6) | (unit3 & 0x3F);
    pSrc += 2;
  }
}

MATXSCRIPT_ALWAYS_INLINE void DFADecoder::AdvanceWithTable_4(const char8_t*& pSrc,
                                                             int32_t curr,
                                                             char32_t& cdpt) noexcept {
  char32_t unit2;  //- The second UTF-8 code unit
  char32_t unit3;  //- The third UTF-8 code unit
  char32_t unit4;  //- The fourth UTF-8 code unit
  int32_t type;    //- The current code unit's character class

  unit2 = pSrc[0];  //- Cache the second code unit
  unit3 = pSrc[1];  //- Cache the third code unit
  unit4 = pSrc[2];  //- Cache the fourth code unit

  type = smTables.maOctetCategory[unit2];      //- Look up the code unit's character class
  curr = smTables.maTransitions[curr + type];  //- Look up the next state
  type = smTables.maOctetCategory[unit3];      //- Look up the code unit's character class
  curr = smTables.maTransitions[curr + type];  //- Look up the next state
  type = smTables.maOctetCategory[unit4];      //- Look up the code unit's character class
  curr = smTables.maTransitions[curr + type];  //- Look up the next state

  //- Compute code point
  if (curr == ERR) {
    cdpt = 0xFFFD;
  } else {
    cdpt = (cdpt << 18) | ((unit2 & 0x3F) << 12) | ((unit3 & 0x3F) << 6) | (unit4 & 0x3F);
    pSrc += 3;
  }
}

//--------------------------------------------------------------------------------------------------
/// \brief  Converts a sequence of UTF-8 code units to a UTF-32 code point.
///
/// \details
///     This static member function reads input octets and uses them to traverse a DFA that
///     recognizes valid sequences of UTF-8 code units.  It prints state information to `stdout`
///     while doing so.  It is simpler and slower than its counterpart `NextCodePoint`.
///
/// \param pSrc
///     A reference to a non-null pointer defining the beginning of the code unit input range.
/// \param pSrcEnd
///     A non-null past-the-end pointer defining the end of the code unit input range.
/// \param cdpt
///     A reference to the output code point.
///
/// \returns
///     An internal flag describing the current DFA state.
//--------------------------------------------------------------------------------------------------
//
MATXSCRIPT_ALWAYS_INLINE DFADecoder::State DFADecoder::AdvanceWithTrace(char8_t const*& pSrc,
                                                                        char8_t const* pSrcEnd,
                                                                        char32_t& cdpt) noexcept {
  char32_t unit;   //- The current UTF-8 code unit
  CharClass type;  //- The UTF-8 "sequence class"
  State next;      //- The next DFA state
  State curr;      //- The current DFA state

  unit = *pSrc++;
  type = smTables.maOctetCategory[unit];
  cdpt = smTables.maFirstOctetMask[type] & unit;
  curr = BGN;
  next = smTables.maTransitions[type];

  PrintStateData(curr, type, (char8_t)unit, next);

  while (next > ERR) {
    if (pSrc < pSrcEnd) {
      unit = *pSrc++;
      cdpt = (cdpt << 6) | (unit & 0x3F);
      type = smTables.maOctetCategory[unit];
      curr = next;
      next = smTables.maTransitions[curr + type];
      PrintStateData(curr, type, (char8_t)unit, next);
    } else {
      return ERR;
    }
  }
  return next;
}

}  // namespace utf8_details
}  // namespace runtime
}  // namespace matxscript
