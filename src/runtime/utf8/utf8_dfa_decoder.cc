// Copyright 2022 ByteDance Ltd. and/or its affiliates.
//==================================================================================================
//  File:       unicode_utils.cpp
//
//  Copyright (c) 2018 Bob Steagall and KEWB Computing, All Rights Reserved
//==================================================================================================
//
#include <matxscript/runtime/utf8/utf8_dfa_decoder.h>

#include <cstdio>
namespace matxscript {
namespace runtime {
namespace utf8_details {
// clang-format off

//- Static member data init.
//
DFADecoder::LookupTables const DFADecoder::smTables = {
    //- Initialize the maFirstUnitTable member array.  This array implements a lookup table that
    //  maps the first code unit of a sequence to: 1. a pre-masked value to start the computation
    //  of the resulting code point; and, 2. the next state in the DFA for this code unit.
    //
    {
        { 0x00, ERR, 0 },   //- 0x80
        { 0x01, ERR, 0 },   //- 0x81
        { 0x02, ERR, 0 },   //- 0x82
        { 0x03, ERR, 0 },   //- 0x83
        { 0x04, ERR, 0 },   //- 0x84
        { 0x05, ERR, 0 },   //- 0x85
        { 0x06, ERR, 0 },   //- 0x86
        { 0x07, ERR, 0 },   //- 0x87
        { 0x08, ERR, 0 },   //- 0x88
        { 0x09, ERR, 0 },   //- 0x89
        { 0x0A, ERR, 0 },   //- 0x8A
        { 0x0B, ERR, 0 },   //- 0x8B
        { 0x0C, ERR, 0 },   //- 0x8C
        { 0x0D, ERR, 0 },   //- 0x8D
        { 0x0E, ERR, 0 },   //- 0x8E
        { 0x0F, ERR, 0 },   //- 0x8F

        { 0x10, ERR, 0 },   //- 0x90
        { 0x11, ERR, 0 },   //- 0x91
        { 0x12, ERR, 0 },   //- 0x92
        { 0x13, ERR, 0 },   //- 0x93
        { 0x14, ERR, 0 },   //- 0x94
        { 0x15, ERR, 0 },   //- 0x95
        { 0x16, ERR, 0 },   //- 0x96
        { 0x17, ERR, 0 },   //- 0x97
        { 0x18, ERR, 0 },   //- 0x98
        { 0x19, ERR, 0 },   //- 0x99
        { 0x1A, ERR, 0 },   //- 0x9A
        { 0x1B, ERR, 0 },   //- 0x9B
        { 0x1C, ERR, 0 },   //- 0x9C
        { 0x1D, ERR, 0 },   //- 0x9D
        { 0x1E, ERR, 0 },   //- 0x9E
        { 0x1F, ERR, 0 },   //- 0x9F

        { 0x20, ERR, 0 },   //- 0xA0
        { 0x21, ERR, 0 },   //- 0xA1
        { 0x22, ERR, 0 },   //- 0xA2
        { 0x23, ERR, 0 },   //- 0xA3
        { 0x24, ERR, 0 },   //- 0xA4
        { 0x25, ERR, 0 },   //- 0xA5
        { 0x26, ERR, 0 },   //- 0xA6
        { 0x27, ERR, 0 },   //- 0xA7
        { 0x28, ERR, 0 },   //- 0xA8
        { 0x29, ERR, 0 },   //- 0xA9
        { 0x2A, ERR, 0 },   //- 0xAA
        { 0x2B, ERR, 0 },   //- 0xAB
        { 0x2C, ERR, 0 },   //- 0xAC
        { 0x2D, ERR, 0 },   //- 0xAD
        { 0x2E, ERR, 0 },   //- 0xAE
        { 0x2F, ERR, 0 },   //- 0xAF

        { 0x30, ERR, 0 },   //- 0xB0
        { 0x31, ERR, 0 },   //- 0xB1
        { 0x32, ERR, 0 },   //- 0xB2
        { 0x33, ERR, 0 },   //- 0xB3
        { 0x34, ERR, 0 },   //- 0xB4
        { 0x35, ERR, 0 },   //- 0xB5
        { 0x36, ERR, 0 },   //- 0xB6
        { 0x37, ERR, 0 },   //- 0xB7
        { 0x38, ERR, 0 },   //- 0xB8
        { 0x39, ERR, 0 },   //- 0xB9
        { 0x3A, ERR, 0 },   //- 0xBA
        { 0x3B, ERR, 0 },   //- 0xBB
        { 0x3C, ERR, 0 },   //- 0xBC
        { 0x3D, ERR, 0 },   //- 0xBD
        { 0x3E, ERR, 0 },   //- 0xBE
        { 0x3F, ERR, 0 },   //- 0xBF

        { 0xC0, ERR, 0 },   //- 0xC0
        { 0xC1, ERR, 0 },   //- 0xC1
        { 0x02, CS1, 2 },   //- 0xC2
        { 0x03, CS1, 2 },   //- 0xC3
        { 0x04, CS1, 2 },   //- 0xC4
        { 0x05, CS1, 2 },   //- 0xC5
        { 0x06, CS1, 2 },   //- 0xC6
        { 0x07, CS1, 2 },   //- 0xC7
        { 0x08, CS1, 2 },   //- 0xC8
        { 0x09, CS1, 2 },   //- 0xC9
        { 0x0A, CS1, 2 },   //- 0xCA
        { 0x0B, CS1, 2 },   //- 0xCB
        { 0x0C, CS1, 2 },   //- 0xCC
        { 0x0D, CS1, 2 },   //- 0xCD
        { 0x0E, CS1, 2 },   //- 0xCE
        { 0x0F, CS1, 2 },   //- 0xCF

        { 0x10, CS1, 2 },   //- 0xD0
        { 0x11, CS1, 2 },   //- 0xD1
        { 0x12, CS1, 2 },   //- 0xD2
        { 0x13, CS1, 2 },   //- 0xD3
        { 0x14, CS1, 2 },   //- 0xD4
        { 0x15, CS1, 2 },   //- 0xD5
        { 0x16, CS1, 2 },   //- 0xD6
        { 0x17, CS1, 2 },   //- 0xD7
        { 0x18, CS1, 2 },   //- 0xD8
        { 0x19, CS1, 2 },   //- 0xD9
        { 0x1A, CS1, 2 },   //- 0xDA
        { 0x1B, CS1, 2 },   //- 0xDB
        { 0x1C, CS1, 2 },   //- 0xDC
        { 0x1D, CS1, 2 },   //- 0xDD
        { 0x1E, CS1, 2 },   //- 0xDE
        { 0x1F, CS1, 2 },   //- 0xDF

        { 0x00, P3A, 3 },   //- 0xE0
        { 0x01, CS2, 3 },   //- 0xE1
        { 0x02, CS2, 3 },   //- 0xE2
        { 0x03, CS2, 3 },   //- 0xE3
        { 0x04, CS2, 3 },   //- 0xE4
        { 0x05, CS2, 3 },   //- 0xE5
        { 0x06, CS2, 3 },   //- 0xE6
        { 0x07, CS2, 3 },   //- 0xE7
        { 0x08, CS2, 3 },   //- 0xE8
        { 0x09, CS2, 3 },   //- 0xE9
        { 0x0A, CS2, 3 },   //- 0xEA
        { 0x0B, CS2, 3 },   //- 0xEB
        { 0x0C, CS2, 3 },   //- 0xEC
        { 0x0D, P3B, 3 },   //- 0xED
        { 0x0E, CS2, 3 },   //- 0xEE
        { 0x0F, CS2, 3 },   //- 0xEF

        { 0x00, P4A, 4 },   //- 0xF0
        { 0x01, CS3, 4 },   //- 0xF1
        { 0x02, CS3, 4 },   //- 0xF2
        { 0x03, CS3, 4 },   //- 0xF3
        { 0x04, P4B, 4 },   //- 0xF4
        { 0xF5, ERR, 0 },   //- 0xF5
        { 0xF6, ERR, 0 },   //- 0xF6
        { 0xF7, ERR, 0 },   //- 0xF7
        { 0xF8, ERR, 0 },   //- 0xF8
        { 0xF9, ERR, 0 },   //- 0xF9
        { 0xFA, ERR, 0 },   //- 0xFA
        { 0xFB, ERR, 0 },   //- 0xFB
        { 0xFC, ERR, 0 },   //- 0xFC
        { 0xFD, ERR, 0 },   //- 0xFD
        { 0xFE, ERR, 0 },   //- 0xFE
        { 0xFF, ERR, 0 },   //- 0xFF
    },

    //- Initialize the maOctetCategory member array.  This array implements a lookup table
    //  that maps an input octet to a corresponding octet category.
    //
    //   0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    F
    //============================================================================================
    {
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 00..0F
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 10..1F
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 20..2F
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 30..3F
                                                                                        //
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 40..4F
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 50..5F
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 60..6F
        ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, ASC, //- 70..7F
                                                                                        //
        CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, CR1, //- 80..8F
        CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, CR2, //- 90..9F
        CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, //- A0..AF
        CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, CR3, //- B0..BF
                                                                                        //
        ILL, ILL, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, //- C0..CF
        L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, L2A, //- D0..DF
        L3A, L3B, L3B, L3B, L3B, L3B, L3B, L3B, L3B, L3B, L3B, L3B, L3B, L3C, L3B, L3B, //- E0..EF
        L4A, L4B, L4B, L4B, L4C, ILL, ILL, ILL, ILL, ILL, ILL, ILL, ILL, ILL, ILL, ILL, //- F0..FF
    },

    //- Initialize the maTransitions member array.  This array implements a lookup table that,
    //  given the current DFA state and an input code unit, indicates the next DFA state.
    //
    //  ILL  ASC  CR1  CR2  CR3  L2A  L3A  L3B  L3C  L4A  L4B  L4C  CLASS/STATE
    //=========================================================================
    {
        err, END, err, err, err, CS1, P3A, CS2, P3B, P4A, CS3, P4B,	  //- BGN|END
        err, err, err, err, err, err, err, err, err, err, err, err,   //- ERR
                                    //
        err, err, END, END, END, err, err, err, err, err, err, err,   //- CS1
        err, err, CS1, CS1, CS1, err, err, err, err, err, err, err,   //- CS2
        err, err, CS2, CS2, CS2, err, err, err, err, err, err, err,   //- CS3
                                    //
        err, err, err, err, CS1, err, err, err, err, err, err, err,   //- P3A
        err, err, CS1, CS1, err, err, err, err, err, err, err, err,   //- P3B
                                    //
        err, err, err, CS2, CS2, err, err, err, err, err, err, err,   //- P4A
        err, err, CS2, err, err, err, err, err, err, err, err, err,   //- P4B
    },

    //- Initialize the maFirstOctetMask member array.  This array implements a lookup table that
    //  maps a character class to a mask that is applied to the first code unit in a sequence.
    //
    {
        0xFF,   //- ILL - C0..C1, F5..FF    Illegal code unit
                //
        0x7F,   //- ASC - 00..7F            ASCII byte range
                //
        0x3F,   //- CR1 - 80..8F            Continuation range 1
        0x3F,   //- CR2 - 90..9F            Continuation range 2
        0x3F,   //- CR3 - A0..BF            Continuation range 3
                //
        0x1F,   //- L2A - C2..DF            Leading byte range 2A / 2-byte sequence
                //
        0x0F,   //- L3A - E0                Leading byte range 3A / 3-byte sequence
        0x0F,   //- L3B - E1..EC, EE..EF    Leading byte range 3B / 3-byte sequence
        0x0F,   //- L3C - ED                Leading byte range 3C / 3-byte sequence
                //
        0x07,   //- L4A - F0                Leading byte range 4A / 4-byte sequence
        0x07,   //- L4B - F1..F3            Leading byte range 4B / 4-byte sequence
        0x07,   //- L4C - F4                Leading byte range 4C / 4-byte sequence
    },
};

//- These are the human-readable names assigned to the code unit categories.
//
char const*     DFADecoder::smClassNames[12] =
{
    "ILL", "ASC", "CR1", "CR2", "CR3", "L2A", "L3A", "L3B", "L3C", "L4A", "L4B", "L4C",
};

//- These are the human-readable names assigned to the various states comprising the DFA.
//
char const*     DFADecoder::smStateNames[9] =
{
    "BGN", "ERR", "CS1", "CS2", "CS3", "P3A", "P3B", "P4A", "P4B",
};

// clang-format on

//--------------------------------------------------------------------------------------------------
/// \brief  Converts a sequence of UTF-8 code units to a sequence of UTF-32 code points.
///
/// \details
///     This static member function reads an input sequence of UTF-8 code units and converts
///     it to an output sequence of UTF-32 code points.  It uses the DFA to perform non-ascii
///     code-unit sequence conversions, but optimizes by checking for ASCII code units and
///     converting them directly to code points.  It uses the `AdvanceWithSmallTable` member
///     function to read and convert input.
///
/// \param pSrc
///     A non-null pointer defining the beginning of the code unit input range.
/// \param pSrcEnd
///     A non-null past-the-end pointer defining the end of the code unit input range.
/// \param pDst
///     A non-null pointer defining the beginning of the code point output range.
///
/// \returns
///     If successful, the number of UTF-32 code points written; otherwise -1 is returned to
///     indicate an error was encountered.
//--------------------------------------------------------------------------------------------------
//
MATXSCRIPT_ALIGN_FUNCTION std::ptrdiff_t DFADecoder::Convert(char8_t const* pSrc,
                                                             char8_t const* pSrcEnd,
                                                             char32_t* pDst) noexcept {
  char32_t* pDstOrig = pDst;
  char32_t cdpt;

  // max_len=4, overflow 3
  // range(head, tail-3), no checking
  pSrcEnd = pSrcEnd - 3;
  while (pSrc < pSrcEnd) {
    char32_t unit = *pSrc++;  //- Cache the first code unit
    if (unit < 0x80) {
      *pDst++ = unit;
    } else {
      FirstUnitInfo info = smTables.maFirstUnitTable[unit - 0x80];
      State state = info.mNextState;
      cdpt = info.mFirstOctet;
      switch (info.size) {
        case 2: {
          // AdvanceWithTable<2>(pSrc, state, cdpt);
          AdvanceWithTable_2(pSrc, state, cdpt);
        } break;
        case 3: {
          // AdvanceWithTable<3>(pSrc, state, cdpt);
          AdvanceWithTable_3(pSrc, state, cdpt);
        } break;
        case 4: {
          // AdvanceWithTable<4>(pSrc, state, cdpt);
          AdvanceWithTable_4(pSrc, state, cdpt);
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
      State state = info.mNextState;
      cdpt = info.mFirstOctet;
      if (pSrc + info.size - 1 > pSrcEnd) {
        *pDst++ = 0xFFFD;
        return pDst - pDstOrig;
      }
      switch (info.size) {
        case 2: {
          // AdvanceWithTable<2>(pSrc, state, cdpt);
          AdvanceWithTable_2(pSrc, state, cdpt);
        } break;
        case 3: {
          // AdvanceWithTable<3>(pSrc, state, cdpt);
          AdvanceWithTable_3(pSrc, state, cdpt);
        } break;
        case 4: {
          // AdvanceWithTable<4>(pSrc, state, cdpt);
          AdvanceWithTable_4(pSrc, state, cdpt);
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

//--------------------------------------------------------------------------------------------------
/// \brief  Trace converts a sequence of UTF-8 code units to a sequence of UTF-32 code points.
///
/// \details
///     This static member function reads an input sequence of UTF-8 code units and converts
///     it to an output sequence of UTF-32 code points.  It uses only the DFA to perform
///     conversion.  It prints current and next state transition information as it proceeds.
///
/// \param pSrc
///     A non-null pointer defining the beginning of the code unit input range.
/// \param pSrcEnd
///     A non-null past-the-end pointer defining the end of the code unit input range.
/// \param pDst
///     A non-null pointer defining the beginning of the code point output range.
///
/// \returns
///     If successful, the number of UTF-32 code points written; otherwise -1 is returned to
///     indicate an error was encountered.
//--------------------------------------------------------------------------------------------------
//
std::ptrdiff_t DFADecoder::ConvertWithTrace(char8_t const* pSrc,
                                            char8_t const* pSrcEnd,
                                            char32_t* pDst) noexcept {
  char32_t* pDstOrig = pDst;
  char32_t cdpt;

  while (pSrc < pSrcEnd) {
    if (AdvanceWithTrace(pSrc, pSrcEnd, cdpt) != ERR) {
      *pDst++ = cdpt;
    } else {
      return -1;
    }
  }

  return pDst - pDstOrig;
}

//--------------------------------------------------------------------------------------------------
/// \brief  Prints state information for tracing versions of converters.
///
/// \param curr
///     The current DFA state.
/// \param type
///     The character class of the lookahead input octet.
/// \param unit
///     The lookahead input octet.
/// \param next
///     The next DFA state, based on the current state and lookahead character class.
//--------------------------------------------------------------------------------------------------
//
void DFADecoder::PrintStateData(State curr, CharClass type, uint32_t unit, State next) {
  uint32_t currState = ((uint32_t)curr) / 12;
  uint32_t nextState = ((uint32_t)next) / 12;
  uint32_t unitValue = unit & 0xFF;

  printf("[%s, %s (0x%02X)] ==>> %s\n",
         smStateNames[currState],
         smClassNames[type],
         unitValue,
         smStateNames[nextState]);
}

}  // namespace utf8_details
}  // namespace runtime
}  // namespace matxscript
