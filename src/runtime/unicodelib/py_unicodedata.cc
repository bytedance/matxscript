// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Modules/unicodedata.c
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
/* ------------------------------------------------------------------------
  this file is copy from python
  https://github.com/python/cpython/blob/3.8/Modules/unicodedata.c

  unicodedata -- Provides access to the Unicode database.

  Data was extracted from the UnicodeData.txt file.
  The current version number is reported in the unidata_version constant.

  Written by Marc-Andre Lemburg (mal@lemburg.com).
  Modified for Python 2.0 by Fredrik Lundh (fredrik@pythonware.com)
  Modified by Martin v. Löwis (martin@v.loewis.de)

  Copyright (c) Corporation for National Research Initiatives.

  ------------------------------------------------------------------------ */

#include <matxscript/runtime/unicodelib/py_unicodedata.h>

#include <stdbool.h>
#include <stddef.h>

#include <string>

#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/runtime_port.h>
#include <matxscript/runtime/uchar_util.h>
#include <matxscript/runtime/unicodelib/py_unicodeobject.h>

namespace matxscript {
namespace runtime {

// clang-format off

/* character properties */

typedef struct {
    const unsigned char category;       /* index into
                                           _PyUnicode_CategoryNames */
    const unsigned char combining;      /* combining class value 0 - 255 */
    const unsigned char bidirectional;  /* index into
                                           _PyUnicode_BidirectionalNames */
    const unsigned char mirrored;       /* true if mirrored in bidir mode */
    const unsigned char east_asian_width;       /* index into
                                                   _PyUnicode_EastAsianWidth */
    const unsigned char normalization_quick_check; /* see is_normalized() */
} _PyUnicode_DatabaseRecord;

typedef struct change_record {
    /* sequence of fields should be the same as in merge_old_version */
    const unsigned char bidir_changed;
    const unsigned char category_changed;
    const unsigned char decimal_changed;
    const unsigned char mirrored_changed;
    const unsigned char east_asian_width_changed;
    const double numeric_changed;
} change_record;

/* ------------- Previous-version API ------------------------------------- */
struct UnicodeDataPreviousDBVersion {
    const char *name;
    const change_record* (*getrecord)(Py_UCS4);
    Py_UCS4 (*normalization)(Py_UCS4);
};

}  // namespace runtime
}  // namespace matxscript

/* data file generated by Tools/unicode/makeunicodedata.py */
#include "py_unicodedata_db.h"

namespace matxscript {
namespace runtime {

static const _PyUnicode_DatabaseRecord*
_getrecord_ex(Py_UCS4 code)
{
    int index;
    if (code >= 0x110000)
        index = 0;
    else {
        index = index1[(code>>SHIFT)];
        index = index2[(index<<SHIFT)+(code&((1<<SHIFT)-1))];
    }

    return &_PyUnicode_Database_Records[index];
}

#define get_old_record(self, v)    ((((UnicodeDataPreviousDBVersion*)self)->getrecord)(v))


static UnicodeDataPreviousDBVersion
new_previous_version(const char*name, const change_record* (*getrecord)(Py_UCS4),
                     Py_UCS4 (*normalization)(Py_UCS4))
{
        UnicodeDataPreviousDBVersion self;
        self.name = name;
        self.getrecord = getrecord;
        self.normalization = normalization;
        return self;
}

// clang-format on

/******************************************************************************
 * PyUnicodeData
 *****************************************************************************/

PyUnicodeData::PyUnicodeData(UCD_VERSION ver) : ucd_version_(ver) {
  static auto previous_ucd = new_previous_version("3.2.0", get_change_3_2_0, normalization_3_2_0);
  switch (ver) {
    case UCD_VERSION::VERSION_3_2_0: {
      previous_ucd_ = &previous_ucd;
    } break;
    default: {
      previous_ucd_ = nullptr;
    } break;
  }
}

long PyUnicodeData::decimal(int chr, long* default_value) const {
  int have_old = 0;
  long rc;
  Py_UCS4 c = (Py_UCS4)chr;

  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed == 0) {
      /* unassigned */
      have_old = 1;
      rc = -1;
    } else if (old->decimal_changed != 0xFF) {
      have_old = 1;
      rc = old->decimal_changed;
    }
  }

  if (!have_old)
    rc = _PyUnicode_ToDecimalDigit(c);
  if (rc < 0) {
    if (default_value == NULL) {
      THROW_PY_ValueError("not a decimal");
      return -1;
    } else {
      return *default_value;
    }
  }
  return rc;
}

long PyUnicodeData::digit(int chr, long* default_value) const {
  long rc;
  Py_UCS4 c = (Py_UCS4)chr;
  rc = _PyUnicode_ToDigit(c);
  if (rc < 0) {
    if (default_value == NULL) {
      THROW_PY_ValueError("not a digit");
      return -1;
    } else {
      return *default_value;
    }
  }
  return rc;
}

double PyUnicodeData::numeric(int chr, double* default_value) const {
  int have_old = 0;
  double rc;
  Py_UCS4 c = (Py_UCS4)chr;

  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed == 0) {
      /* unassigned */
      have_old = 1;
      rc = -1.0;
    } else if (old->decimal_changed != 0xFF) {
      have_old = 1;
      rc = old->decimal_changed;
    }
  }

  if (!have_old)
    rc = _PyUnicode_ToNumeric(c);
  if (rc == -1.0) {
    if (default_value == NULL) {
      THROW_PY_ValueError("not a numeric character");
      return rc;
    } else {
      return *default_value;
    }
  }
  return rc;
}

string_view PyUnicodeData::category(int chr) const {
  int index;
  Py_UCS4 c = (Py_UCS4)chr;
  index = (int)_getrecord_ex(c)->category;
  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed != 0xFF)
      index = old->category_changed;
  }
  return _PyUnicode_CategoryNames[index];
}

string_view PyUnicodeData::bidirectional(int chr) const {
  int index;
  Py_UCS4 c = (Py_UCS4)chr;
  index = (int)_getrecord_ex(c)->bidirectional;
  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed == 0)
      index = 0; /* unassigned */
    else if (old->bidir_changed != 0xFF)
      index = old->bidir_changed;
  }
  return _PyUnicode_BidirectionalNames[index];
}

int PyUnicodeData::combining(int chr) const {
  int index;
  Py_UCS4 c = (Py_UCS4)chr;
  index = (int)_getrecord_ex(c)->combining;
  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed == 0)
      index = 0; /* unassigned */
  }
  return index;
}

int PyUnicodeData::mirrored(int chr) const {
  int index;
  Py_UCS4 c = (Py_UCS4)chr;
  index = (int)_getrecord_ex(c)->mirrored;
  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed == 0)
      index = 0; /* unassigned */
    else if (old->mirrored_changed != 0xFF)
      index = old->mirrored_changed;
  }
  return index;
}

string_view PyUnicodeData::east_asian_width(int chr) const {
  int index;
  Py_UCS4 c = (Py_UCS4)chr;
  index = (int)_getrecord_ex(c)->east_asian_width;
  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed == 0)
      index = 0; /* unassigned */
    else if (old->east_asian_width_changed != 0xFF)
      index = old->east_asian_width_changed;
  }
  return _PyUnicode_EastAsianWidthNames[index];
}

String PyUnicodeData::decomposition(int chr) const {
  char decomp[256];
  int code, index, count;
  size_t i;
  unsigned int prefix_index;
  Py_UCS4 c = (Py_UCS4)chr;

  code = (int)c;

  if (previous_ucd_) {
    const change_record* old = get_old_record(previous_ucd_, c);
    if (old->category_changed == 0)
      return ""; /* unassigned */
  }

  if (code < 0 || code >= 0x110000)
    index = 0;
  else {
    index = decomp_index1[(code >> DECOMP_SHIFT)];
    index = decomp_index2[(index << DECOMP_SHIFT) + (code & ((1 << DECOMP_SHIFT) - 1))];
  }

  /* high byte is number of hex bytes (usually one or two), low byte
     is prefix code (from*/
  count = decomp_data[index] >> 8;

  /* XXX: could allocate the PyString up front instead
     (strlen(prefix) + 5 * count + 1 bytes) */

  /* Based on how index is calculated above and decomp_data is generated
     from Tools/unicode/makeunicodedata.py, it should not be possible
     to overflow decomp_prefix. */
  prefix_index = decomp_data[index] & 255;
  assert(prefix_index < MATXSCRIPT_ARRAY_LENGTH(decomp_prefix));

  /* copy prefix */
  i = strlen(decomp_prefix[prefix_index]);
  memcpy(decomp, decomp_prefix[prefix_index], i);

  while (count-- > 0) {
    if (i)
      decomp[i++] = ' ';
    assert(i < sizeof(decomp));
    snprintf(decomp + i, sizeof(decomp) - i, "%04X", decomp_data[++index]);
    i += strlen(decomp + i);
  }
  return String(decomp, i);
}

static void get_decomp_record(
    UnicodeDataPreviousDBVersion* self, Py_UCS4 code, int* index, int* prefix, int* count) {
  if (code >= 0x110000) {
    *index = 0;
  } else if (self && get_old_record(self, code)->category_changed == 0) {
    /* unassigned in old version */
    *index = 0;
  } else {
    *index = decomp_index1[(code >> DECOMP_SHIFT)];
    *index = decomp_index2[(*index << DECOMP_SHIFT) + (code & ((1 << DECOMP_SHIFT) - 1))];
  }

  /* high byte is number of hex bytes (usually one or two), low byte
     is prefix code (from*/
  *count = decomp_data[*index] >> 8;
  *prefix = decomp_data[*index] & 255;

  (*index)++;
}

#define SBase 0xAC00
#define LBase 0x1100
#define VBase 0x1161
#define TBase 0x11A7
#define LCount 19
#define VCount 21
#define TCount 28
#define NCount (VCount * TCount)
#define SCount (LCount * NCount)

static Unicode nfd_nfkd(UnicodeDataPreviousDBVersion* self, const unicode_view& input, int k) {
  ssize_t i, o;
  /* Longest decomposition in Unicode 3.2: U+FDFA */
  Py_UCS4 stack[20];
  ssize_t space, isize;
  int index, prefix, count, stackptr;
  unsigned char prev, cur;
  Unicode result;

  stackptr = 0;
  isize = input.size();
  space = isize;
  /* Overallocate at most 10 characters. */
  if (space > 10) {
    if (space <= MATXSCRIPT_SSIZE_T_MAX - 10)
      space += 10;
  } else {
    space *= 2;
  }
  result.reserve(space);
  i = o = 0;

  while (i < isize) {
    stack[stackptr++] = input[i++];
    while (stackptr) {
      Py_UCS4 code = stack[--stackptr];
      /* Hangul Decomposition. */
      if (SBase <= code && code < (SBase + SCount)) {
        int SIndex = code - SBase;
        int L = LBase + SIndex / NCount;
        int V = VBase + (SIndex % NCount) / TCount;
        int T = TBase + SIndex % TCount;
        result.push_back(L);
        result.push_back(V);
        space -= 2;
        if (T != TBase) {
          result.push_back(T);
          space--;
        }
        continue;
      }
      /* normalization changes */
      if (self) {
        Py_UCS4 value = ((UnicodeDataPreviousDBVersion*)self)->normalization(code);
        if (value != 0) {
          stack[stackptr++] = value;
          continue;
        }
      }

      /* Other decompositions. */
      get_decomp_record(self, code, &index, &prefix, &count);

      /* Copy character if it is not decomposable, or has a
         compatibility decomposition, but we do NFD. */
      if (!count || (prefix && !k)) {
        result.push_back(code);
        space--;
        continue;
      }
      /* Copy decomposition onto the stack, in reverse
         order.  */
      while (count) {
        code = decomp_data[index + (--count)];
        stack[stackptr++] = code;
      }
    }
  }

  /* Sort canonically. */
  i = 0;
  prev = _getrecord_ex(result[i])->combining;
  for (i++; i < result.size(); i++) {
    cur = _getrecord_ex(result[i])->combining;
    if (prev == 0 || cur == 0 || prev <= cur) {
      prev = cur;
      continue;
    }
    /* Non-canonical order. Need to switch *i with previous. */
    o = i - 1;
    while (1) {
      Py_UCS4 tmp = result[o + 1];
      result[o + 1] = result[o];
      result[o] = tmp;
      o--;
      if (o < 0)
        break;
      prev = _getrecord_ex(result[o])->combining;
      if (prev == 0 || prev <= cur)
        break;
    }
    prev = _getrecord_ex(result[i])->combining;
  }
  return result;
}

static int find_nfc_index(UnicodeDataPreviousDBVersion* self, struct reindex* nfc, Py_UCS4 code) {
  unsigned int index;
  for (index = 0; nfc[index].start; index++) {
    unsigned int start = nfc[index].start;
    if (code < start)
      return -1;
    if (code <= start + nfc[index].count) {
      unsigned int delta = code - start;
      return nfc[index].index + delta;
    }
  }
  return -1;
}

static Unicode nfc_nfkc(UnicodeDataPreviousDBVersion* self, const unicode_view& input, int k) {
  const Py_UCS4* data;
  ssize_t i, i1, o, len;
  int f, l, index, index1, comb;
  Py_UCS4 code;
  ssize_t skipped[20];
  int cskipped = 0;
  Unicode result_nfkd;
  Unicode result;

  result_nfkd = nfd_nfkd(self, input, k);
  data = result_nfkd.data();
  len = result_nfkd.size();

  /* We allocate a buffer for the output.
     If we find that we made no changes, we still return
     the NFD result. */
  result.reserve(result_nfkd.size());
  i = o = 0;

again:
  while (i < len) {
    for (index = 0; index < cskipped; index++) {
      if (skipped[index] == i) {
        /* *i character is skipped.
           Remove from list. */
        skipped[index] = skipped[cskipped - 1];
        cskipped--;
        i++;
        goto again; /* continue while */
      }
    }
    /* Hangul Composition. We don't need to check for <LV,T>
       pairs, since we always have decomposed data. */
    code = data[i];
    if (LBase <= code && code < (LBase + LCount) && i + 1 < len && VBase <= data[i + 1] &&
        data[i + 1] < (VBase + VCount)) {
      /* check L character is a modern leading consonant (0x1100 ~ 0x1112)
         and V character is a modern vowel (0x1161 ~ 0x1175). */
      int LIndex, VIndex;
      LIndex = code - LBase;
      VIndex = data[i + 1] - VBase;
      code = SBase + (LIndex * VCount + VIndex) * TCount;
      i += 2;
      if (i < len && TBase < data[i] && data[i] < (TBase + TCount)) {
        /* check T character is a modern trailing consonant
           (0x11A8 ~ 0x11C2). */
        code += data[i] - TBase;
        i++;
      }
      result.push_back(code);
      o++;
      continue;
    }

    /* code is still input[i] here */
    f = find_nfc_index(self, nfc_first, code);
    if (f == -1) {
      result.push_back(code);
      o++;
      i++;
      continue;
    }
    /* Find next unblocked character. */
    i1 = i + 1;
    comb = 0;
    /* output base character for now; might be updated later. */
    result.push_back(data[i]);
    while (i1 < len) {
      Py_UCS4 code1 = data[i1];
      int comb1 = _getrecord_ex(code1)->combining;
      if (comb) {
        if (comb1 == 0)
          break;
        if (comb >= comb1) {
          /* Character is blocked. */
          i1++;
          continue;
        }
      }
      l = find_nfc_index(self, nfc_last, code1);
      /* i1 cannot be combined with i. If i1
         is a starter, we don't need to look further.
         Otherwise, record the combining class. */
      if (l == -1) {
      not_combinable:
        if (comb1 == 0)
          break;
        comb = comb1;
        i1++;
        continue;
      }
      index = f * TOTAL_LAST + l;
      index1 = comp_index[index >> COMP_SHIFT];
      code = comp_data[(index1 << COMP_SHIFT) + (index & ((1 << COMP_SHIFT) - 1))];
      if (code == 0)
        goto not_combinable;

      /* Replace the original character. */
      result[o] = code;
      /* Mark the second character unused. */
      assert(cskipped < 20);
      skipped[cskipped++] = i1;
      i1++;
      f = find_nfc_index(self, nfc_first, result[o]);
      if (f == -1)
        break;
    }
    /* Output character was already written.
       Just advance the indices. */
    o++;
    i++;
  }
  if (o == len) {
    /* No changes. Return original string. */
    return result_nfkd;
  }
  return result;
}

// This needs to match the logic in makeunicodedata.py
// which constructs the quickcheck data.
typedef enum { YES = 0, MAYBE = 1, NO = 2 } QuickcheckResult;

/* Run the Unicode normalization "quickcheck" algorithm.
 *
 * Return YES or NO if quickcheck determines the input is certainly
 * normalized or certainly not, and MAYBE if quickcheck is unable to
 * tell.
 *
 * If `yes_only` is true, then return MAYBE as soon as we determine
 * the answer is not YES.
 *
 * For background and details on the algorithm, see UAX #15:
 *   https://www.unicode.org/reports/tr15/#Detecting_Normalization_Forms
 */
static QuickcheckResult is_normalized_quickcheck(
    UnicodeDataPreviousDBVersion* self, const unicode_view& input, int nfc, int k, bool yes_only) {
  /* An older version of the database is requested, quickchecks must be
     disabled. */
  if (self)
    return NO;

  ssize_t i, len;
  const Py_UCS4* data;
  unsigned char prev_combining = 0;

  /* The two quickcheck bits at this shift have type QuickcheckResult. */
  int quickcheck_shift = (nfc ? 4 : 0) + (k ? 2 : 0);

  QuickcheckResult result = YES; /* certainly normalized, unless we find something */

  i = 0;
  data = input.data();
  len = input.size();
  while (i < len) {
    Py_UCS4 ch = data[i++];
    const _PyUnicode_DatabaseRecord* record = _getrecord_ex(ch);

    unsigned char combining = record->combining;
    if (combining && prev_combining > combining)
      return NO; /* non-canonical sort order, not normalized */
    prev_combining = combining;

    unsigned char quickcheck_whole = record->normalization_quick_check;
    if (yes_only) {
      if (quickcheck_whole & (3 << quickcheck_shift))
        return MAYBE;
    } else {
      switch ((quickcheck_whole >> quickcheck_shift) & 3) {
        case NO:
          return NO;
        case MAYBE:
          result = MAYBE; /* this string might need normalization */
      }
    }
  }
  return result;
}

bool PyUnicodeData::is_normalized(int32_t form, const unicode_view& input) const {
  if (input.size() == 0) {
    /* special case empty input strings. */
    return true;
  }

  int nfc = 0;
  int k = 0;
  QuickcheckResult m;

  Unicode cmp;
  int match = 0;

  if (form == UnicodeNormalForm::NFC) {
    nfc = 1;
  } else if (form == UnicodeNormalForm::NFKC) {
    nfc = 1;
    k = 1;
  } else if (form == UnicodeNormalForm::NFD) {
    /* matches default values for `nfc` and `k` */
  } else if (form == UnicodeNormalForm::NFKD) {
    k = 1;
  } else {
    THROW_PY_ValueError("invalid normalization form");
    return false;
  }

  m = is_normalized_quickcheck(previous_ucd_, input, nfc, k, false);

  if (m == MAYBE) {
    cmp = (nfc ? nfc_nfkc : nfd_nfkd)(previous_ucd_, input, k);
    return input == cmp.view();
  } else {
    return m == YES;
  }
}

Unicode PyUnicodeData::normalize(int32_t form, const unicode_view& input) const {
  if (input.empty()) {
    /* Special case empty input strings, since resizing
       them  later would cause internal errors. */
    return input;
  }

  if (form == UnicodeNormalForm::NFC) {
    if (is_normalized_quickcheck(previous_ucd_, input, 1, 0, true) == YES) {
      return input;
    }
    return nfc_nfkc(previous_ucd_, input, 0);
  }
  if (form == UnicodeNormalForm::NFKC) {
    if (is_normalized_quickcheck(previous_ucd_, input, 1, 1, true) == YES) {
      return input;
    }
    return nfc_nfkc(previous_ucd_, input, 1);
  }
  if (form == UnicodeNormalForm::NFD) {
    if (is_normalized_quickcheck(previous_ucd_, input, 0, 0, true) == YES) {
      return input;
    }
    return nfd_nfkd(previous_ucd_, input, 0);
  }
  if (form == UnicodeNormalForm::NFKD) {
    if (is_normalized_quickcheck(previous_ucd_, input, 0, 1, true) == YES) {
      return input;
    }
    return nfd_nfkd(previous_ucd_, input, 1);
  }
  THROW_PY_ValueError("invalid normalization form");
  return Unicode();
}

}  // namespace runtime
}  // namespace matxscript

// clang-format off

/* -------------------------------------------------------------------- */
/* unicode character name tables */

/* data file generated by Tools/unicode/makeunicodedata.py */
#include "py_unicodename_db.h"

/* -------------------------------------------------------------------- */
/* database code (cut and pasted from the unidb package) */

namespace matxscript {
namespace runtime {

static unsigned long
_gethash(const char *s, int len, int scale)
{
    int i;
    unsigned long h = 0;
    unsigned long ix;
    for (i = 0; i < len; i++) {
        h = (h * scale) + (unsigned char) UCHAR_TOUPPER(UCHAR_MASK(s[i]));
        ix = h & 0xff000000;
        if (ix)
            h = (h ^ ((ix>>24) & 0xff)) & 0x00ffffff;
    }
    return h;
}

static const char * const hangul_syllables[][3] = {
    { "G",  "A",   ""   },
    { "GG", "AE",  "G"  },
    { "N",  "YA",  "GG" },
    { "D",  "YAE", "GS" },
    { "DD", "EO",  "N", },
    { "R",  "E",   "NJ" },
    { "M",  "YEO", "NH" },
    { "B",  "YE",  "D"  },
    { "BB", "O",   "L"  },
    { "S",  "WA",  "LG" },
    { "SS", "WAE", "LM" },
    { "",   "OE",  "LB" },
    { "J",  "YO",  "LS" },
    { "JJ", "U",   "LT" },
    { "C",  "WEO", "LP" },
    { "K",  "WE",  "LH" },
    { "T",  "WI",  "M"  },
    { "P",  "YU",  "B"  },
    { "H",  "EU",  "BS" },
    { 0,    "YI",  "S"  },
    { 0,    "I",   "SS" },
    { 0,    0,     "NG" },
    { 0,    0,     "J"  },
    { 0,    0,     "C"  },
    { 0,    0,     "K"  },
    { 0,    0,     "T"  },
    { 0,    0,     "P"  },
    { 0,    0,     "H"  }
};

/* These ranges need to match makeunicodedata.py:cjk_ranges. */
static int
is_unified_ideograph(Py_UCS4 code)
{
    return
        (0x3400 <= code && code <= 0x4DB5)   || /* CJK Ideograph Extension A */
        (0x4E00 <= code && code <= 0x9FEF)   || /* CJK Ideograph */
        (0x20000 <= code && code <= 0x2A6D6) || /* CJK Ideograph Extension B */
        (0x2A700 <= code && code <= 0x2B734) || /* CJK Ideograph Extension C */
        (0x2B740 <= code && code <= 0x2B81D) || /* CJK Ideograph Extension D */
        (0x2B820 <= code && code <= 0x2CEA1) || /* CJK Ideograph Extension E */
        (0x2CEB0 <= code && code <= 0x2EBEF);   /* CJK Ideograph Extension F */
}

/* macros used to determine if the given code point is in the PUA range that
 * we are using to store aliases and named sequences */
#define IS_ALIAS(cp) ((cp >= aliases_start) && (cp < aliases_end))
#define IS_NAMED_SEQ(cp) ((cp >= named_sequences_start) && \
                          (cp < named_sequences_end))

static int
_getucname(UnicodeDataPreviousDBVersion *self, Py_UCS4 code, char* buffer, int buflen,
           int with_alias_and_seq)
{
    /* Find the name associated with the given code point.
     * If with_alias_and_seq is 1, check for names in the Private Use Area 15
     * that we are using for aliases and named sequences. */
    int offset;
    int i;
    int word;
    const unsigned char* w;

    if (code >= 0x110000)
        return 0;

    /* XXX should we just skip all the code points in the PUAs here? */
    if (!with_alias_and_seq && (IS_ALIAS(code) || IS_NAMED_SEQ(code)))
        return 0;

    if (self) {
        /* in 3.2.0 there are no aliases and named sequences */
        const change_record *old;
        if (IS_ALIAS(code) || IS_NAMED_SEQ(code))
            return 0;
        old = get_old_record(self, code);
        if (old->category_changed == 0) {
            /* unassigned */
            return 0;
        }
    }

    if (SBase <= code && code < SBase+SCount) {
        /* Hangul syllable. */
        int SIndex = code - SBase;
        int L = SIndex / NCount;
        int V = (SIndex % NCount) / TCount;
        int T = SIndex % TCount;

        if (buflen < 27)
            /* Worst case: HANGUL SYLLABLE <10chars>. */
            return 0;
        strcpy(buffer, "HANGUL SYLLABLE ");
        buffer += 16;
        strcpy(buffer, hangul_syllables[L][0]);
        buffer += strlen(hangul_syllables[L][0]);
        strcpy(buffer, hangul_syllables[V][1]);
        buffer += strlen(hangul_syllables[V][1]);
        strcpy(buffer, hangul_syllables[T][2]);
        buffer += strlen(hangul_syllables[T][2]);
        *buffer = '\0';
        return 1;
    }

    if (is_unified_ideograph(code)) {
        if (buflen < 28)
            /* Worst case: CJK UNIFIED IDEOGRAPH-20000 */
            return 0;
        sprintf(buffer, "CJK UNIFIED IDEOGRAPH-%X", code);
        return 1;
    }

    /* get offset into phrasebook */
    offset = phrasebook_offset1[(code>>phrasebook_shift)];
    offset = phrasebook_offset2[(offset<<phrasebook_shift) +
                               (code&((1<<phrasebook_shift)-1))];
    if (!offset)
        return 0;

    i = 0;

    for (;;) {
        /* get word index */
        word = phrasebook[offset] - phrasebook_short;
        if (word >= 0) {
            word = (word << 8) + phrasebook[offset+1];
            offset += 2;
        } else
            word = phrasebook[offset++];
        if (i) {
            if (i > buflen)
                return 0; /* buffer overflow */
            buffer[i++] = ' ';
        }
        /* copy word string from lexicon.  the last character in the
           word has bit 7 set.  the last word in a string ends with
           0x80 */
        w = lexicon + lexicon_offset[word];
        while (*w < 128) {
            if (i >= buflen)
                return 0; /* buffer overflow */
            buffer[i++] = *w++;
        }
        if (i >= buflen)
            return 0; /* buffer overflow */
        buffer[i++] = *w & 127;
        if (*w == 128)
            break; /* end of word */
    }

    return 1;
}

static int
_cmpname(UnicodeDataPreviousDBVersion *self, int code, const char* name, int namelen)
{
    /* check if code corresponds to the given name */
    int i;
    char buffer[NAME_MAXLEN+1];
    if (!_getucname(self, code, buffer, NAME_MAXLEN, 1))
        return 0;
    for (i = 0; i < namelen; i++) {
        if (UCHAR_TOUPPER(UCHAR_MASK(name[i])) != buffer[i])
            return 0;
    }
    return buffer[namelen] == '\0';
}

static void
find_syllable(const char *str, int *len, int *pos, int count, int column)
{
    int i, len1;
    *len = -1;
    for (i = 0; i < count; i++) {
        const char *s = hangul_syllables[i][column];
        len1 = MATXSCRIPT_SAFE_DOWNCAST(strlen(s), size_t, int);
        if (len1 <= *len)
            continue;
        if (strncmp(str, s, len1) == 0) {
            *len = len1;
            *pos = i;
        }
    }
    if (*len == -1) {
        *len = 0;
    }
}

static int
_check_alias_and_seq(unsigned int cp, Py_UCS4* code, int with_named_seq)
{
    /* check if named sequences are allowed */
    if (!with_named_seq && IS_NAMED_SEQ(cp))
        return 0;
    /* if the code point is in the PUA range that we use for aliases,
     * convert it to obtain the right code point */
    if (IS_ALIAS(cp))
        *code = name_aliases[cp-aliases_start];
    else
        *code = cp;
    return 1;
}

static int
_getcode(UnicodeDataPreviousDBVersion* self, const char* name, int namelen, Py_UCS4* code,
         int with_named_seq)
{
    /* Return the code point associated with the given name.
     * Named aliases are resolved too (unless self != NULL (i.e. we are using
     * 3.2.0)).  If with_named_seq is 1, returns the PUA code point that we are
     * using for the named sequence, and the caller must then convert it. */
    unsigned int h, v;
    unsigned int mask = code_size-1;
    unsigned int i, incr;

    /* Check for hangul syllables. */
    if (strncmp(name, "HANGUL SYLLABLE ", 16) == 0) {
        int len, L = -1, V = -1, T = -1;
        const char *pos = name + 16;
        find_syllable(pos, &len, &L, LCount, 0);
        pos += len;
        find_syllable(pos, &len, &V, VCount, 1);
        pos += len;
        find_syllable(pos, &len, &T, TCount, 2);
        pos += len;
        if (L != -1 && V != -1 && T != -1 && pos-name == namelen) {
            *code = SBase + (L*VCount+V)*TCount + T;
            return 1;
        }
        /* Otherwise, it's an illegal syllable name. */
        return 0;
    }

    /* Check for unified ideographs. */
    if (strncmp(name, "CJK UNIFIED IDEOGRAPH-", 22) == 0) {
        /* Four or five hexdigits must follow. */
        v = 0;
        name += 22;
        namelen -= 22;
        if (namelen != 4 && namelen != 5)
            return 0;
        while (namelen--) {
            v *= 16;
            if (*name >= '0' && *name <= '9')
                v += *name - '0';
            else if (*name >= 'A' && *name <= 'F')
                v += *name - 'A' + 10;
            else
                return 0;
            name++;
        }
        if (!is_unified_ideograph(v))
            return 0;
        *code = v;
        return 1;
    }

    /* the following is the same as python's dictionary lookup, with
       only minor changes.  see the makeunicodedata script for more
       details */

    h = (unsigned int) _gethash(name, namelen, code_magic);
    i = (~h) & mask;
    v = code_hash[i];
    if (!v)
        return 0;
    if (_cmpname(self, v, name, namelen))
        return _check_alias_and_seq(v, code, with_named_seq);
    incr = (h ^ (h >> 3)) & mask;
    if (!incr)
        incr = mask;
    for (;;) {
        i = (i + incr) & mask;
        v = code_hash[i];
        if (!v)
            return 0;
        if (_cmpname(self, v, name, namelen))
            return _check_alias_and_seq(v, code, with_named_seq);
        incr = incr << 1;
        if (incr > mask)
            incr = incr ^ code_poly;
    }
}

// clang-format on

String PyUnicodeData::name(int chr, String* default_value) const {
  char name[NAME_MAXLEN + 1];
  Py_UCS4 c = (Py_UCS4)chr;

  if (!_getucname(previous_ucd_, c, name, NAME_MAXLEN, 0)) {
    if (default_value == NULL) {
      THROW_PY_ValueError("no such name");
      return "";
    } else {
      return *default_value;
    }
  }
  return String(name);
}

Unicode PyUnicodeData::lookup(string_view name) const {
  Py_UCS4 code;
  unsigned int index;
  if (name.size() > NAME_MAXLEN) {
    THROW_PY_ValueError("name too long");
    return Unicode();
  }

  if (!_getcode(previous_ucd_, name.data(), (int)name.size(), &code, 1)) {
    THROW_PY_ValueError("undefined character name ", name);
    return Unicode();
  }
  /* check if code is in the PUA range that we use for named sequences
     and convert it */
  if (IS_NAMED_SEQ(code)) {
    index = code - named_sequences_start;
    Unicode result;
    result.reserve(named_sequences[index].seqlen);
    for (int i = 0; i < named_sequences[index].seqlen; ++i) {
      result.push_back(named_sequences[index].seq[i]);
    }
    return result;
  }
  return Unicode({code});
}

}  // namespace runtime
}  // namespace matxscript
