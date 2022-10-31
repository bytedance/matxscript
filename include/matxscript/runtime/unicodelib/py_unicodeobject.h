// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
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

#include <cstdint>
#include <string>

#include <matxscript/runtime/container/string_view.h>

namespace matxscript {
namespace runtime {

/* Py_UCS4 and Py_UCS2 are typedefs for the respective
   unicode representations. */
typedef char32_t Py_UCS4;
typedef char16_t Py_UCS2;
typedef unsigned char Py_UCS1;

typedef std::basic_string<Py_UCS4> unicode_string;

extern int _PyUnicode_ToLowerFull(Py_UCS4 ch, Py_UCS4* res) noexcept;
extern int _PyUnicode_ToUpperFull(Py_UCS4 ch, Py_UCS4* res) noexcept;
extern int _PyUnicode_IsCaseIgnorable(Py_UCS4 ch) noexcept;
extern int _PyUnicode_IsCased(Py_UCS4 ch) noexcept;
extern int _PyUnicode_IsDigit(Py_UCS4 ch) noexcept;
extern int _PyUnicode_ToDigit(Py_UCS4 ch) noexcept;
extern int _PyUnicode_IsAlpha(Py_UCS4 ch) noexcept;
extern int _PyUnicode_ToDecimalDigit(Py_UCS4 ch) noexcept;
extern double _PyUnicode_ToNumeric(Py_UCS4 ch) noexcept;
extern int _PyUnicode_IsWhitespace(const Py_UCS4 ch) noexcept;

/* --- Constants ---------------------------------------------------------- */

/* This Unicode character will be used as replacement character during
   decoding if the errors argument is set to "replace". Note: the
   Unicode character U+FFFD is the official REPLACEMENT CHARACTER in
   Unicode 3.0. */

#define PY_UNICODE_REPLACEMENT_CHARACTER ((Py_UCS4)0xFFFD)

/* === Public API ========================================================= */

/* Similar to PyUnicode_FromUnicode(), but u points to UTF-8 encoded bytes */
unicode_string PyUnicode_FromStringAndSize(const char* u, /* UTF-8 encoded string */
                                           int64_t size   /* size of buffer */
);

/* Similar to PyUnicode_FromUnicode(), but u points to null-terminated
   UTF-8 encoded bytes.  The size is determined with strlen(). */
unicode_string PyUnicode_FromString(const char* u /* UTF-8 encoded string */
);

/* Decode obj to a Unicode object.

   bytes, bytearray and other bytes-like objects are decoded according to the
   given encoding and error handler. The encoding and error handler can be
   NULL to have the interface use UTF-8 and "strict".

   All other objects (including Unicode objects) raise an exception.

   The API returns NULL in case of an error. The caller is responsible
   for decref'ing the returned objects.

*/

unicode_string PyUnicode_FromEncodedObject(string_view obj,      /* Bytes */
                                           const char* encoding, /* encoding */
                                           const char* errors    /* error handling */
);

unicode_string PyUnicode_FromFormatV(const char* format, /* ASCII-encoded string  */
                                     va_list vargs);
unicode_string PyUnicode_FromFormat(const char* format, /* ASCII-encoded string  */
                                    ...);

/* --- Unicode ordinals --------------------------------------------------- */

/* Create a Unicode Object from the given Unicode code point ordinal.

   The ordinal must be in range(0x110000). A ValueError is
   raised in case it is not.

*/

unicode_string PyUnicode_FromOrdinal(int ordinal);

/* === Builtin Codecs =====================================================

   Many of these APIs take two arguments encoding and errors. These
   parameters encoding and errors have the same semantics as the ones
   of the builtin str() API.

   Setting encoding to NULL causes the default encoding (UTF-8) to be used.

   Error handling is set by errors which may also be set to NULL
   meaning to use the default handling defined for the codec. Default
   error handling for all builtin codecs is "strict" (ValueErrors are
   raised).

   The codecs all use a similar interface. Only deviation from the
   generic ones are documented.

*/

/* --- Manage the default encoding ---------------------------------------- */

/* Returns "utf-8".  */
string_view PyUnicode_GetDefaultEncoding(void);

/* --- Generic Codecs ----------------------------------------------------- */

/* Create a Unicode object by decoding the encoded string s of the
   given size. */

unicode_string PyUnicode_Decode(const char* s,        /* encoded string */
                                ssize_t size,         /* size of buffer */
                                const char* encoding, /* encoding */
                                const char* errors    /* error handling */
);

/* Encodes a Unicode object and returns the result as Python string
   object. */

std::string PyUnicode_AsEncodedString(const unicode_string& unicode, /* Unicode object */
                                      const char* encoding,          /* encoding */
                                      const char* errors             /* error handling */
);

/* Build an encoding map. */
// TODO(matx-team): fix same as include/unicodeobject.h

}  // namespace runtime
}  // namespace matxscript
