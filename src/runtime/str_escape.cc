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
#include <matxscript/runtime/str_escape.h>

namespace matxscript {
namespace runtime {

static const char HexDigits[16] = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

/*!
 * \brief Create a String with escape.
 * \param data The data
 * \param size The size of the string.
 * \param use_octal_escape True to use octal escapes instead of hex. If producing C
 *      strings, use octal escapes to avoid ambiguously-long hex escapes.
 * \return the Result string.
 */
String BytesEscape(const char* data, size_t size, bool use_octal_escape) {
  String result;
  result.reserve(4 * size);
  for (size_t i = 0; i < size; ++i) {
    unsigned char c = data[i];
    if (c >= ' ' && c <= '~' && c != '\\' && c != '"') {
      result.push_back(c);
    } else {
      result.push_back('\\');
      switch (c) {
        case '"':
          result.push_back('"');
          break;
        case '\\':
          result.push_back('\\');
          break;
        case '\t':
          result.push_back('t');
          break;
        case '\r':
          result.push_back('r');
          break;
        case '\n':
          result.push_back('n');
          break;
        case '\v':
          result.push_back('v');
          break;
        case '\b':
          result.push_back('b');
          break;
        case '\f':
          result.push_back('f');
          break;
        case '\a':
          result.push_back('a');
          break;
        case '\?':
          result.push_back('?');
          break;
        default:
          if (use_octal_escape) {
            result.push_back(static_cast<unsigned char>('0' + ((c >> 6) & 0x03)));
            result.push_back(static_cast<unsigned char>('0' + ((c >> 3) & 0x07)));
            result.push_back(static_cast<unsigned char>('0' + (c & 0x07)));
          } else {
            result.push_back('x');
            result.push_back(HexDigits[c >> 4]);
            result.push_back(HexDigits[c & 0xf]);
          }
      }
    }
  }
  return result;
}

String UnicodeEscape(const char32_t* data, size_t size) {
  runtime::String result;
  result.reserve(2 + size * 12);
  for (size_t i = 0; i < size; ++i) {
    char32_t codepoint = data[i];
    if (static_cast<unsigned>(codepoint) >= 0x80) {
      // Unicode escaping
      result.push_back('\\');
      if (codepoint <= 0xD7FF || (codepoint >= 0xE000 && codepoint <= 0xFFFF)) {
        result.push_back('u');
        result.push_back(HexDigits[(codepoint >> 12) & 15]);
        result.push_back(HexDigits[(codepoint >> 8) & 15]);
        result.push_back(HexDigits[(codepoint >> 4) & 15]);
        result.push_back(HexDigits[(codepoint)&15]);
      } else {
        MXCHECK(codepoint >= 0x010000 && codepoint <= 0x10FFFF);
        result.push_back('U');
        result.push_back(HexDigits[(codepoint >> 28) & 15]);
        result.push_back(HexDigits[(codepoint >> 24) & 15]);
        result.push_back(HexDigits[(codepoint >> 20) & 15]);
        result.push_back(HexDigits[(codepoint >> 16) & 15]);
        result.push_back(HexDigits[(codepoint >> 12) & 15]);
        result.push_back(HexDigits[(codepoint >> 8) & 15]);
        result.push_back(HexDigits[(codepoint >> 4) & 15]);
        result.push_back(HexDigits[(codepoint)&15]);
      }
    } else {
      auto c = static_cast<unsigned char>(codepoint);
      result.push_back('\\');
      result.push_back('u');
      result.push_back('0');
      result.push_back('0');
      result.push_back(HexDigits[c >> 4]);
      result.push_back(HexDigits[c & 0xF]);
    }
  }
  return result;
}

}  // namespace runtime
}  // namespace matxscript
