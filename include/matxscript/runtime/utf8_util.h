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

#include <string>

#include <matxscript/runtime/unicodelib/unicode_ops.h>

namespace matxscript {
namespace runtime {

class String;
class Unicode;

static constexpr const short UTF8_BYTE_LEN[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};

static inline constexpr short OneCharLen(const char* src) noexcept {
  return UTF8_BYTE_LEN[(*((const unsigned char*)src)) >> 4];
}

extern size_t UTF8CharCounts(string_view str) noexcept;
static inline size_t UTF8CharCounts(const char* s, size_t len) noexcept {
  return UTF8CharCounts(string_view(s, len));
}
extern Unicode UTF8Decode(string_view input);
extern Unicode UTF8Decode(const char* s, size_t len);
extern String UTF8Encode(unicode_view input);
extern String UTF8Encode(const uint32_t* s, size_t len);
extern String UTF8Encode(const char32_t* s, size_t len);

extern String UTF8DoLower(string_view input);
extern String UTF8DoUpper(string_view input);
extern bool UTF8IsDigit(string_view input);
extern bool UTF8IsAlpha(string_view input);

extern String AsciiDoLower(string_view sv);
extern String AsciiDoUpper(string_view sv);

static inline bool AsciiIsDigit(string_view input) noexcept {
  return std::all_of(input.begin(), input.end(), [](char c) { return c >= '0' && c <= '9'; });
}

static inline bool AsciiIsAlpha(string_view input) noexcept {
  return std::all_of(input.begin(), input.end(), [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
  });
}

}  // namespace runtime
}  // namespace matxscript
