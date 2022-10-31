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
#include <matxscript/runtime/utf8_util.h>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/utf8/decoders.h>
#include <matxscript/runtime/utf8/encoders.h>

namespace matxscript {
namespace runtime {

Unicode UTF8Decode(string_view input) {
  return UTF8Decode(input.data(), input.length());
}

Unicode UTF8Decode(const char* s_ptr, size_t len) {
  constexpr int SMALL_BUFFER_SIZE = 64;
  auto* s_u_ptr = (const unsigned char*)(s_ptr);
  if (len <= SMALL_BUFFER_SIZE) {  // small string
    char32_t buffer[SMALL_BUFFER_SIZE];
    auto size = utf8_details::GreedyTableDecoder::Convert(s_u_ptr, s_u_ptr + len, buffer);
    return Unicode(buffer, size);
  } else {
    // for avoid allocate memory
    auto unit_size = utf8_details::GreedyTableDecoder::CountUnitSize(s_u_ptr, s_u_ptr + len);
    Unicode::ContainerType unicodes(unit_size, Unicode::ContainerType::NoInit{});
    auto size = utf8_details::GreedyTableDecoder::Convert(s_u_ptr, s_u_ptr + len, unicodes.data());
    assert(size == unit_size);
    return Unicode(std::move(unicodes));
  }
}

String UTF8Encode(unicode_view input) {
  return UTF8Encode(input.data(), input.length());
}

String UTF8Encode(const uint32_t* s, size_t len) {
  auto bytes_size = utf8_details::GreedyCountBytesSize(s, s + len);
  String::ContainerType bytes(bytes_size, String::ContainerType::NoInit{});
  auto size = utf8_details::GreedyEncoder(s, s + len, (unsigned char*)(bytes.data()));
  assert(size == bytes_size);
  return String(std::move(bytes));
}

String UTF8Encode(const char32_t* s, size_t len) {
  return UTF8Encode(reinterpret_cast<const uint32_t*>(s), len);
}

size_t UTF8CharCounts(string_view str) noexcept {
  size_t count = 0;
  const char* start = str.data();
  const char* end = str.data() + str.size();
  while (start < end) {
    int char_length = OneCharLen(start);
    start += char_length;
    ++count;
  }
  return count;
}

String UTF8DoLower(string_view input) {
  auto codepoints_raw = UTF8Decode(input);
  auto codepoints_lower = py_unicode_do_lower(codepoints_raw);
  return UTF8Encode(codepoints_lower);
}

String UTF8DoUpper(string_view input) {
  auto codepoints_raw = UTF8Decode(input);
  auto codepoints_lower = py_unicode_do_upper(codepoints_raw);
  return UTF8Encode(codepoints_lower);
}

bool UTF8IsDigit(string_view input) {
  auto codepoints = UTF8Decode(input);
  return py_unicode_isdigit(codepoints);
}

bool UTF8IsAlpha(string_view input) {
  auto codepoints = UTF8Decode(input);
  return py_unicode_isalpha(codepoints);
}

String AsciiDoLower(string_view sv) {
  String::ContainerType buffer(sv.size(), String::ContainerType::NoInit{});
  auto* s_ptr = buffer.data();
  for (auto it = sv.begin(); it < sv.end(); ++it) {
    char c = *it;
    if (c <= 'Z' && c >= 'A') {
      *s_ptr = static_cast<char>(c - 'A' + 'a');
    } else {
      *s_ptr = c;
    }
    ++s_ptr;
  }
  return String(std::move(buffer));
}

String AsciiDoUpper(string_view sv) {
  String::ContainerType buffer(sv.size(), String::ContainerType::NoInit{});
  auto* s_ptr = buffer.data();
  for (auto it = sv.begin(); it < sv.end(); ++it) {
    auto c = *it;
    if (c <= 'z' && c >= 'a') {
      *s_ptr = static_cast<char>(c - 'a' + 'A');
    } else {
      *s_ptr = c;
    }
    ++s_ptr;
  }
  return String(std::move(buffer));
}

}  // namespace runtime
}  // namespace matxscript
