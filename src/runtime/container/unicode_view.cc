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
#include <matxscript/runtime/container/unicode_view.h>

#include <algorithm>
#include <ostream>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

namespace {
// This is significantly faster for case-sensitive matches with very
// few possible matches.  See unit test for benchmarks.
const unicode_view::value_type* memmatch(const unicode_view::value_type* phaystack,
                                         size_t haylen,
                                         const unicode_view::value_type* pneedle,
                                         size_t neelen) {
  if (0 == neelen) {
    return phaystack;  // even if haylen is 0
  }
  if (haylen < neelen)
    return nullptr;
  const unicode_view::value_type* match;
  const unicode_view::value_type* hayend = phaystack + haylen - neelen + 1;
  while ((match = std::char_traits<unicode_view::value_type>::find(
              phaystack, hayend - phaystack, pneedle[0]))) {
    if (std::char_traits<unicode_view::value_type>::compare(match, pneedle, neelen) == 0)
      return match;
    else
      phaystack = match + 1;
  }
  return nullptr;
}

}  // namespace

std::ostream& operator<<(std::ostream& o, unicode_view piece) {
  auto s = UTF8Encode(piece.data(), piece.size());
  o << string_view(s);
  return o;
}

unicode_view::size_type unicode_view::find(unicode_view s, size_type pos) const noexcept {
  if (empty() || pos > length_) {
    if (empty() && pos == 0 && s.empty())
      return 0;
    return npos;
  }
  const value_type* result = memmatch(ptr_ + pos, length_ - pos, s.ptr_, s.length_);
  return result ? result - ptr_ : npos;
}

unicode_view::size_type unicode_view::find(value_type c, size_type pos) const noexcept {
  if (empty() || pos >= length_) {
    return npos;
  }
  const value_type* result = std::char_traits<value_type>::find(ptr_ + pos, length_ - pos, c);
  return result != nullptr ? result - ptr_ : npos;
}

unicode_view::size_type unicode_view::rfind(unicode_view s, size_type pos) const noexcept {
  if (length_ < s.length_)
    return npos;
  if (s.empty())
    return std::min(length_, pos);
  const value_type* last = ptr_ + std::min(length_ - s.length_, pos) + s.length_;
  const value_type* result = std::find_end(ptr_, last, s.ptr_, s.ptr_ + s.length_);
  return result != last ? result - ptr_ : npos;
}

// Search range is [0..pos] inclusive.  If pos == npos, search everything.
unicode_view::size_type unicode_view::rfind(value_type c, size_type pos) const noexcept {
  // Note: memrchr() is not available on Windows.
  if (empty())
    return npos;
  for (size_type i = std::min(pos, length_ - 1);; --i) {
    if (ptr_[i] == c) {
      return i;
    }
    if (i == 0)
      break;
  }
  return npos;
}

unicode_view::size_type unicode_view::find_first_of(unicode_view s, size_type pos) const noexcept {
  if (empty() || s.empty()) {
    return npos;
  }
  // Avoid the cost of LookupTable() for a single-character search.
  if (s.length_ == 1)
    return find_first_of(s.ptr_[0], pos);
  for (size_type i = pos; i < length_; ++i) {
    if (s.find(ptr_[i]) != npos) {
      return i;
    }
  }
  return npos;
}

unicode_view::size_type unicode_view::find_first_not_of(unicode_view s,
                                                        size_type pos) const noexcept {
  if (empty())
    return npos;
  // Avoid the cost of LookupTable() for a single-character search.
  if (s.length_ == 1)
    return find_first_not_of(s.ptr_[0], pos);

  for (size_type i = pos; i < length_; ++i) {
    if (s.find(ptr_[i]) == npos) {
      return i;
    }
  }
  return npos;
}

unicode_view::size_type unicode_view::find_first_not_of(value_type c,
                                                        size_type pos) const noexcept {
  if (empty())
    return npos;
  for (; pos < length_; ++pos) {
    if (ptr_[pos] != c) {
      return pos;
    }
  }
  return npos;
}

unicode_view::size_type unicode_view::find_last_of(unicode_view s, size_type pos) const noexcept {
  if (empty() || s.empty())
    return npos;
  // Avoid the cost of LookupTable() for a single-character search.
  if (s.length_ == 1)
    return find_last_of(s.ptr_[0], pos);
  for (size_type i = std::min(pos, length_ - 1);; --i) {
    if (s.find(ptr_[i]) != npos) {
      return i;
    }
    if (i == 0)
      break;
  }
  return npos;
}

unicode_view::size_type unicode_view::find_last_not_of(unicode_view s,
                                                       size_type pos) const noexcept {
  if (empty())
    return npos;
  size_type i = std::min(pos, length_ - 1);
  if (s.empty())
    return i;
  // Avoid the cost of LookupTable() for a single-character search.
  if (s.length_ == 1)
    return find_last_not_of(s.ptr_[0], pos);
  for (;; --i) {
    if (s.find(ptr_[i]) == npos) {
      return i;
    }
    if (i == 0)
      break;
  }
  return npos;
}

unicode_view::size_type unicode_view::find_last_not_of(value_type c, size_type pos) const noexcept {
  if (empty())
    return npos;
  size_type i = std::min(pos, length_ - 1);
  for (;; --i) {
    if (ptr_[i] != c) {
      return i;
    }
    if (i == 0)
      break;
  }
  return npos;
}

constexpr unicode_view::size_type unicode_view::npos;
constexpr unicode_view::size_type unicode_view::kMaxSize;

}  // namespace runtime
}  // namespace matxscript
