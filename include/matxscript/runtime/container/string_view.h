// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from abseil-cpp.
 * https://github.com/abseil/abseil-cpp/blob/master/absl/strings/string_view.h
 *
 * Copyright 2017 The Abseil Authors.
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

// -----------------------------------------------------------------------------
// File: string_view.h
// -----------------------------------------------------------------------------
//
// This file contains the definition of the `absl::string_view` class. A
// `string_view` points to a contiguous span of characters, often part or all of
// another `std::string`, double-quoted string literal, character array, or even
// another `string_view`.
//
// This `absl::string_view` abstraction is designed to be a drop-in
// replacement for the C++17 `std::string_view` abstraction.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#include <matxscript/runtime/bytes_hash.h>
#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {

class string_view {
 public:
  using traits_type = std::char_traits<char>;
  using value_type = char;
  using pointer = char*;
  using const_pointer = const char*;
  using reference = char&;
  using const_reference = const char&;
  using const_iterator = const char*;
  using iterator = const_iterator;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = const_reverse_iterator;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  static constexpr size_type npos = static_cast<size_type>(-1);
  static constexpr int32_t _UNKNOWN_CATEGORY = INT8_MIN;

  // Null `string_view` constructor
  constexpr string_view() noexcept : ptr_(nullptr), length_(0), category_(_UNKNOWN_CATEGORY) {
  }

  // Implicit constructors

  template <typename Allocator>
  string_view(  // NOLINT(runtime/explicit)
      const std::basic_string<char, std::char_traits<char>, Allocator>& str) noexcept
      // This is implemented in terms of `string_view(p, n)` so `str.size()`
      // doesn't need to be reevaluated after `ptr_` is set.
      : string_view(str.data(), str.size()) {
  }

  // Implicit constructor of a `string_view` from NUL-terminated `str`. When
  // accepting possibly null strings, use `absl::NullSafeStringView(str)`
  // instead (see below).
  constexpr string_view(const char* str) noexcept  // NOLINT(runtime/explicit)
      : ptr_(str), length_(str ? (traits_type::length(str)) : 0), category_(_UNKNOWN_CATEGORY) {
  }

  // Implicit constructor of a `string_view` from a `const char*` and length.
  constexpr string_view(const char* data, size_type len) noexcept
      : ptr_(data), length_(len), category_(_UNKNOWN_CATEGORY) {
  }

  // only for String
  constexpr explicit string_view(const char* data, size_type len, int32_t category) noexcept
      : ptr_(data), length_(len), category_(category) {
  }

  // NOTE: Harmlessly omitted to work around gdb bug.
  //   constexpr string_view(const string_view&) noexcept = default;
  //   string_view& operator=(const string_view&) noexcept = default;

  // Iterators

  // string_view::begin()
  //
  // Returns an iterator pointing to the first character at the beginning of the
  // `string_view`, or `end()` if the `string_view` is empty.
  constexpr const_iterator begin() const noexcept {
    return ptr_;
  }

  // string_view::end()
  //
  // Returns an iterator pointing just beyond the last character at the end of
  // the `string_view`. This iterator acts as a placeholder; attempting to
  // access it results in undefined behavior.
  constexpr const_iterator end() const noexcept {
    return ptr_ + length_;
  }

  // string_view::cbegin()
  //
  // Returns a const iterator pointing to the first character at the beginning
  // of the `string_view`, or `end()` if the `string_view` is empty.
  constexpr const_iterator cbegin() const noexcept {
    return begin();
  }

  // string_view::cend()
  //
  // Returns a const iterator pointing just beyond the last character at the end
  // of the `string_view`. This pointer acts as a placeholder; attempting to
  // access its element results in undefined behavior.
  constexpr const_iterator cend() const noexcept {
    return end();
  }

  // string_view::rbegin()
  //
  // Returns a reverse iterator pointing to the last character at the end of the
  // `string_view`, or `rend()` if the `string_view` is empty.
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  // string_view::rend()
  //
  // Returns a reverse iterator pointing just before the first character at the
  // beginning of the `string_view`. This pointer acts as a placeholder;
  // attempting to access its element results in undefined behavior.
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }

  // string_view::crbegin()
  //
  // Returns a const reverse iterator pointing to the last character at the end
  // of the `string_view`, or `crend()` if the `string_view` is empty.
  const_reverse_iterator crbegin() const noexcept {
    return rbegin();
  }

  // string_view::crend()
  //
  // Returns a const reverse iterator pointing just before the first character
  // at the beginning of the `string_view`. This pointer acts as a placeholder;
  // attempting to access its element results in undefined behavior.
  const_reverse_iterator crend() const noexcept {
    return rend();
  }

  // Capacity Utilities

  // string_view::size()
  //
  // Returns the number of characters in the `string_view`.
  constexpr size_type size() const noexcept {
    return length_;
  }

  // string_view::length()
  //
  // Returns the number of characters in the `string_view`. Alias for `size()`.
  constexpr size_type length() const noexcept {
    return size();
  }

  // string_view::max_size()
  //
  // Returns the maximum number of characters the `string_view` can hold.
  constexpr size_type max_size() const noexcept {
    return kMaxSize;
  }

  // string_view::empty()
  //
  // Checks if the `string_view` is empty (refers to no characters).
  constexpr bool empty() const noexcept {
    return length_ == 0;
  }

  // string_view::operator[]
  //
  // Returns the ith element of the `string_view` using the array operator.
  // Note that this operator does not perform any bounds checking.
  constexpr const_reference operator[](size_type i) const noexcept {
    return MATXSCRIPT_ASSERT(i < size()), ptr_[i];
  }

  // string_view::at()
  //
  // Returns the ith element of the `string_view`. Bounds checking is performed,
  // and an exception of type `std::out_of_range` will be thrown on invalid
  // access.
  constexpr const_reference at(size_type i) const {
    return MATXSCRIPT_LIKELY(i < size()) ? ptr_[i] : throw std::out_of_range("string_view::at"),
           ptr_[i];
  }

  // string_view::front()
  //
  // Returns the first element of a `string_view`.
  constexpr const_reference front() const noexcept {
    return MATXSCRIPT_ASSERT(!empty()), ptr_[0];
  }

  // string_view::back()
  //
  // Returns the last element of a `string_view`.
  constexpr const_reference back() const noexcept {
    return MATXSCRIPT_ASSERT(!empty()), ptr_[size() - 1];
  }

  // string_view::data()
  //
  // Returns a pointer to the underlying character array (which is of course
  // stored elsewhere). Note that `string_view::data()` may contain embedded nul
  // characters, but the returned buffer may or may not be NUL-terminated;
  // therefore, do not pass `data()` to a routine that expects a NUL-terminated
  // std::string.
  constexpr const_pointer data() const noexcept {
    return ptr_;
  }

  // Modifiers

  // string_view::remove_prefix()
  //
  // Removes the first `n` characters from the `string_view`. Note that the
  // underlying std::string is not changed, only the view.
  void remove_prefix(size_type n) noexcept {
    MATXSCRIPT_ASSERT(n <= length_);
    ptr_ += n;
    length_ -= n;
  }

  // string_view::remove_suffix()
  //
  // Removes the last `n` characters from the `string_view`. Note that the
  // underlying std::string is not changed, only the view.
  void remove_suffix(size_type n) noexcept {
    MATXSCRIPT_ASSERT(n <= length_);
    length_ -= n;
  }

  // string_view::swap()
  //
  // Swaps this `string_view` with another `string_view`.
  void swap(string_view& s) noexcept {
    auto t = *this;
    *this = s;
    s = t;
  }

  // Explicit conversion operators

  // Converts to `std::basic_string`.
  template <typename A>
  explicit operator std::basic_string<char, traits_type, A>() const {
    if (!data())
      return {};
    return std::basic_string<char, traits_type, A>(data(), size());
  }

  // string_view::copy()
  //
  // Copies the contents of the `string_view` at offset `pos` and length `n`
  // into `buf`.
  size_type copy(char* buf, size_type n, size_type pos = 0) const {
    if (MATXSCRIPT_UNLIKELY(pos > length_)) {
      throw std::out_of_range("string_view::copy");
    }
    size_type rlen = (std::min)(length_ - pos, n);
    if (rlen > 0) {
      const char* start = ptr_ + pos;
      traits_type::copy(buf, start, rlen);
    }
    return rlen;
  }

  // string_view::substr()
  //
  // Returns a "substring" of the `string_view` (at offset `pos` and length
  // `n`) as another string_view. This function throws `std::out_of_bounds` if
  // `pos > size`.
  string_view substr(size_type pos, size_type n = npos) const {
    return MATXSCRIPT_UNLIKELY(pos > length_)
               ? (throw std::out_of_range("string_view::substr"), string_view{})
               : string_view(ptr_ + pos, Min(n, length_ - pos));
  }

  // Nonstandard
  string_view SubStrNoCheck(size_type pos, size_type n = npos) const noexcept {
    return string_view(ptr_ + pos, Min(n, length_ - pos));
  }

  // string_view::compare()
  //
  // Performs a lexicographical comparison between the `string_view` and
  // another `absl::string_view`, returning -1 if `this` is less than, 0 if
  // `this` is equal to, and 1 if `this` is greater than the passed std::string
  // view. Note that in the case of data equality, a further comparison is made
  // on the respective sizes of the two `string_view`s to determine which is
  // smaller, equal, or greater.
  constexpr int compare(string_view x) const noexcept {
    return CompareImpl(
        length_,
        x.length_,
        length_ == 0 || x.length_ == 0
            ? 0
            : traits_type::compare(ptr_, x.ptr_, length_ < x.length_ ? length_ : x.length_));
  }

  // Overload of `string_view::compare()` for comparing a substring of the
  // 'string_view` and another `absl::string_view`.
  int compare(size_type pos1, size_type count1, string_view v) const {
    return substr(pos1, count1).compare(v);
  }

  // Overload of `string_view::compare()` for comparing a substring of the
  // `string_view` and a substring of another `absl::string_view`.
  int compare(
      size_type pos1, size_type count1, string_view v, size_type pos2, size_type count2) const {
    return substr(pos1, count1).compare(v.substr(pos2, count2));
  }

  // Overload of `string_view::compare()` for comparing a `string_view` and a
  // a different  C-style std::string `s`.
  int compare(const char* s) const noexcept {
    return compare(string_view(s));
  }

  // Overload of `string_view::compare()` for comparing a substring of the
  // `string_view` and a different std::string C-style std::string `s`.
  int compare(size_type pos1, size_type count1, const char* s) const {
    return substr(pos1, count1).compare(string_view(s));
  }

  // Overload of `string_view::compare()` for comparing a substring of the
  // `string_view` and a substring of a different C-style std::string `s`.
  int compare(size_type pos1, size_type count1, const char* s, size_type count2) const {
    return substr(pos1, count1).compare(string_view(s, count2));
  }

  // Find Utilities

  // string_view::find()
  //
  // Finds the first occurrence of the substring `s` within the `string_view`,
  // returning the position of the first character's match, or `npos` if no
  // match was found.
  size_type find(string_view s, size_type pos = 0) const noexcept;

  // Overload of `string_view::find()` for finding the given character `c`
  // within the `string_view`.
  size_type find(char c, size_type pos = 0) const noexcept;

  // string_view::rfind()
  //
  // Finds the last occurrence of a substring `s` within the `string_view`,
  // returning the position of the first character's match, or `npos` if no
  // match was found.
  size_type rfind(string_view s, size_type pos = npos) const noexcept;

  // Overload of `string_view::rfind()` for finding the last given character `c`
  // within the `string_view`.
  size_type rfind(char c, size_type pos = npos) const noexcept;

  // string_view::find_first_of()
  //
  // Finds the first occurrence of any of the characters in `s` within the
  // `string_view`, returning the start position of the match, or `npos` if no
  // match was found.
  size_type find_first_of(string_view s, size_type pos = 0) const noexcept;

  // Overload of `string_view::find_first_of()` for finding a character `c`
  // within the `string_view`.
  size_type find_first_of(char c, size_type pos = 0) const noexcept {
    return find(c, pos);
  }

  // string_view::find_last_of()
  //
  // Finds the last occurrence of any of the characters in `s` within the
  // `string_view`, returning the start position of the match, or `npos` if no
  // match was found.
  size_type find_last_of(string_view s, size_type pos = npos) const noexcept;

  // Overload of `string_view::find_last_of()` for finding a character `c`
  // within the `string_view`.
  size_type find_last_of(char c, size_type pos = npos) const noexcept {
    return rfind(c, pos);
  }

  // string_view::find_first_not_of()
  //
  // Finds the first occurrence of any of the characters not in `s` within the
  // `string_view`, returning the start position of the first non-match, or
  // `npos` if no non-match was found.
  size_type find_first_not_of(string_view s, size_type pos = 0) const noexcept;

  // Overload of `string_view::find_first_not_of()` for finding a character
  // that is not `c` within the `string_view`.
  size_type find_first_not_of(char c, size_type pos = 0) const noexcept;

  // string_view::find_last_not_of()
  //
  // Finds the last occurrence of any of the characters not in `s` within the
  // `string_view`, returning the start position of the last non-match, or
  // `npos` if no non-match was found.
  size_type find_last_not_of(string_view s, size_type pos = npos) const noexcept;

  // Overload of `string_view::find_last_not_of()` for finding a character
  // that is not `c` within the `string_view`.
  size_type find_last_not_of(char c, size_type pos = npos) const noexcept;

  // only for Any Converter
  constexpr int32_t category() const noexcept {
    return category_;
  }

 private:
  static constexpr size_type kMaxSize = (std::numeric_limits<difference_type>::max)();

  static constexpr size_t Min(size_type length_a, size_type length_b) noexcept {
    return length_a < length_b ? length_a : length_b;
  }

  static constexpr int CompareImpl(size_type length_a,
                                   size_type length_b,
                                   int compare_result) noexcept {
    return compare_result == 0
               ? static_cast<int>(length_a > length_b) - static_cast<int>(length_a < length_b)
               : static_cast<int>(compare_result > 0) - static_cast<int>(compare_result < 0);
  }

  const char* ptr_;
  size_type length_;
  int32_t category_;
};

// This large function is defined inline so that in a fairly common case where
// one of the arguments is a literal, the compiler can elide a lot of the
// following comparisons.
inline constexpr bool operator==(string_view x, string_view y) noexcept {
  return x.size() == y.size() &&
         (x.empty() || string_view::traits_type::compare(x.data(), y.data(), x.size()) == 0);
}

inline constexpr bool operator!=(string_view x, string_view y) noexcept {
  return !(x == y);
}

inline constexpr bool operator<(string_view x, string_view y) noexcept {
  return x.compare(y) < 0;
}

inline constexpr bool operator>(string_view x, string_view y) noexcept {
  return y < x;
}

inline constexpr bool operator<=(string_view x, string_view y) noexcept {
  return !(y < x);
}

inline constexpr bool operator>=(string_view x, string_view y) noexcept {
  return !(x < y);
}

// IO Insertion Operator
std::ostream& operator<<(std::ostream& o, string_view piece);

// ClippedSubstr()
//
// Like `s.substr(pos, n)`, but clips `pos` to an upper bound of `s.size()`.
// Provided because std::string_view::substr throws if `pos > size()`
inline string_view ClippedSubstr(string_view s, size_t pos, size_t n = string_view::npos) {
  pos = (std::min)(pos, static_cast<size_t>(s.size()));
  return s.substr(pos, n);
}

struct std_string_hash {
  std::size_t operator()(const ::std::string& str) const {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
  std::size_t operator()(::matxscript::runtime::string_view str) const {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
  std::size_t operator()(const char* str) const {
    return operator()(::matxscript::runtime::string_view(str));
  }
};

struct std_string_equal_to {
  bool operator()(const ::std::string& a, const ::std::string& b) const {
    return a == b;
  }
  bool operator()(const ::std::string& a, ::matxscript::runtime::string_view b) const {
    return a == b;
  }
  bool operator()(::matxscript::runtime::string_view a, const ::std::string& b) const {
    return a == b;
  }
  bool operator()(const ::std::string& a, const char* b) const {
    return operator()(a, ::matxscript::runtime::string_view(b));
  }
  bool operator()(const char* a, const ::std::string& b) const {
    return operator()(::matxscript::runtime::string_view(a), b);
  }
};

}  // namespace runtime
}  // namespace matxscript

namespace std {

template <>
struct hash<::matxscript::runtime::string_view> {
  std::size_t operator()(::matxscript::runtime::string_view str) const {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
  std::size_t operator()(const ::std::string& str) const {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
  std::size_t operator()(const char* str) const {
    return operator()(::matxscript::runtime::string_view(str));
  }
};

template <>
struct equal_to<::matxscript::runtime::string_view> {
  bool operator()(::matxscript::runtime::string_view a,
                  ::matxscript::runtime::string_view b) const {
    return a == b;
  }
  bool operator()(const ::std::string& a, ::matxscript::runtime::string_view b) const {
    return a == b;
  }
  bool operator()(::matxscript::runtime::string_view a, const ::std::string& b) const {
    return a == b;
  }
  bool operator()(::matxscript::runtime::string_view a, const char* b) const {
    return operator()(a, ::matxscript::runtime::string_view(b));
  }
  bool operator()(const char* a, ::matxscript::runtime::string_view b) const {
    return operator()(::matxscript::runtime::string_view(a), b);
  }
};

}  // namespace std
