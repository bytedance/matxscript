// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement:
 * The structure of the unicode_view is inspired by abseil-cpp.
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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iosfwd>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#include <matxscript/runtime/runtime_port.h>
#include <matxscript/runtime/unicodelib/py_unicodeobject.h>

namespace matxscript {
namespace runtime {

class unicode_view {
 public:
  using traits_type = std::char_traits<Py_UCS4>;
  using value_type = Py_UCS4;
  using pointer = Py_UCS4*;
  using const_pointer = const Py_UCS4*;
  using reference = Py_UCS4&;
  using const_reference = const Py_UCS4&;
  using const_iterator = const Py_UCS4*;
  using iterator = const_iterator;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = const_reverse_iterator;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;

  static constexpr size_type npos = static_cast<size_type>(-1);
  static constexpr int32_t _UNKNOWN_CATEGORY = INT8_MIN;

  // Null `unicode_view` constructor
  constexpr unicode_view() noexcept : ptr_(nullptr), length_(0), category_(_UNKNOWN_CATEGORY) {
  }

  // Implicit constructors

  template <typename Allocator>
  unicode_view(  // NOLINT(runtime/explicit)
      const std::basic_string<value_type, std::char_traits<value_type>, Allocator>& str) noexcept
      // This is implemented in terms of `unicode_view(p, n)` so `str.size()`
      // doesn't need to be reevaluated after `ptr_` is set.
      : unicode_view(str.data(), str.size()) {
  }

  // Implicit constructor of a `unicode_view` from NUL-terminated `str`. When
  // accepting possibly null strings, use `absl::NullSafeStringView(str)`
  // instead (see below).
  constexpr unicode_view(const value_type* str) noexcept  // NOLINT(runtime/explicit)
      : ptr_(str), length_(str ? (traits_type::length(str)) : 0), category_(_UNKNOWN_CATEGORY) {
  }

  // Implicit constructor of a `unicode_view` from a `const value_type*` and length.
  constexpr unicode_view(const value_type* data, size_type len) noexcept
      : ptr_(data), length_(len), category_(_UNKNOWN_CATEGORY) {
  }

  // only for Unicode
  constexpr explicit unicode_view(const value_type* data, size_type len, int32_t category) noexcept
      : ptr_(data), length_(len), category_(category) {
  }

  // NOTE: Harmlessly omitted to work around gdb bug.
  //   constexpr unicode_view(const unicode_view&) noexcept = default;
  //   unicode_view& operator=(const unicode_view&) noexcept = default;

  // Iterators

  // unicode_view::begin()
  //
  // Returns an iterator pointing to the first character at the beginning of the
  // `unicode_view`, or `end()` if the `unicode_view` is empty.
  constexpr const_iterator begin() const noexcept {
    return ptr_;
  }

  // unicode_view::end()
  //
  // Returns an iterator pointing just beyond the last character at the end of
  // the `unicode_view`. This iterator acts as a placeholder; attempting to
  // access it results in undefined behavior.
  constexpr const_iterator end() const noexcept {
    return ptr_ + length_;
  }

  // unicode_view::cbegin()
  //
  // Returns a const iterator pointing to the first character at the beginning
  // of the `unicode_view`, or `end()` if the `unicode_view` is empty.
  constexpr const_iterator cbegin() const noexcept {
    return begin();
  }

  // unicode_view::cend()
  //
  // Returns a const iterator pointing just beyond the last character at the end
  // of the `unicode_view`. This pointer acts as a placeholder; attempting to
  // access its element results in undefined behavior.
  constexpr const_iterator cend() const noexcept {
    return end();
  }

  // unicode_view::rbegin()
  //
  // Returns a reverse iterator pointing to the last character at the end of the
  // `unicode_view`, or `rend()` if the `unicode_view` is empty.
  const_reverse_iterator rbegin() const noexcept {
    return const_reverse_iterator(end());
  }

  // unicode_view::rend()
  //
  // Returns a reverse iterator pointing just before the first character at the
  // beginning of the `unicode_view`. This pointer acts as a placeholder;
  // attempting to access its element results in undefined behavior.
  const_reverse_iterator rend() const noexcept {
    return const_reverse_iterator(begin());
  }

  // unicode_view::crbegin()
  //
  // Returns a const reverse iterator pointing to the last character at the end
  // of the `unicode_view`, or `crend()` if the `unicode_view` is empty.
  const_reverse_iterator crbegin() const noexcept {
    return rbegin();
  }

  // unicode_view::crend()
  //
  // Returns a const reverse iterator pointing just before the first character
  // at the beginning of the `unicode_view`. This pointer acts as a placeholder;
  // attempting to access its element results in undefined behavior.
  const_reverse_iterator crend() const noexcept {
    return rend();
  }

  // Capacity Utilities

  // unicode_view::size()
  //
  // Returns the number of characters in the `unicode_view`.
  constexpr size_type size() const noexcept {
    return length_;
  }

  // unicode_view::length()
  //
  // Returns the number of characters in the `unicode_view`. Alias for `size()`.
  constexpr size_type length() const noexcept {
    return size();
  }

  // unicode_view::max_size()
  //
  // Returns the maximum number of characters the `unicode_view` can hold.
  constexpr size_type max_size() const noexcept {
    return kMaxSize;
  }

  // unicode_view::empty()
  //
  // Checks if the `unicode_view` is empty (refers to no characters).
  constexpr bool empty() const noexcept {
    return length_ == 0;
  }

  // unicode_view::operator[]
  //
  // Returns the ith element of the `unicode_view` using the array operator.
  // Note that this operator does not perform any bounds checking.
  constexpr const_reference operator[](size_type i) const noexcept {
    return MATXSCRIPT_ASSERT(i < size()), ptr_[i];
  }

  // unicode_view::at()
  //
  // Returns the ith element of the `unicode_view`. Bounds checking is performed,
  // and an exception of type `std::out_of_range` will be thrown on invalid
  // access.
  constexpr const_reference at(size_type i) const {
    return MATXSCRIPT_LIKELY(i < size()) ? ptr_[i] : throw std::out_of_range("unicode_view::at"),
           ptr_[i];
  }

  // unicode_view::front()
  //
  // Returns the first element of a `unicode_view`.
  constexpr const_reference front() const noexcept {
    return MATXSCRIPT_ASSERT(!empty()), ptr_[0];
  }

  // unicode_view::back()
  //
  // Returns the last element of a `unicode_view`.
  constexpr const_reference back() const noexcept {
    return MATXSCRIPT_ASSERT(!empty()), ptr_[size() - 1];
  }

  // unicode_view::data()
  //
  // Returns a pointer to the underlying character array (which is of course
  // stored elsewhere). Note that `unicode_view::data()` may contain embedded nul
  // characters, but the returned buffer may or may not be NUL-terminated;
  // therefore, do not pass `data()` to a routine that expects a NUL-terminated
  // std::string.
  constexpr const_pointer data() const noexcept {
    return ptr_;
  }

  // Modifiers

  // unicode_view::remove_prefix()
  //
  // Removes the first `n` characters from the `unicode_view`. Note that the
  // underlying std::string is not changed, only the view.
  void remove_prefix(size_type n) noexcept {
    MATXSCRIPT_ASSERT(n <= length_);
    ptr_ += n;
    length_ -= n;
  }

  // unicode_view::remove_suffix()
  //
  // Removes the last `n` characters from the `unicode_view`. Note that the
  // underlying std::string is not changed, only the view.
  void remove_suffix(size_type n) noexcept {
    MATXSCRIPT_ASSERT(n <= length_);
    length_ -= n;
  }

  // unicode_view::swap()
  //
  // Swaps this `unicode_view` with another `unicode_view`.
  void swap(unicode_view& s) noexcept {
    auto t = *this;
    *this = s;
    s = t;
  }

  // Explicit conversion operators

  // Converts to `std::basic_string`.
  template <typename A>
  explicit operator std::basic_string<value_type, traits_type, A>() const {
    if (!data())
      return {};
    return std::basic_string<value_type, traits_type, A>(data(), size());
  }

  // unicode_view::copy()
  //
  // Copies the contents of the `unicode_view` at offset `pos` and length `n`
  // into `buf`.
  size_type copy(value_type* buf, size_type n, size_type pos = 0) const {
    if (MATXSCRIPT_UNLIKELY(pos > length_)) {
      throw std::out_of_range("unicode_view::copy");
    }
    size_type rlen = (std::min)(length_ - pos, n);
    if (rlen > 0) {
      const value_type* start = ptr_ + pos;
      traits_type::copy(buf, start, rlen);
    }
    return rlen;
  }

  // unicode_view::substr()
  //
  // Returns a "substring" of the `unicode_view` (at offset `pos` and length
  // `n`) as another unicode_view. This function throws `std::out_of_bounds` if
  // `pos > size`.
  unicode_view substr(size_type pos, size_type n = npos) const {
    return MATXSCRIPT_UNLIKELY(pos > length_)
               ? (throw std::out_of_range("unicode_view::substr"), unicode_view{})
               : unicode_view(ptr_ + pos, Min(n, length_ - pos));
  }

  // Nonstandard
  unicode_view SubStrNoCheck(size_type pos, size_type n = npos) const noexcept {
    return unicode_view(ptr_ + pos, Min(n, length_ - pos));
  }

  // unicode_view::compare()
  //
  // Performs a lexicographical comparison between the `unicode_view` and
  // another `absl::unicode_view`, returning -1 if `this` is less than, 0 if
  // `this` is equal to, and 1 if `this` is greater than the passed std::string
  // view. Note that in the case of data equality, a further comparison is made
  // on the respective sizes of the two `unicode_view`s to determine which is
  // smaller, equal, or greater.
  constexpr int compare(unicode_view x) const noexcept {
    return CompareImpl(
        length_,
        x.length_,
        length_ == 0 || x.length_ == 0
            ? 0
            : traits_type::compare(ptr_, x.ptr_, length_ < x.length_ ? length_ : x.length_));
  }

  // Overload of `unicode_view::compare()` for comparing a substring of the
  // 'unicode_view` and another `absl::unicode_view`.
  int compare(size_type pos1, size_type count1, unicode_view v) const {
    return substr(pos1, count1).compare(v);
  }

  // Overload of `unicode_view::compare()` for comparing a substring of the
  // `unicode_view` and a substring of another `absl::unicode_view`.
  int compare(
      size_type pos1, size_type count1, unicode_view v, size_type pos2, size_type count2) const {
    return substr(pos1, count1).compare(v.substr(pos2, count2));
  }

  // Overload of `unicode_view::compare()` for comparing a `unicode_view` and a
  // a different  C-style std::string `s`.
  int compare(const value_type* s) const noexcept {
    return compare(unicode_view(s));
  }

  // Overload of `unicode_view::compare()` for comparing a substring of the
  // `unicode_view` and a different std::string C-style std::string `s`.
  int compare(size_type pos1, size_type count1, const value_type* s) const {
    return substr(pos1, count1).compare(unicode_view(s));
  }

  // Overload of `unicode_view::compare()` for comparing a substring of the
  // `unicode_view` and a substring of a different C-style std::string `s`.
  int compare(size_type pos1, size_type count1, const value_type* s, size_type count2) const {
    return substr(pos1, count1).compare(unicode_view(s, count2));
  }

  // Find Utilities

  // unicode_view::find()
  //
  // Finds the first occurrence of the substring `s` within the `unicode_view`,
  // returning the position of the first character's match, or `npos` if no
  // match was found.
  size_type find(unicode_view s, size_type pos = 0) const noexcept;

  // Overload of `unicode_view::find()` for finding the given character `c`
  // within the `unicode_view`.
  size_type find(value_type c, size_type pos = 0) const noexcept;

  // unicode_view::rfind()
  //
  // Finds the last occurrence of a substring `s` within the `unicode_view`,
  // returning the position of the first character's match, or `npos` if no
  // match was found.
  size_type rfind(unicode_view s, size_type pos = npos) const noexcept;

  // Overload of `unicode_view::rfind()` for finding the last given character `c`
  // within the `unicode_view`.
  size_type rfind(value_type c, size_type pos = npos) const noexcept;

  // unicode_view::find_first_of()
  //
  // Finds the first occurrence of any of the characters in `s` within the
  // `unicode_view`, returning the start position of the match, or `npos` if no
  // match was found.
  size_type find_first_of(unicode_view s, size_type pos = 0) const noexcept;

  // Overload of `unicode_view::find_first_of()` for finding a character `c`
  // within the `unicode_view`.
  size_type find_first_of(value_type c, size_type pos = 0) const noexcept {
    return find(c, pos);
  }

  // unicode_view::find_last_of()
  //
  // Finds the last occurrence of any of the characters in `s` within the
  // `unicode_view`, returning the start position of the match, or `npos` if no
  // match was found.
  size_type find_last_of(unicode_view s, size_type pos = npos) const noexcept;

  // Overload of `unicode_view::find_last_of()` for finding a character `c`
  // within the `unicode_view`.
  size_type find_last_of(value_type c, size_type pos = npos) const noexcept {
    return rfind(c, pos);
  }

  // unicode_view::find_first_not_of()
  //
  // Finds the first occurrence of any of the characters not in `s` within the
  // `unicode_view`, returning the start position of the first non-match, or
  // `npos` if no non-match was found.
  size_type find_first_not_of(unicode_view s, size_type pos = 0) const noexcept;

  // Overload of `unicode_view::find_first_not_of()` for finding a character
  // that is not `c` within the `unicode_view`.
  size_type find_first_not_of(value_type c, size_type pos = 0) const noexcept;

  // unicode_view::find_last_not_of()
  //
  // Finds the last occurrence of any of the characters not in `s` within the
  // `unicode_view`, returning the start position of the last non-match, or
  // `npos` if no non-match was found.
  size_type find_last_not_of(unicode_view s, size_type pos = npos) const noexcept;

  // Overload of `unicode_view::find_last_not_of()` for finding a character
  // that is not `c` within the `unicode_view`.
  size_type find_last_not_of(value_type c, size_type pos = npos) const noexcept;

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

  const value_type* ptr_;
  size_type length_;
  int32_t category_;
};

// This large function is defined inline so that in a fairly common case where
// one of the arguments is a literal, the compiler can elide a lot of the
// following comparisons.
constexpr bool operator==(unicode_view x, unicode_view y) noexcept {
  return x.size() == y.size() &&
         (x.empty() || unicode_view::traits_type::compare(x.data(), y.data(), x.size()) == 0);
}

constexpr bool operator!=(unicode_view x, unicode_view y) noexcept {
  return !(x == y);
}

constexpr bool operator<(unicode_view x, unicode_view y) noexcept {
  return x.compare(y) < 0;
}

constexpr bool operator>(unicode_view x, unicode_view y) noexcept {
  return y < x;
}

constexpr bool operator<=(unicode_view x, unicode_view y) noexcept {
  return !(y < x);
}

constexpr bool operator>=(unicode_view x, unicode_view y) noexcept {
  return !(x < y);
}

// IO Insertion Operator
std::ostream& operator<<(std::ostream& o, unicode_view piece);

// ClippedSubstr()
//
// Like `s.substr(pos, n)`, but clips `pos` to an upper bound of `s.size()`.
// Provided because std::unicode_view::substr throws if `pos > size()`
inline unicode_view ClippedSubstr(unicode_view s, size_t pos, size_t n = unicode_view::npos) {
  pos = (std::min)(pos, static_cast<size_t>(s.size()));
  return s.substr(pos, n);
}

}  // namespace runtime
}  // namespace matxscript

namespace std {

template <>
struct hash<::matxscript::runtime::unicode_view> {
  std::size_t operator()(const ::matxscript::runtime::unicode_view& str) const {
    constexpr size_t ele_size = sizeof(::matxscript::runtime::unicode_view::value_type);
    return ::matxscript::runtime::BytesHash(str.data(), str.size() * ele_size);
  }
};

}  // namespace std
