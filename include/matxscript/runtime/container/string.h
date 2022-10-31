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

#include <matxscript/runtime/container/string_core.h>
#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

// Forward declare TArgValue
class Unicode;
class List;
class Tuple;
class Iterator;
template <typename T>
class FTList;

/*!
 * \brief Reference to string objects.
 *
 * \code
 *
 * // Example to create runtime String reference object from std::string
 * std::string s = "hello world";
 *
 * // You can create the reference from existing std::string
 * String ref{std::move(s)};
 *
 * // You can rebind the reference to another string.
 * ref = std::string{"hello world2"};
 *
 * // You can use the reference as hash map key
 * std::unordered_map<String, int32_t> m;
 * m[ref] = 1;
 *
 * // You can compare the reference object with other string objects
 * assert(ref == "hello world", true);
 *
 * // You can convert the reference to std::string again
 * string s2 = (string)ref;
 *
 * \endcode
 */
class String {
 public:
  static constexpr bool _type_is_nullable = false;

 public:
  using self_view = string_view;
  using size_type = self_view::size_type;

  static constexpr size_type npos = string_view::npos;
  // types
  using traits_type = std::char_traits<char>;
  using value_type = char;
  using pointer = char*;
  using const_pointer = const char*;
  using reference = char&;
  using const_reference = const char&;
  using iterator = char*;
  using const_iterator = const char*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using difference_type = std::ptrdiff_t;
  using ContainerType = string_core<value_type>;

 public:
  // Should always true
  bool isSane() const noexcept;
  // Custom operator with string view
  operator self_view() const noexcept;
  self_view view() const noexcept;

  // Convert with Any
  void MoveTo(MATXScriptAny* value) noexcept;
  static String MoveFromNoCheck(MATXScriptAny* value) noexcept;

  // C++11 21.4.2 construct/copy/destroy
  String() noexcept = default;
  String(const String& other) = default;
  String(String&& other) noexcept = default;

  // This is defined for compatibility with std::string
  String(const std::string& other) : data_(other.data(), other.size()) {
  }

  String(const String& str, size_type pos, size_type n) : String(str.view().substr(pos, n)) {
  }

  String(const String& str, size_type pos) : String(str.view().substr(pos)) {
  }

  String(const value_type* other) : data_(other, traits_type::length(other)) {
  }

  String(const value_type* other, size_type n) : data_(other, n) {
  }

  String(size_type n, value_type c) : data_(n, c) {
  }

  template <typename IterType,
            typename = typename std::enable_if<!std::is_same<IterType, value_type*>::value>::type>
  String(IterType first, IterType last) {
    size_type num = std::distance(first, last);
    this->reserve(num);
    while (first != last) {
      this->push_back(*first);
      ++first;
    }
  }

  // Specialization for const char*, const char*
  String(const value_type* b, const value_type* e) : data_(b, e - b) {
  }

  // Nonstandard constructor
  explicit String(const ContainerType& other) : data_(other) {
  }
  explicit String(ContainerType&& other) noexcept : data_(std::move(other)) {
  }

  String(self_view other) : data_(other.data(), other.size(), other.category()) {
  }

  // Construction from initialization list
  String(std::initializer_list<value_type> il) : data_(il.begin(), il.size()) {
  }

  ~String() noexcept = default;

  // Copy assignment
  String& operator=(const String& other);

  // Move assignment
  String& operator=(String&& other) noexcept;

  String& operator=(const value_type* other);

  String& operator=(value_type other);

  String& operator=(std::initializer_list<value_type> il);

  // Compatibility with std::string and std::basic_string_view
  String& operator=(const std::string& other);
  operator ::std::string() const;

  // clang-format off
#if MATXSCRIPT_USE_CXX17_STRING_VIEW
  explicit operator std::basic_string_view<value_type, std::char_traits<value_type>>()
      const noexcept {
    return {data(), size_t(size())};
  }
#elif MATXSCRIPT_USE_CXX14_STRING_VIEW
  explicit operator std::experimental::basic_string_view<value_type, std::char_traits<value_type>>()
      const noexcept {
    return {data(), size_t(size())};
  }
#endif
  // clang-format on

  // Nonstandard assignment
  String& operator=(self_view other);

  // C++11 21.4.3 iterators:
  iterator begin();
  const_iterator begin() const noexcept;
  const_iterator cbegin() const noexcept;

  iterator end();
  const_iterator end() const noexcept;
  const_iterator cend() const noexcept;

  reverse_iterator rbegin();
  const_reverse_iterator rbegin() const noexcept;
  const_reverse_iterator crbegin() const noexcept;

  reverse_iterator rend();
  const_reverse_iterator rend() const noexcept;
  const_reverse_iterator crend() const noexcept;

  // Added by C++11
  // C++11 21.4.5, element access:
  const value_type& front() const noexcept;
  const value_type& back() const noexcept;
  value_type& front();
  value_type& back();
  void pop_back();

  // C++11 21.4.4 capacity:
  int64_t size() const noexcept;

  int64_t length() const noexcept;

  size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max();
  }

  void resize(size_type n, value_type c = '\0');
  void resizeNoInit(size_type n);  // custom resize

  int64_t capacity() const noexcept;

  void reserve(size_type res_arg = 0);

  void shrink_to_fit();

  void clear();

  bool empty() const noexcept;

  // C++11 21.4.5 element access:
  const_reference operator[](size_type pos) const noexcept;
  reference operator[](size_type pos);

  const_reference at(size_type n) const;
  reference at(size_type n);

  // C++11 21.4.6 modifiers:
  String& operator+=(const String& str);
  String& operator+=(const value_type* s);
  String& operator+=(value_type c);
  String& operator+=(std::initializer_list<value_type> il);

  String& append(self_view str);
  String& append(self_view str, size_type pos, size_type n);
  String& append(const value_type* s, size_type n);
  String& append(const value_type* s);
  String& append(size_type n, value_type c);

  template <class InputIterator>
  String& append(InputIterator first, InputIterator last) {
    size_type num = std::distance(first, last);
    this->reserve(size() + num);
    while (first != last) {
      this->append(1, *first);
      ++first;
    }
    return *this;
  }

  String& append(std::initializer_list<value_type> il) {
    return append(il.begin(), il.end());
  }

  void push_back(value_type c);

  // Assignment no need self_view
  String& assign(const String& str);
  String& assign(String&& str) noexcept;
  String& assign(const String& str, size_type pos, size_type n);
  String& assign(const value_type* s, size_type n);
  String& assign(const value_type* s);
  template <class ItOrLength, class ItOrChar>
  String& assign(ItOrLength first_or_n, ItOrChar last_or_c) {
    return assign(String(first_or_n, last_or_c));
  }
  String& assign(std::initializer_list<value_type> il) {
    return assign(il.begin(), il.size());
  }

  // Nonstandard assignment
  String& assign(const string_view& s);
  inline String& assign(const std::string& s) {
    return assign(s.data(), s.size());
  }

  String& insert(size_type pos1, const String& str);
  String& insert(size_type pos1, const String& str, size_type pos2, size_type n);
  String& insert(size_type pos, const value_type* str, size_type n);
  String& insert(size_type pos, const value_type* str);
  String& insert(size_type pos, size_type n, value_type c);

  String& insert(size_type pos, self_view s);  // custom

  // TODO: insert by iterator

  String& erase(size_type pos = 0, size_type n = npos);

  // TODO: erase by iterator

  // Replaces at most n1 chars of *this, starting with pos1 with the
  // content of s
  String& replace(size_type pos1, size_type n1, self_view s);

  // Replaces at most n1 chars of *this, starting with pos1,
  // with at most n2 chars of s starting with pos2
  String& replace(size_type pos1, size_type n1, self_view s, size_type pos2, size_type n2);

  // Replaces at most n1 chars of *this, starting with pos, with n2
  // occurrences of c
  //
  // consolidated with
  //
  // Replaces at most n1 chars of *this, starting with pos, with at
  // most n2 chars of str.  str must have at least n2 chars.
  String& replace(size_type pos, size_type n1, self_view s, size_type n2);
  String& replace(size_type pos, size_type n1, size_type n2, value_type c);

  // TODO: replace by iterator

  // TODO: add copy

  // TODO: add swap

  const value_type* c_str() const noexcept;
  const value_type* data() const noexcept;

  size_type find(self_view str, size_type pos = 0) const;
  size_type find(value_type c, size_type pos = 0) const;

  size_type rfind(self_view str, size_type pos = npos) const;
  size_type rfind(value_type c, size_type pos = npos) const;

  size_type find_first_of(self_view str, size_type pos = 0) const;
  size_type find_first_of(value_type c, size_type pos = 0) const;

  size_type find_last_of(self_view str, size_type pos = npos) const;
  size_type find_last_of(value_type c, size_type pos = npos) const;

  size_type find_first_not_of(self_view str, size_type pos = 0) const;
  size_type find_first_not_of(value_type c, size_type pos = 0) const;

  size_type find_last_not_of(self_view str, size_type pos = npos) const;
  size_type find_last_not_of(value_type c, size_type pos = npos) const;

  // TODO: find by pointer

  String substr(size_type pos = 0, size_type n = npos) const;

  int compare(const String& other) const noexcept;

  int compare(const std::string& other) const noexcept;

  int compare(const value_type* other) const noexcept;

  friend inline std::basic_istream<value_type, std::char_traits<value_type>>& getline(
      std::basic_istream<value_type, std::char_traits<value_type>>& is,
      String& str,
      value_type delim) {
    std::string stl_str;
    auto& new_is = getline(is, stl_str, delim);
    str = stl_str;
    return new_is;
  }

  friend inline std::basic_istream<value_type, std::char_traits<value_type>>& getline(
      std::basic_istream<value_type, std::char_traits<value_type>>& is, String& str) {
    return getline(is, str, '\n');
  }

  /******************************************************************************
   * Nonstandard logical operator
   *****************************************************************************/

  inline bool operator==(self_view other) const noexcept {
    return view() == other;
  }
  inline bool operator!=(self_view other) const noexcept {
    return view() != other;
  }
  inline bool operator<(self_view other) const noexcept {
    return view() < other;
  }
  inline bool operator>(self_view other) const noexcept {
    return view() > other;
  }
  inline bool operator<=(self_view other) const noexcept {
    return view() <= other;
  }
  inline bool operator>=(self_view other) const noexcept {
    return view() >= other;
  }

  /******************************************************************************
   * Python bytes builtin methods
   *****************************************************************************/
 public:
  // python generic iterator
  Iterator iter() const;

  // python methods
  String replace(self_view old_s, self_view new_s, int64_t count = -1) const;
  String repeat(int64_t times) const;
  String lower() const;
  String upper() const;
  bool isdigit() const noexcept;
  bool isalpha() const noexcept;
  Unicode decode() const;
  bool contains(self_view str) const noexcept;

  // bytes are immutable, no index_store/slice_store
  int64_t get_item(int64_t pos) const;
  String get_slice(int64_t b, int64_t e, int64_t step = 1) const;

  List split(self_view sep = nullptr, int64_t maxsplit = -1) const;

  template <typename T>
  FTList<T> split_ft(self_view sep = nullptr, int64_t maxsplit = -1) const {
    return StringHelper::SplitFT<T>(view(), sep, maxsplit);
  }

  String join(const RTValue& iterable) const;
  String join(const Iterator& iter) const;
  String join(const List& list) const;
  String join(const FTList<String>& list) const;

  bool endswith(self_view suffix,
                int64_t start = 0,
                int64_t end = std::numeric_limits<int64_t>::max()) const noexcept;
  bool endswith(const Tuple& suffixes,
                int64_t start = 0,
                int64_t end = std::numeric_limits<int64_t>::max()) const;
  bool endswith(const Any& suffix_or_suffixes,
                int64_t start = 0,
                int64_t end = std::numeric_limits<int64_t>::max()) const;

  bool startswith(self_view prefix,
                  int64_t start = 0,
                  int64_t end = std::numeric_limits<int64_t>::max()) const noexcept;
  bool startswith(const Tuple& prefix_or_prefixes,
                  int64_t start = 0,
                  int64_t end = std::numeric_limits<int64_t>::max()) const;
  bool startswith(const Any& prefix_or_prefixes,
                  int64_t start = 0,
                  int64_t end = std::numeric_limits<int64_t>::max()) const;

  String lstrip(self_view chars = self_view{}) const;
  String rstrip(self_view chars = self_view{}) const;
  String strip(self_view chars = self_view{}) const;

  int64_t count(self_view x,
                int64_t start = 0,
                int64_t end = std::numeric_limits<int64_t>::max()) const noexcept;

  static String Concat(self_view lhs, self_view rhs);
  static String Concat(std::initializer_list<self_view> args);

 private:
  ContainerType data_;

  friend struct StringHelper;
  friend struct RTValue;
  friend struct RTView;
};

namespace TypeIndex {
template <>
struct type_index_traits<String> {
  static constexpr int32_t value = kRuntimeString;
};
template <>
struct type_index_traits<string_view> {
  static constexpr int32_t value = kRuntimeString;
};
}  // namespace TypeIndex

// Overload + operator
inline String operator+(const String& lhs, const String& rhs) {
  return StringHelper::Concat(String::self_view(lhs), String::self_view(rhs));
}

inline String operator+(const char* lhs, const String& rhs) {
  return StringHelper::Concat(String::self_view(lhs), String::self_view(rhs));
}

inline String operator+(const String& lhs, const char* rhs) {
  return StringHelper::Concat(String::self_view(lhs), String::self_view(rhs));
}

inline String operator+(const String& lhs, const std::string& rhs) {
  return StringHelper::Concat(String::self_view(lhs), String::self_view(rhs));
}

inline String operator+(const std::string& lhs, const String& rhs) {
  return StringHelper::Concat(String::self_view(lhs), String::self_view(rhs));
}

inline String operator+(const String& lhs, const String::self_view& rhs) {
  return StringHelper::Concat(String::self_view(lhs), rhs);
}

inline String operator+(const String::self_view& lhs, const String& rhs) {
  return StringHelper::Concat(lhs, String::self_view(rhs));
}

inline std::ostream& operator<<(std::ostream& out, const String& input) {
  out.write(input.data(), input.size());
  return out;
}

}  // namespace runtime
}  // namespace matxscript

namespace std {

template <>
struct hash<::matxscript::runtime::String> {
  std::size_t operator()(const ::matxscript::runtime::String& str) const noexcept {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
  std::size_t operator()(::matxscript::runtime::string_view str) const noexcept {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
  std::size_t operator()(const ::std::string& str) const noexcept {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
  std::size_t operator()(const char* str) const noexcept {
    return operator()(::matxscript::runtime::string_view(str));
  }
};

template <>
struct equal_to<::matxscript::runtime::String> {
  bool operator()(const ::matxscript::runtime::String& a,
                  const ::matxscript::runtime::String& b) const noexcept {
    return a == b;
  }
  bool operator()(::matxscript::runtime::string_view a,
                  const ::matxscript::runtime::String& b) const noexcept {
    return a == b.view();
  }
  bool operator()(const ::matxscript::runtime::String& a,
                  ::matxscript::runtime::string_view b) const noexcept {
    return a.view() == b;
  }
  bool operator()(const ::std::string& a, const ::matxscript::runtime::String& b) const noexcept {
    return ::matxscript::runtime::string_view(a) == b.view();
  }
  bool operator()(const ::matxscript::runtime::String& a, const ::std::string& b) const noexcept {
    return a.view() == ::matxscript::runtime::string_view(b);
  }
  bool operator()(const ::matxscript::runtime::String& a, const char* b) const noexcept {
    return operator()(a, ::matxscript::runtime::string_view(b));
  }
  bool operator()(const char* a, const ::matxscript::runtime::String& b) const noexcept {
    return operator()(::matxscript::runtime::string_view(a), b);
  }
};

}  // namespace std
