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

#include <matxscript/ir/_base/object_equal.h>
#include <matxscript/ir/_base/object_hash.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

// StringRef is only for IR

namespace matxscript {
namespace runtime {

// Forward declare TArgValue
class StringNode;

/*! \brief An object representing string. It's POD type. */
class StringNode : public Object {
 public:
  using DataContainer = String;
  using self_view = string_view;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeStringRef;
  static constexpr const char* _type_key = "runtime.StringRef";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(StringNode, Object);

 public:
  /*! \brief Container that holds the memory. */
  DataContainer data_container;

  StringNode();
  StringNode(DataContainer o);

  operator self_view() const noexcept;

 private:
  friend class StringRef;
};

class StringRef : public ObjectRef {
 public:
  static constexpr bool _type_is_nullable = false;
  using ContainerType = StringNode;

 public:
  // data holder
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

  const StringNode* get() const;
  const StringNode* operator->() const {
    return get();
  }

  // iterators
  iterator begin();
  const_iterator begin() const;
  iterator end();
  const_iterator end() const;
  reverse_iterator rbegin();
  const_reverse_iterator rbegin() const;
  reverse_iterator rend();
  const_reverse_iterator rend() const;

  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  operator self_view() const noexcept;
  self_view view() const noexcept;

  // default constructor
  StringRef();
  explicit StringRef(ObjectPtr<Object> n) : ObjectRef(std::move(n)) {
  }
  StringRef(const StringRef& other) = default;
  StringRef& operator=(const StringRef& other) = default;
  StringRef(StringRef&& other) = default;
  StringRef& operator=(StringRef&& other) = default;

  // constructor from other
  StringRef(String other);
  StringRef(const value_type* const data);
  StringRef(const value_type* data, size_type len);

  // assign from other
  StringRef& operator=(String other);
  StringRef& operator=(const char* other);

  inline bool operator==(self_view other) const noexcept {
    return view() == other;
  }
  inline bool operator!=(self_view other) const noexcept {
    return view() != other;
  }

  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  StringNode* CopyOnWrite();

  /*!
   * \brief Compares this StringRef object to other
   *
   * \param other The StringRef to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const StringRef& other) const;

  /*!
   * \brief Compares this StringRef object to other
   *
   * \param other The string to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const String& other) const;

  /*!
   * \brief Compares this to other
   *
   * \param other The character array to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const value_type* other) const;

  /*!
   * \brief Returns a pointer to the char array in the string.
   *
   * \return const char*
   */
  const value_type* c_str() const;
  const value_type* data() const;

  /*!
   * \brief Return the length of the string
   *
   * \return size_type string length
   */
  int64_t size() const;

  /*!
   * \brief Return the length of the string
   *
   * \return size_type string length
   */
  int64_t length() const;

  /*!
   * \brief Retun if the string is empty
   *
   * \return true if empty, false otherwise.
   */
  bool empty() const;

  // clang-format off
#if MATXSCRIPT_USE_CXX17_STRING_VIEW
  explicit operator std::basic_string_view<value_type, std::char_traits<value_type>>() const noexcept {
    return {data(), size()};
  }
#elif MATXSCRIPT_USE_CXX14_STRING_VIEW
  explicit operator std::experimental::basic_string_view<value_type, std::char_traits<value_type>>()
      const noexcept {
    return {data(), size()};
  }
#endif
  // clang-format on
  operator String() const;

  /*!
   * \brief Check if a TArgValue can be converted to StringRef, i.e. it can be String or
   * StringRef \param val The value to be checked \return A boolean indicating if val can be
   * converted to StringRef
   */
  inline static bool CanConvertFrom(const Any& val) {
    return val.type_code() == TypeIndex::kRuntimeString ||
           val.type_code() == TypeIndex::kRuntimeUnicode ||
           val.type_code() == TypeIndex::kRuntimeStringRef ||
           val.type_code() == TypeIndex::kRuntimeDataType;
  }

  static StringRef Concat(self_view lhs, self_view rhs);

 private:
  /*! \brief Return data_ as type of pointer of StringRefObject */
  StringNode* GetStringNode() const;

  /*! \return The underlying StringRefObject */
  StringNode* CreateOrGetStringNode();

  // Overload + operator
  friend StringRef operator+(const StringRef& lhs, const StringRef& rhs);
  friend StringRef operator+(const StringRef& lhs, const char* rhs);
  friend StringRef operator+(const char* lhs, const StringRef& rhs);
  friend struct ::matxscript::runtime::ObjectEqual;
};

// Overload + operator
inline StringRef operator+(const StringRef& lhs, const StringRef& rhs) {
  return StringRef::Concat(StringRef::self_view(lhs), StringRef::self_view(rhs));
}

inline StringRef operator+(const char* lhs, const StringRef& rhs) {
  return StringRef::Concat(StringRef::self_view(lhs), StringRef::self_view(rhs));
}

inline StringRef operator+(const StringRef& lhs, const char* rhs) {
  return StringRef::Concat(StringRef::self_view(lhs), StringRef::self_view(rhs));
}

inline std::ostream& operator<<(std::ostream& out, const StringRef& input) {
  out.write(input.data(), input.size());
  return out;
}

template <>
inline StringRef Any::As<StringRef>() const {
  switch (value_.code) {
    case TypeIndex::kRuntimeString: {
      return StringRef(AsNoCheck<String>());
    } break;
    case TypeIndex::kRuntimeDataType: {
      return DLDataType2String(value_.data.v_type);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return StringRef(AsNoCheck<Unicode>().encode());
    } break;
    default: {
      return AsObjectRef<StringRef>();
    } break;
  }
}

template <>
inline StringRef Any::AsNoCheck<StringRef>() const {
  return As<StringRef>();
}

}  // namespace runtime
}  // namespace matxscript

namespace std {

template <>
struct hash<::matxscript::runtime::StringRef> {
  std::size_t operator()(const ::matxscript::runtime::StringRef& str) const {
    return ::matxscript::runtime::BytesHash(str.data(), str.size());
  }
};

template <>
struct equal_to<::matxscript::runtime::StringRef> {
  std::size_t operator()(const ::matxscript::runtime::string_view& lhs,
                         const ::matxscript::runtime::string_view& rhs) const {
    return lhs == rhs;
  }
};

}  // namespace std
