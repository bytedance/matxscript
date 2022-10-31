// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file is inspired by incubator-tvm.
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

#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include <matxscript/runtime/_almost_equal.h>
#include <matxscript/runtime/_is_comparable.h>
#include <matxscript/runtime/_is_hashable.h>
#include <matxscript/runtime/bytes_hash.h>
#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/container/unicode_view.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/demangle.h>
#include <matxscript/runtime/object.h>

namespace matxscript {
namespace runtime {

class RTValue;
class RTView;
class Any;
class FTObjectBase;

// When RTValue is a pod, the default is value copy, otherwise it is reference copy

// macro to check type code.
#define MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(CODE, T)                         \
  MXCHECK_EQ(CODE, T) << "[RTValue] expected " << TypeIndex2Str(T) << " but get " \
                      << TypeIndex2Str(CODE)

template <typename TObjectRef>
struct ObjectView {
  using ContainerType = typename TObjectRef::ContainerType;
  inline explicit ObjectView() noexcept : ref{nullptr} {
  }
  inline explicit ObjectView(const Any& val, bool check = false);
  inline ~ObjectView() {
    ref.data_.data_ = nullptr;
  }

  ObjectView(ObjectView&&) noexcept = default;
  ObjectView& operator=(ObjectView&&) noexcept = default;
  ObjectView(const ObjectView& other) noexcept {
    auto* object_node = static_cast<Object*>(other.ref.data_.data_);
    ref = TObjectRef(ObjectPtr<Object>::MoveFromRValueRefArg(&object_node));
  }
  ObjectView& operator=(const ObjectView& other) noexcept {
    ref.data_.data_ = nullptr;
    auto* object_node = static_cast<Object*>(other.ref.data_.data_);
    ref = TObjectRef(ObjectPtr<Object>::MoveFromRValueRefArg(&object_node));
    return *this;
  }

  inline const TObjectRef& data() const noexcept {
    return ref;
  }

 private:
  TObjectRef ref{ObjectPtr<Object>(nullptr)};
};

std::type_index GetFTObjectBaseStdTypeIndex(const Object* o) noexcept;

struct Any {
 public:
  // pod method
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE bool Is() const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    return Is<U>(std::is_base_of<ObjectRef, TYPE>{});
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE bool Is(std::false_type) const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    return TypeIndex::type_index_traits<TYPE>::value == value_.code;
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE bool Is(std::true_type) const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    return IsObjectRef<TYPE>();
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE U As() const {
    return AsDefault<U>(std::is_base_of<ObjectRef, U>{});
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE U AsNoCheck() const {
    return AsDefaultNoCheck<U>(std::is_base_of<ObjectRef, U>{});
  }

  template <class U>
  MATXSCRIPT_ALWAYS_INLINE U AsDefault(std::true_type) const {
    return AsObjectRef<U>();
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE U AsDefault(std::false_type) const {
    return U(*this);
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE U AsDefaultNoCheck(std::true_type) const {
    return AsObjectRefNoCheck<U>();
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE U AsDefaultNoCheck(std::false_type) const {
    return U(*this);
  }

  // is none
  // TODO: remove me
  constexpr int is_nullptr() const noexcept {
    return value_.code == TypeIndex::kRuntimeNullptr;
  }
  constexpr const MATXScriptAny& value() const noexcept {
    return value_;
  }
  constexpr int type_code() const noexcept {
    return value_.code;
  }

  String type_name() const;

  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template <typename T>
  T* ptr() const noexcept {
    return static_cast<T*>(value_.data.v_handle);
  }
  // ObjectRef handling
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  bool IsObjectRef(std::true_type) const {
    using TYPE_OBJECT_REF =
        typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type;
    if (std::is_same<TYPE_OBJECT_REF, FTObjectBase>::value) {
      return true;
    }
    if (value_.code != TypeIndex::type_index_traits<TYPE_OBJECT_REF>::value) {
      return false;
    }
    auto ty_idx = GetFTObjectBaseStdTypeIndex(ptr<Object>());
    return string_view(ty_idx.name()) == string_view(typeid(TYPE_OBJECT_REF).name());
  }
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  bool IsObjectRef(std::false_type) const {
    return value_.code >= 0 ? IsConvertible<TObjectRef>(ptr<Object>()) : false;
  }
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  bool IsObjectRef() const {
    if (TObjectRef::_type_is_nullable && value_.code == TypeIndex::kRuntimeNullptr) {
      return true;
    }
    return IsObjectRef<TObjectRef>(
        std::is_base_of<
            FTObjectBase,
            typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>());
  }
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  inline TObjectRef AsObjectRef() const;

  template <class TObjectRef>
  TObjectRef AsObjectRefNoCheck() const noexcept {
    static_assert(std::is_base_of<ObjectRef, TObjectRef>::value,
                  "Conversion only works for ObjectRef");
    return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
  }

  template <
      class TObjectRef,
      typename = typename std::enable_if<
          std::is_base_of<ObjectRef,
                          typename std::remove_cv<
                              typename std::remove_reference<TObjectRef>::type>::type>::value &&
          !std::is_same<String,
                        typename std::remove_cv<
                            typename std::remove_reference<TObjectRef>::type>::type>::value &&
          !std::is_same<Unicode,
                        typename std::remove_cv<
                            typename std::remove_reference<TObjectRef>::type>::type>::value>::type>
  ObjectView<TObjectRef> AsObjectView() const {
    return ObjectView<TObjectRef>(*this, true);
  }

  template <class TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  ObjectView<TObjectRef> AsObjectViewNoCheck() const {
    return ObjectView<TObjectRef>(*this, false);
  }

  bool IsString() const noexcept;
  bool IsUnicode() const noexcept;

  static std::size_t Hash(const Any& a);
  static bool Equal(const Any& lhs, const Any& rhs);
  static bool LessThan(const Any& lhs, const Any& rhs);
  static bool LessEqual(const Any& lhs, const Any& rhs);
  static bool GreaterThan(const Any& lhs, const Any& rhs);
  static bool GreaterEqual(const Any& lhs, const Any& rhs);

 protected:
  constexpr Any() noexcept : value_({0, 0, TypeIndex::kRuntimeNullptr}) {
  }
  constexpr Any(MATXScriptAny value) noexcept : value_(value) {
  }
  constexpr Any(const MATXScriptAny* value) noexcept : value_(*value) {
  }
  /*! \brief The value */
  MATXScriptAny value_;
  template <typename TObjectRef>
  friend struct ObjectView;
  friend class ArithOps;
  friend struct SmartHash;
  friend struct SmartEqualTo;
  friend class RTView;
  friend class RTValue;
};

template <>
double Any::As<double>() const;
template <>
double Any::AsNoCheck<double>() const;

template <>
MATXSCRIPT_ALWAYS_INLINE float Any::As<float>() const {
  return As<double>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE float Any::AsNoCheck<float>() const {
  return AsNoCheck<double>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE int64_t Any::As<int64_t>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeInteger);
  return value_.data.v_int64;
}
template <>
MATXSCRIPT_ALWAYS_INLINE int64_t Any::AsNoCheck<int64_t>() const {
  return value_.data.v_int64;
}

template <>
MATXSCRIPT_ALWAYS_INLINE uint64_t Any::As<uint64_t>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE uint64_t Any::AsNoCheck<uint64_t>() const {
  return AsNoCheck<int64_t>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE int32_t Any::As<int32_t>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE int32_t Any::AsNoCheck<int32_t>() const {
  return AsNoCheck<int64_t>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE uint32_t Any::As<uint32_t>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE uint32_t Any::AsNoCheck<uint32_t>() const {
  return AsNoCheck<int64_t>();
}

template <>
bool Any::As<bool>() const;
template <>
bool Any::AsNoCheck<bool>() const;

template <>
MATXSCRIPT_ALWAYS_INLINE char Any::As<char>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE char Any::AsNoCheck<char>() const {
  return AsNoCheck<int64_t>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE unsigned char Any::As<unsigned char>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE unsigned char Any::AsNoCheck<unsigned char>() const {
  return AsNoCheck<int64_t>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE char32_t Any::As<char32_t>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE char32_t Any::AsNoCheck<char32_t>() const {
  return AsNoCheck<int64_t>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE signed char Any::As<signed char>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE signed char Any::AsNoCheck<signed char>() const {
  return AsNoCheck<int64_t>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE short int Any::As<short int>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE short int Any::AsNoCheck<short int>() const {
  return AsNoCheck<int64_t>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE unsigned short int Any::As<unsigned short int>() const {
  return As<int64_t>();
}
template <>
MATXSCRIPT_ALWAYS_INLINE unsigned short int Any::AsNoCheck<unsigned short int>() const {
  return AsNoCheck<int64_t>();
}

template <>
void* Any::As<void*>() const;
template <>
void* Any::AsNoCheck<void*>() const;

template <>
string_view Any::As<string_view>() const;
template <>
string_view Any::AsNoCheck<string_view>() const;
template <>
MATXSCRIPT_ALWAYS_INLINE String Any::As<String>() const {
  return String(As<string_view>());
}
template <>
MATXSCRIPT_ALWAYS_INLINE String Any::AsNoCheck<String>() const {
  return String(AsNoCheck<string_view>());
}

template <>
unicode_view Any::As<unicode_view>() const;
template <>
unicode_view Any::AsNoCheck<unicode_view>() const;
template <>
MATXSCRIPT_ALWAYS_INLINE Unicode Any::As<Unicode>() const {
  return Unicode(As<unicode_view>());
}
template <>
MATXSCRIPT_ALWAYS_INLINE Unicode Any::AsNoCheck<Unicode>() const {
  return Unicode(AsNoCheck<unicode_view>());
}

template <>
DataType Any::As<DataType>() const;
template <>
DataType Any::AsNoCheck<DataType>() const;

// RTView is only used to pass MATXValue and type_code,
// and provides conversion to other types, without increasing the reference count
class RTView : public Any {
 public:
  // reuse converter from parent
  using Any::As;
  using Any::AsNoCheck;
  using Any::AsObjectRef;
  using Any::IsObjectRef;

  // destructor do nothing
  ~RTView() noexcept = default;
  // constructor
  /*! \brief default constructor */
  constexpr RTView() noexcept : Any() {
  }
  constexpr RTView(MATXScriptAny value) noexcept : Any(value) {
  }
  constexpr RTView(const MATXScriptAny* value) noexcept : Any(*value) {
  }
  RTView(RTView&& other) noexcept : Any(other.value_) {
    other.value_.code = TypeIndex::kRuntimeNullptr;
  }
  RTView(const RTView& other) noexcept = default;
  RTView& operator=(RTView&& other) noexcept {
    value_ = other.value_;
    other.value_.code = TypeIndex::kRuntimeNullptr;
    return *this;
  }
  RTView& operator=(const RTView& other) noexcept = default;

  // number
  RTView(int32_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTView(uint32_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTView(int64_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTView(uint64_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTView(float val) noexcept {
    value_.data.v_float64 = val;
    value_.code = TypeIndex::kRuntimeFloat;
  }
  RTView(double val) noexcept {
    value_.data.v_float64 = val;
    value_.code = TypeIndex::kRuntimeFloat;
  }
  RTView(bool val) noexcept {
    value_.data.v_int64 = val ? 1 : 0;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTView(void* val) noexcept {
    value_.data.v_handle = val;
    value_.code = TypeIndex::kRuntimeOpaqueHandle;
  }
  RTView(std::nullptr_t) noexcept : Any() {
  }
  RTView(const char* str) noexcept : RTView(string_view(str)) {
  }
  RTView(const char32_t* str) noexcept : RTView(unicode_view(str)) {
  }
  RTView(const char* str, size_t len) noexcept : RTView(string_view(str, len)) {
  }
  RTView(const char32_t* str, size_t len) noexcept : RTView(unicode_view(str, len)) {
  }

  RTView(DataType dtype) noexcept {
    value_.data.v_type = dtype;
    value_.code = TypeIndex::kRuntimeDataType;
  }

  // object
  template <class TObjectRef,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  RTView(const TObjectRef& other) noexcept {
    if (other.data_.data_) {
      value_.code = other.data_.data_->type_index();
      value_.data.v_handle = other.data_.data_;
    } else {
      value_.code = TypeIndex::kRuntimeNullptr;
    }
  }

  // string
  RTView(const string_view& other) noexcept;
  RTView(const String& other) noexcept;
  // unicode
  RTView(const unicode_view& other) noexcept;
  RTView(const Unicode& other) noexcept;

  inline RTView(const RTValue& other) noexcept;

 public:
  // Assign operators
  RTView& operator=(double value) noexcept {
    value_.data.v_float64 = value;
    value_.code = TypeIndex::kRuntimeFloat;
    return *this;
  }
  RTView& operator=(std::nullptr_t value) noexcept {
    value_.code = TypeIndex::kRuntimeNullptr;
    return *this;
  }
  RTView& operator=(int64_t value) noexcept {
    value_.data.v_int64 = value;
    value_.code = TypeIndex::kRuntimeInteger;
    return *this;
  }
  RTView& operator=(uint64_t value) noexcept {
    value_.data.v_int64 = value;
    value_.code = TypeIndex::kRuntimeInteger;
    return *this;
  }
  RTView& operator=(int32_t value) noexcept {
    return operator=((int64_t)(value));
  }
  RTView& operator=(uint32_t value) noexcept {
    return operator=((uint64_t)(value));
  }
  RTView& operator=(bool value) noexcept {
    return operator=((int64_t)(value));
  }
  RTView& operator=(void* value) noexcept {
    value_.data.v_handle = value;
    value_.code = TypeIndex::kRuntimeOpaqueHandle;
    return *this;
  }
  RTView& operator=(DataType dtype) noexcept {
    value_.data.v_type = dtype;
    value_.code = TypeIndex::kRuntimeDataType;
    return *this;
  }

  // ObjectRef handling
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  RTView& operator=(const TObjectRef& other) noexcept {
    if (other.data_.data_) {
      value_.code = other.data_.data_->type_index();
      value_.data.v_handle = other.data_.data_;
    } else {
      value_.code = TypeIndex::kRuntimeNullptr;
    }
    return *this;
  }

  // string
  RTView& operator=(const String& other) noexcept;
  RTView& operator=(const string_view& other) noexcept;

  // unicode
  RTView& operator=(const Unicode& other) noexcept;
  RTView& operator=(const unicode_view& other) noexcept;

 public:
  friend class RTValue;
  template <typename TObjectRef>
  friend struct ObjectView;
  friend class ArithOps;
  friend struct SmartHash;
  friend struct SmartEqualTo;
};

class RTValue : public Any {
 public:
  // reuse converter from parent
  using Any::As;
  using Any::AsNoCheck;
  using Any::AsObjectRef;
  using Any::IsObjectRef;

 public:
  // constructor
  /*! \brief default constructor */
  constexpr RTValue() noexcept : Any() {
  }
  // for fast copy
  struct ScalarValueFlag {};
  constexpr RTValue(const MATXScriptAny& value, ScalarValueFlag) noexcept : Any(&value) {
  }
  RTValue(RTValue&& other) noexcept : Any(other.value_) {
    other.value_.data.v_handle = nullptr;
    other.value_.code = TypeIndex::kRuntimeNullptr;
  }
  explicit RTValue(const Any& other);
  RTValue(const RTValue& other);
  RTValue(const RTView& view);

  /*! \brief destructor */
  ~RTValue() noexcept {
    this->Clear();
  }

  void MoveToCHost(MATXScriptAny* ret_value) noexcept;
  static RTValue MoveFromCHost(MATXScriptAny* value) noexcept;
  static RTValue MoveFromCHost(MATXScriptAny value) noexcept;
  void CopyToCHost(MATXScriptAny* ret_value) const;
  static void CopyFromCHostToCHost(const MATXScriptAny* from, MATXScriptAny* to);
  static RTValue CopyFromCHost(const MATXScriptAny* value);
  static RTValue CopyFromCHost(MATXScriptAny value);
  static void DestroyCHost(MATXScriptAny* value) noexcept;

  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  TObjectRef MoveToObjectRef() {
    if (TObjectRef::_type_is_nullable && value_.code == TypeIndex::kRuntimeNullptr) {
      return TObjectRef();
    }
    // TODO: more message
    MXCHECK(IsObjectRef<TObjectRef>())
        << "[Any] expected: " << DemangleType(typeid(TObjectRef).name())
        << ", but get: " << type_name();
    value_.code = TypeIndex::kRuntimeNullptr;
    auto* ref = static_cast<Object*>(value_.data.v_handle);
    return TObjectRef(ObjectPtr<Object>::MoveFromRValueRefArg(&ref));
  }

  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  TObjectRef MoveToObjectRefNoCheck() noexcept {
    value_.code = TypeIndex::kRuntimeNullptr;
    auto* ref = static_cast<Object*>(value_.data.v_handle);
    return TObjectRef(ObjectPtr<Object>::MoveFromRValueRefArg(&ref));
  }

  String MoveToBytes();
  String MoveToBytesNoCheck();
  Unicode MoveToUnicode();
  Unicode MoveToUnicodeNoCheck();

  // number
  RTValue(int32_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTValue(uint32_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTValue(int64_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTValue(uint64_t val) noexcept {
    value_.data.v_int64 = val;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTValue(float val) noexcept {
    value_.data.v_float64 = val;
    value_.code = TypeIndex::kRuntimeFloat;
  }
  RTValue(double val) noexcept {
    value_.data.v_float64 = val;
    value_.code = TypeIndex::kRuntimeFloat;
  }
  RTValue(bool val) noexcept {
    value_.data.v_int64 = val ? 1 : 0;
    value_.code = TypeIndex::kRuntimeInteger;
  }
  RTValue(const char* str);
  RTValue(const char32_t* str);
  RTValue(const char* str, size_t len) : RTValue(String(str, len)) {
  }
  RTValue(const char32_t* str, size_t len) : RTValue(Unicode(str, len)) {
  }
  RTValue(std::nullptr_t) noexcept : Any() {
  }
  RTValue(void* val) noexcept {
    value_.data.v_handle = val;
    value_.code = TypeIndex::kRuntimeOpaqueHandle;
  }
  RTValue(void* val, int32_t code) noexcept {
    value_.data.v_handle = val;
    value_.code = code;
  }
  RTValue(DataType val) noexcept {
    value_.data.v_type = val;
    value_.code = TypeIndex::kRuntimeDataType;
  }

  // object
  template <class TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  RTValue(TObjectRef val) noexcept {
    if (val.data_.data_) {
      value_.code = val.data_.data_->type_index();
      // move the handle out
      value_.data.v_handle = val.data_.data_;
      val.data_.data_ = nullptr;
    } else {
      value_.code = TypeIndex::kRuntimeNullptr;
    }
  }

  // string
  RTValue(String val) noexcept {
    val.MoveTo(&value_);
  }
  RTValue(string_view val) {
    string_core<String::value_type> str(val.data(), val.size(), val.category());
    str.MoveTo(&value_.data.v_str_store, &value_.pad);
    value_.code = TypeIndex::kRuntimeString;
  }
  // unicode
  RTValue(Unicode val) noexcept {
    val.MoveTo(&value_);
  }
  RTValue(unicode_view val) {
    string_core<Unicode::value_type> str(val.data(), val.size(), val.category());
    str.MoveTo(&value_.data.v_str_store, &value_.pad);
    value_.code = TypeIndex::kRuntimeUnicode;
  }

 public:
  bool operator==(const RTValue& other) const;
  inline bool operator!=(const RTValue& other) const {
    return !operator==(other);
  }

  template <typename T,
            typename = typename std::enable_if<std::is_convertible<T, RTValue>::value>::type>
  inline bool operator==(const T& other) const {
    return operator==(RTValue(other));
  }

  template <typename T,
            typename = typename std::enable_if<std::is_convertible<T, RTValue>::value>::type>
  inline bool operator!=(const T& other) const {
    return !operator==(other);
  }

 public:
  // Assign operators
  RTValue& operator=(RTValue&& other) noexcept;
  RTValue& operator=(const RTValue& other);
  RTValue& operator=(double value) noexcept;
  RTValue& operator=(std::nullptr_t value) noexcept;
  RTValue& operator=(int64_t value) noexcept;
  RTValue& operator=(uint64_t value) noexcept;
  RTValue& operator=(int32_t value) noexcept;
  RTValue& operator=(uint32_t value) noexcept;
  RTValue& operator=(bool value) noexcept;
  RTValue& operator=(const char* value);
  RTValue& operator=(const char32_t* value);
  RTValue& operator=(void* value) noexcept;
  RTValue& operator=(DataType dtype) noexcept;
  RTValue& operator=(const RTView& other) {
    return operator=(RTValue(static_cast<const Any&>(other)));
  }

  // ObjectRef handling
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef,
                typename std::remove_cv<typename std::remove_reference<TObjectRef>::type>::type>::
                                                   value>::type>
  inline RTValue& operator=(TObjectRef other) noexcept;

  // String
  RTValue& operator=(String other) noexcept;
  RTValue& operator=(const string_view& other) {
    return operator=(String(other));
  }
  // Unicode
  RTValue& operator=(Unicode other) noexcept;
  RTValue& operator=(const unicode_view& other) {
    return operator=(Unicode(other));
  }

 private:
  template <typename T>
  inline void Assign(const T& other) {
    if (other.value_.code < 0) {
      // pod
      SwitchToPOD(other.value_.code);
      value_ = other.value_;
    } else {
      // object
      operator=(other.template As<ObjectRef>());
    }
  }
  // get the internal container.
  void SwitchToPOD(int type_code) noexcept;
  void SwitchToObject(int type_code, ObjectPtr<Object> other) noexcept;
  void Clear() noexcept {
    DestroyCHost(&value_);
  }

 protected:
  friend class RTView;
  friend class ArithOps;
  friend struct SmartHash;
  friend struct SmartEqualTo;
  template <typename TObjectRef>
  friend struct ObjectView;
};

template <typename TObjectRef, typename>
inline RTValue& RTValue::operator=(TObjectRef other) noexcept {
  using ContainerType = typename TObjectRef::ContainerType;
  const Object* ptr = other.data_.get();
  if (ptr != nullptr) {
    SwitchToObject(ptr->type_index(), std::move(other.data_));
  } else {
    SwitchToPOD(TypeIndex::kRuntimeNullptr);
  }
  return *this;
}

extern std::ostream& operator<<(std::ostream& out, const Any& input);

extern const RTValue None;

template <typename TObjectRef, typename>
inline TObjectRef Any::AsObjectRef() const {
  if (TObjectRef::_type_is_nullable && value_.code == TypeIndex::kRuntimeNullptr) {
    return TObjectRef{ObjectPtr<Object>(nullptr)};
  }
  MXCHECK(IsObjectRef<TObjectRef>())
      << "expected: " << DemangleType(typeid(TObjectRef).name()) << ", but get: " << type_name();
  return TObjectRef(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <typename TObjectRef>
inline ObjectView<TObjectRef>::ObjectView(const Any& val, bool check) {
  if (check) {
    MXCHECK(val.IsObjectRef<TObjectRef>())
        << "[RTValue] expected: " << DemangleType(typeid(TObjectRef).name())
        << ", but get: " << val.type_name();
  }
  auto* object_node = static_cast<Object*>(val.value_.data.v_handle);
  ref = TObjectRef(ObjectPtr<Object>::MoveFromRValueRefArg(&object_node));
}

inline RTView::RTView(const RTValue& other) noexcept : RTView(other.value_) {
}

template <>
MATXSCRIPT_ALWAYS_INLINE RTView Any::As<RTView>() const {
  return RTView{value_};
}
template <>
MATXSCRIPT_ALWAYS_INLINE RTView Any::AsNoCheck<RTView>() const {
  return RTView{value_};
}

template <>
MATXSCRIPT_ALWAYS_INLINE RTValue Any::As<RTValue>() const {
  return RTValue(RTView{value_});
}
template <>
MATXSCRIPT_ALWAYS_INLINE RTValue Any::AsNoCheck<RTValue>() const {
  return RTValue(RTView{value_});
}

// TODO: remove TArgs
class TArgs {
 public:
  const MATXScriptAny* values;
  int num_args;
  TArgs(const MATXScriptAny* values, int num_args) : values(values), num_args(num_args) {
  }
  inline int size() const {
    return num_args;
  }
  RTView operator[](int i) const {
    return RTView{values[i]};
  }
};

namespace {
template <typename From, typename To>
struct has_auto_operator_t {
  template <typename U>
  static constexpr auto judge(U*) ->
      typename std::is_same<To, decltype(std::declval<U>().operator To())>::type;

  template <typename>
  static constexpr std::false_type judge(...);
  typedef decltype(judge<From>(0)) type;
  static constexpr bool value = type::value;
};

template <typename T>
struct is_runtime_value {
  using type = typename std::
      is_base_of<Any, typename std::remove_cv<typename std::remove_reference<T>::type>::type>::type;
  static constexpr bool value = type::value;
};

template <typename... Pack>
struct __my_pack {};

template <typename>
struct _all_is_runtime_value;

template <typename A, typename... R>
struct _all_is_runtime_value<__my_pack<A, R...>> {
  using type = typename std::conditional<is_runtime_value<A>::value,
                                         typename _all_is_runtime_value<__my_pack<R...>>::type,
                                         std::false_type>::type;
  static constexpr bool value = type::value;
};

template <typename A>
struct _all_is_runtime_value<__my_pack<A>> {
  using type = typename is_runtime_value<A>::type;
  static constexpr bool value = type::value;
};

template <typename... R>
struct all_is_runtime_value {
  using type = typename _all_is_runtime_value<__my_pack<R...>>::type;
  static constexpr bool value = type::value;
};

template <typename>
struct _one_of_is_runtime_value;

template <typename A, typename... R>
struct _one_of_is_runtime_value<__my_pack<A, R...>> {
  using type =
      typename std::conditional<is_runtime_value<A>::value,
                                std::true_type,
                                typename _one_of_is_runtime_value<__my_pack<R...>>::type>::type;
  static constexpr bool value = type::value;
};

template <typename A>
struct _one_of_is_runtime_value<__my_pack<A>> {
  using type = typename is_runtime_value<A>::type;
  static constexpr bool value = type::value;
};

template <typename... R>
struct one_of_is_runtime_value {
  using type = typename _one_of_is_runtime_value<__my_pack<R...>>::type;
  static constexpr bool value = type::value;
};

}  // namespace

template <typename To>
struct GenericValueConverter {
  using return_type = To;

 protected:
  static constexpr int32_t DEFAULT_REL = 0;
  static constexpr int32_t IS_SAME = 1;
  static constexpr int32_t FROM_ANY = 2;
  static constexpr int32_t HAS_OPERATOR = 3;

 public:
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE auto operator()(U&& n) {
    using TO_TYPE = typename std::remove_cv<typename std::remove_reference<To>::type>::type;
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    constexpr int32_t cat =
        std::is_same<U_TYPE, TO_TYPE>::value
            ? IS_SAME
            : (std::is_base_of<Any, U_TYPE>::value
                   ? FROM_ANY
                   : (has_auto_operator_t<U_TYPE, TO_TYPE>::value ? HAS_OPERATOR : DEFAULT_REL));
    return operator()(std::forward<U>(n), std::integral_constant<int32_t, cat>{});
  }

 private:
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE auto operator()(U&& n, std::integral_constant<int32_t, DEFAULT_REL>) {
    return To(std::forward<U>(n));
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE auto operator()(U&& n, std::integral_constant<int32_t, IS_SAME>) {
    return std::forward<U>(n);
  }
  template <class U>
  MATXSCRIPT_ALWAYS_INLINE auto operator()(U&& n, std::integral_constant<int32_t, HAS_OPERATOR>) {
    return n.operator To();
  }

  template <class U>
  MATXSCRIPT_ALWAYS_INLINE auto operator()(U&& n, std::integral_constant<int32_t, FROM_ANY>) {
    return n.template As<To>();
  }
  static constexpr int32_t MOVE_ANY_TO_UNICODE = 6;
  static constexpr int32_t MOVE_ANY_TO_BYTES = 7;
  static constexpr int32_t MOVE_ANY_TO_OBJECT = 8;
  static constexpr int32_t COPY_ANY_TO_OTHER = 9;
  MATXSCRIPT_ALWAYS_INLINE auto operator()(RTValue&& n, std::integral_constant<int32_t, FROM_ANY>) {
    using TO_TYPE = typename std::remove_cv<typename std::remove_reference<To>::type>::type;
    constexpr int32_t cat =
        std::is_same<TO_TYPE, Unicode>::value
            ? MOVE_ANY_TO_UNICODE
            : (std::is_same<TO_TYPE, String>::value
                   ? MOVE_ANY_TO_BYTES
                   : (std::is_base_of<ObjectRef, TO_TYPE>::value ? MOVE_ANY_TO_OBJECT
                                                                 : COPY_ANY_TO_OTHER));
    return cast_any_to(std::move(n), std::integral_constant<int32_t, cat>{});
  }
  MATXSCRIPT_ALWAYS_INLINE auto cast_any_to(RTValue&& n,
                                            std::integral_constant<int32_t, MOVE_ANY_TO_BYTES>) {
    return n.MoveToBytes();
  }
  MATXSCRIPT_ALWAYS_INLINE auto cast_any_to(RTValue&& n,
                                            std::integral_constant<int32_t, MOVE_ANY_TO_UNICODE>) {
    return n.MoveToUnicode();
  }
  MATXSCRIPT_ALWAYS_INLINE auto cast_any_to(RTValue&& n,
                                            std::integral_constant<int32_t, MOVE_ANY_TO_OBJECT>) {
    using TO_TYPE = typename std::remove_cv<typename std::remove_reference<To>::type>::type;
    return n.template MoveToObjectRef<TO_TYPE>();
  }
  MATXSCRIPT_ALWAYS_INLINE auto cast_any_to(RTValue&& n,
                                            std::integral_constant<int32_t, COPY_ANY_TO_OTHER>) {
    using TO_TYPE = typename std::remove_cv<typename std::remove_reference<To>::type>::type;
    return n.template As<TO_TYPE>();
  }
};

struct SmartHash {
  typedef ::matxscript::runtime::ska::fibonacci_hash_policy
      hash_policy;  // for gcc492 flat_hash_map

  // overload
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const std::string& val) const {
    return ::matxscript::runtime::BytesHash(val.data(), val.size());
  }
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const char* val) const {
    return ::matxscript::runtime::BytesHash(&val, std::char_traits<char>::length(val));
  }
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const char32_t* val) const {
    auto byte_size = sizeof(char32_t) * std::char_traits<char32_t>::length(val);
    return ::matxscript::runtime::BytesHash(&val, byte_size);
  }
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(::matxscript::runtime::string_view val) const {
    return ::matxscript::runtime::BytesHash(val.data(), val.size());
  }
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(::matxscript::runtime::unicode_view val) const {
    auto byte_size = sizeof(::matxscript::runtime::unicode_view::value_type) * val.size();
    return ::matxscript::runtime::BytesHash(val.data(), byte_size);
  }
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const ::matxscript::runtime::String& val) const {
    return std::hash<::matxscript::runtime::String>()(val);
  }
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const ::matxscript::runtime::Unicode& val) const {
    return std::hash<::matxscript::runtime::Unicode>()(val);
  }

  // main entry
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const T& val) const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    constexpr int32_t code = TypeIndex::type_index_traits<TYPE>::value;
    constexpr int32_t refine_code = code >= 0 ? 0 : code;
    return Dispatch(val, std::integral_constant<int32_t, refine_code>{});
  }

 protected:
  using _IntType = std::integral_constant<int32_t, TypeIndex::type_index_traits<int64_t>::value>;
  using _FloatType = std::integral_constant<int32_t, TypeIndex::type_index_traits<double>::value>;
  using _StringType = std::integral_constant<int32_t, TypeIndex::type_index_traits<char*>::value>;
  using _UnicodeType =
      std::integral_constant<int32_t, TypeIndex::type_index_traits<char32_t*>::value>;
  using _ObjectRefType = std::integral_constant<int32_t, TypeIndex::kRuntimeObject>;
  using _UnkwnownType = std::integral_constant<int32_t, TypeIndex::kRuntimeUnknown>;

  // step1: dispatch int
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t Dispatch(const T& val, _IntType) const {
    return ::matxscript::runtime::ScalarHash<T>()(val);
  }

  // step2: dispatch float
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t Dispatch(const T& val, _FloatType) const {
    return ::matxscript::runtime::ScalarHash<T>()(val);
  }

  // step3: dispatch ObjectRef
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t Dispatch(const T& val, _ObjectRefType) const {
    return Any::Hash(RTView(val));
  }

  // step4: dispatch Any
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t Dispatch(const T& val, _UnkwnownType) const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return DispatchAny(val, std::is_base_of<Any, TYPE>{});
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t DispatchAny(const T& val, std::true_type) const {
    return Any::Hash(val);
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t DispatchAny(const T& val, std::false_type) const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return DispatchAutoCastRTView(val, typename has_auto_operator_t<TYPE, RTView>::type{});
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t DispatchAutoCastRTView(const T& val, std::true_type) const {
    return Any::Hash(val.operator RTView());
  }
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t DispatchAutoCastRTView(const T& val, std::false_type) const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return DispatchAutoCastRTValue(val, typename has_auto_operator_t<TYPE, RTValue>::type{});
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t DispatchAutoCastRTValue(const T& val, std::true_type) const {
    return Any::Hash(val.operator RTValue());
  }
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t DispatchAutoCastRTValue(const T& val,
                                                               std::false_type) const {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return std::hash<TYPE>()(val);
  }
};

struct SmartEqualTo {
  // equal main entry
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T1& lhs, const T2& rhs) const {
    using TYPE1 = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
    using TYPE2 = typename std::remove_cv<typename std::remove_reference<T2>::type>::type;
    constexpr int32_t code1 = TypeIndex::type_index_traits<TYPE1>::value;
    constexpr int32_t code2 = TypeIndex::type_index_traits<TYPE2>::value;
    constexpr int32_t refine_code1 = code1 >= 0 ? 0 : code1;
    constexpr int32_t refine_code2 = code2 >= 0 ? 0 : code2;
    return Dispatch(lhs,
                    rhs,
                    std::integral_constant<int32_t, refine_code1>{},
                    std::integral_constant<int32_t, refine_code2>{});
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T& lhs, const T& rhs) {
    return lhs == rhs;
  }

 protected:
  using _IntType = std::integral_constant<int32_t, TypeIndex::type_index_traits<int64_t>::value>;
  using _FloatType = std::integral_constant<int32_t, TypeIndex::type_index_traits<double>::value>;
  using _StringType = std::integral_constant<int32_t, TypeIndex::type_index_traits<char*>::value>;
  using _UnicodeType =
      std::integral_constant<int32_t, TypeIndex::type_index_traits<char32_t*>::value>;
  using _ObjectRefType = std::integral_constant<int32_t, TypeIndex::kRuntimeObject>;
  using _UnkwnownType = std::integral_constant<int32_t, TypeIndex::kRuntimeUnknown>;

  // lhs: int
  // rhs: Any or int
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, _IntType, _IntType) {
    return lhs == rhs;
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, _IntType, M) {
    switch (rhs.type_code()) {
      case ::matxscript::runtime::TypeIndex::kRuntimeInteger: {
        return lhs == rhs.value_.data.v_int64;
      } break;
      case ::matxscript::runtime::TypeIndex::kRuntimeFloat: {
        return floating_point::AlmostEquals(static_cast<double>(lhs), rhs.value_.data.v_float64);
      } break;
      default: {
        return false;
      } break;
    }
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T1>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, M, _IntType) {
    return Dispatch(rhs, lhs, _IntType{}, M{});
  }

  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs,
                                                const T2& rhs,
                                                _FloatType,
                                                _FloatType) {
    return floating_point::AlmostEquals(lhs, rhs);
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, _FloatType, M) {
    switch (rhs.type_code()) {
      case ::matxscript::runtime::TypeIndex::kRuntimeInteger: {
        return floating_point::AlmostEquals(static_cast<double>(lhs),
                                            static_cast<double>(rhs.value_.data.v_int64));
      } break;
      case ::matxscript::runtime::TypeIndex::kRuntimeFloat: {
        return floating_point::AlmostEquals(static_cast<double>(lhs), rhs.value_.data.v_float64);
      } break;
      default: {
        return false;
      } break;
    }
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T1>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, M, _FloatType) {
    return Dispatch(rhs, lhs, _FloatType{}, M{});
  }

  // lhs: float or int
  // rhs: float or int
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs,
                                                const T2& rhs,
                                                _FloatType,
                                                _IntType) {
    using Float = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
    return floating_point::AlmostEquals<Float, Float>(lhs, rhs);
  }

  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs,
                                                const T2& rhs,
                                                _IntType,
                                                _FloatType) {
    using Float = typename std::remove_cv<typename std::remove_reference<T2>::type>::type;
    return floating_point::AlmostEquals<Float, Float>(lhs, rhs);
  }

  // lhs: str
  // rhs: Any or str
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const ::matxscript::runtime::string_view& lhs,
                                                const ::matxscript::runtime::string_view& rhs,
                                                _StringType,
                                                _StringType) {
    return lhs == rhs;
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, _StringType, M) {
    if (rhs.value_.code != ::matxscript::runtime::TypeIndex::kRuntimeString) {
      return false;
    }
    return Dispatch(lhs,
                    rhs.template AsNoCheck<::matxscript::runtime::string_view>(),
                    _StringType{},
                    _StringType{});
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T1>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, M, _StringType) {
    return Dispatch(rhs, lhs, _StringType{}, M{});
  }

  // lhs: unicode
  // rhs: Any or unicode
  // lhs: str
  // rhs: Any or str
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const ::matxscript::runtime::unicode_view& lhs,
                                                const ::matxscript::runtime::unicode_view& rhs,
                                                _UnicodeType,
                                                _UnicodeType) {
    return lhs == rhs;
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, _UnicodeType, M) {
    if (rhs.value_.code != ::matxscript::runtime::TypeIndex::kRuntimeUnicode) {
      return false;
    }
    return Dispatch(lhs,
                    rhs.template AsNoCheck<::matxscript::runtime::unicode_view>(),
                    _UnicodeType{},
                    _UnicodeType{});
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T1>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, M, _UnicodeType) {
    return Dispatch(rhs, lhs, _UnicodeType{}, M{});
  }

  // lhs: ObjectRef
  // rhs: Any or ObjectRef
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs,
                                                const T2& rhs,
                                                _ObjectRefType,
                                                _ObjectRefType) {
    return Any::Equal(RTView(lhs), RTView(rhs));
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, _ObjectRefType, M) {
    using TYPE1 = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
    if (!rhs.template IsObjectRef<TYPE1>()) {
      return false;
    }
    return Any::Equal(RTView(lhs), rhs);
  }
  template <typename T1,
            typename T2,
            typename M,
            typename = typename std::enable_if<is_runtime_value<T1>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs, const T2& rhs, M, _ObjectRefType) {
    return Dispatch(rhs, lhs, _ObjectRefType{}, M{});
  }

  // lhs: Any
  // rhs: Any
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static bool Dispatch(const T1& lhs,
                                                const T2& rhs,
                                                _UnkwnownType,
                                                _UnkwnownType) {
    using TYPE1 = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
    using TYPE2 = typename std::remove_cv<typename std::remove_reference<T2>::type>::type;
    return DispatchAny(lhs, rhs, std::is_base_of<Any, TYPE1>{}, std::is_base_of<Any, TYPE2>{});
  }
  MATXSCRIPT_ALWAYS_INLINE static bool DispatchAny(const ::matxscript::runtime::Any& lhs,
                                                   const ::matxscript::runtime::Any& rhs,
                                                   std::true_type,
                                                   std::true_type) {
    return ::matxscript::runtime::Any::Equal(lhs, rhs);
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE static bool DispatchAny(const ::matxscript::runtime::Any& lhs,
                                                   const T& rhs,
                                                   std::true_type,
                                                   std::false_type) {
    GenericValueConverter<RTView> Caster;
    return ::matxscript::runtime::Any::Equal(lhs, Caster(rhs));
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE static bool DispatchAny(const T& lhs,
                                                   const ::matxscript::runtime::Any& rhs,
                                                   std::false_type,
                                                   std::true_type) {
    GenericValueConverter<RTView> Caster;
    return ::matxscript::runtime::Any::Equal(Caster(lhs), rhs);
  }

  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE static bool DispatchAny(const T& lhs,
                                                   const ::matxscript::runtime::Any& rhs,
                                                   std::false_type,
                                                   std::false_type) {
    GenericValueConverter<RTView> Caster;
    return ::matxscript::runtime::Any::Equal(Caster(lhs), Caster(rhs));
  }
};

}  // namespace runtime
}  // namespace matxscript

namespace std {
template <>
struct hash<::matxscript::runtime::Any> {
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const T& val) const {
    return ::matxscript::runtime::SmartHash()(val);
  }
};

template <>
struct equal_to<::matxscript::runtime::Any> {
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T1& lhs, const T2& rhs) const {
    return ::matxscript::runtime::SmartEqualTo()(lhs, rhs);
  }
};

template <>
struct hash<::matxscript::runtime::RTView> {
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const T& val) const {
    return ::matxscript::runtime::SmartHash()(val);
  }
};

template <>
struct equal_to<::matxscript::runtime::RTView> {
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T1& lhs, const T2& rhs) const {
    return ::matxscript::runtime::SmartEqualTo()(lhs, rhs);
  }
};

template <>
struct hash<::matxscript::runtime::RTValue> {
  template <typename T>
  MATXSCRIPT_ALWAYS_INLINE std::size_t operator()(const T& val) const {
    return ::matxscript::runtime::SmartHash()(val);
  }
};

template <>
struct equal_to<::matxscript::runtime::RTValue> {
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T1& lhs, const T2& rhs) const {
    return ::matxscript::runtime::SmartEqualTo()(lhs, rhs);
  }
};

template <>
struct less<::matxscript::runtime::Any> {
  template <
      typename T1,
      typename T2,
      typename = typename std::enable_if<::matxscript::runtime::is_runtime_value<T1>::value>::type,
      typename = typename std::enable_if<::matxscript::runtime::is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T1& lhs, const T2& rhs) const {
    return ::matxscript::runtime::Any::LessThan(
        static_cast<const ::matxscript::runtime::Any&>(lhs),
        static_cast<const ::matxscript::runtime::Any&>(rhs));
  }
};

template <>
struct less<::matxscript::runtime::RTView> {
  template <
      typename T1,
      typename T2,
      typename = typename std::enable_if<::matxscript::runtime::is_runtime_value<T1>::value>::type,
      typename = typename std::enable_if<::matxscript::runtime::is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T1& lhs, const T2& rhs) const {
    return ::matxscript::runtime::Any::LessThan(
        static_cast<const ::matxscript::runtime::Any&>(lhs),
        static_cast<const ::matxscript::runtime::Any&>(rhs));
  }
};

template <>
struct less<::matxscript::runtime::RTValue> {
  template <
      typename T1,
      typename T2,
      typename = typename std::enable_if<::matxscript::runtime::is_runtime_value<T1>::value>::type,
      typename = typename std::enable_if<::matxscript::runtime::is_runtime_value<T2>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE bool operator()(const T1& lhs, const T2& rhs) const {
    return ::matxscript::runtime::Any::LessThan(
        static_cast<const ::matxscript::runtime::Any&>(lhs),
        static_cast<const ::matxscript::runtime::Any&>(rhs));
  }
};

}  // namespace std
