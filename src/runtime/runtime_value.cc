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
#include <matxscript/runtime/runtime_value.h>

#include <matxscript/runtime/container/_ft_object_base.h>
#include <matxscript/runtime/container/ndarray.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/function.h>
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>
#include <matxscript/runtime/py_commons/pystrtod.h>

namespace matxscript {
namespace runtime {

// Any
String Any::type_name() const {
  if (value_.code == TypeIndex::kRuntimeFTList || value_.code == TypeIndex::kRuntimeFTDict ||
      value_.code == TypeIndex::kRuntimeFTSet) {
    auto ty_idx = GetFTObjectBaseStdTypeIndex(ptr<Object>());
    return DemangleType(ty_idx.name());
  }
  return TypeIndex2Str(value_.code);
}

template <>
double Any::As<double>() const {
  // Allow automatic conversion from int to float
  // This avoids errors when user pass in int from
  // the frontend while the API expects a float.
  switch (value_.code) {
    case TypeIndex::kRuntimeInteger: {
      return static_cast<double>(value_.data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return value_.data.v_float64;
    } break;
    default: {
      MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeFloat);
      return 0;
    }
  }
}

template <>
double Any::AsNoCheck<double>() const {
  if (value_.code == TypeIndex::kRuntimeInteger) {
    return static_cast<double>(value_.data.v_int64);
  }
  return value_.data.v_float64;
}

template <>
bool Any::As<bool>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeInteger);
  return value_.data.v_int64 != 0;
}
template <>
bool Any::AsNoCheck<bool>() const {
  return value_.data.v_int64 != 0;
}

template <>
void* Any::As<void*>() const {
  if (value_.code == TypeIndex::kRuntimeNullptr)
    return nullptr;
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeOpaqueHandle);
  return value_.data.v_handle;
}

template <>
void* Any::AsNoCheck<void*>() const {
  if (value_.code == TypeIndex::kRuntimeNullptr)
    return nullptr;
  return value_.data.v_handle;
}

template <>
string_view Any::As<string_view>() const {
  if (value_.code == TypeIndex::kRuntimeNullptr)
    return {};
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeString);
  return StringHelper::AsViewNoCheck(&value_);
}
template <>
string_view Any::AsNoCheck<string_view>() const {
  if (value_.code == TypeIndex::kRuntimeNullptr)
    return {};
  return StringHelper::AsViewNoCheck(&value_);
}

template <>
unicode_view Any::As<unicode_view>() const {
  if (value_.code == TypeIndex::kRuntimeNullptr)
    return {};
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeUnicode);
  return UnicodeHelper::AsViewNoCheck(&value_);
}
template <>
unicode_view Any::AsNoCheck<unicode_view>() const {
  if (value_.code == TypeIndex::kRuntimeNullptr)
    return {};
  return UnicodeHelper::AsViewNoCheck(&value_);
}

template <>
DataType Any::As<DataType>() const {
  switch (value_.code) {
    case TypeIndex::kRuntimeString: {
      return DataType(String2DLDataType(AsNoCheck<string_view>()));
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return DataType(String2DLDataType(As<Unicode>().encode()));
    } break;
    default: {
      MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeDataType);
      return DataType(value_.data.v_type);
    } break;
  }
}

template <>
DataType Any::AsNoCheck<DataType>() const {
  switch (value_.code) {
    case TypeIndex::kRuntimeString: {
      return DataType(String2DLDataType(AsNoCheck<string_view>()));
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return DataType(String2DLDataType(As<Unicode>().encode()));
    } break;
    default: {
      return DataType(value_.data.v_type);
    } break;
  }
}

bool Any::IsString() const noexcept {
  return value_.code == TypeIndex::kRuntimeString;
}

bool Any::IsUnicode() const noexcept {
  return value_.code == TypeIndex::kRuntimeUnicode;
}

// RTView
RTView::RTView(const string_view& val) noexcept {
  value_.code = TypeIndex::kRuntimeString;
  if (val.category() >= 0) {
    constexpr auto max_len = sizeof(value_.data.v_str_store.v_small_bytes);
    assert(val.size() <= max_len);
    value_.pad = val.size();
    String::traits_type::copy(
        reinterpret_cast<char*>(value_.data.v_str_store.v_small_bytes), val.data(), val.size());
  } else {
    value_.pad = val.category();
    value_.data.v_str_store.v_ml.bytes = (unsigned char*)(val.data());
    value_.data.v_str_store.v_ml.size = val.size();
  }
}

RTView::RTView(const String& val) noexcept {
  value_.code = TypeIndex::kRuntimeString;
  val.data_.CopyTo(&value_.data.v_str_store, &value_.pad);
}

RTView::RTView(const unicode_view& val) noexcept {
  value_.code = TypeIndex::kRuntimeUnicode;
  if (val.category() >= 0) {
    constexpr auto max_len = sizeof(value_.data.v_str_store.v_small_chars) / sizeof(char32_t);
    assert(val.size() <= max_len);
    value_.pad = val.size();
    Unicode::traits_type::copy(value_.data.v_str_store.v_small_chars, val.data(), val.size());
  } else {
    value_.pad = val.category();
    value_.data.v_str_store.v_ml.chars = (char32_t*)(val.data());
    value_.data.v_str_store.v_ml.size = val.size();
  }
}

RTView::RTView(const Unicode& val) noexcept {
  value_.code = TypeIndex::kRuntimeUnicode;
  val.data_.CopyTo(&value_.data.v_str_store, &value_.pad);
}

RTView& RTView::operator=(const String& other) noexcept {
  value_.code = TypeIndex::kRuntimeString;
  other.data_.CopyTo(&value_.data.v_str_store, &value_.pad);
  return *this;
}

RTView& RTView::operator=(const Unicode& other) noexcept {
  value_.code = TypeIndex::kRuntimeUnicode;
  other.data_.CopyTo(&value_.data.v_str_store, &value_.pad);
  return *this;
}

RTView& RTView::operator=(const string_view& other) noexcept {
  return operator=(RTView(other));
}

RTView& RTView::operator=(const unicode_view& other) noexcept {
  return operator=(RTView(other));
}

// RTValue
RTValue::RTValue(const char* const str)
    : RTValue(String(str, std::char_traits<char>::length(str))) {
}

RTValue::RTValue(const char32_t* str)
    : RTValue(Unicode(str, std::char_traits<char32_t>::length(str))) {
}

RTValue& RTValue::operator=(String other) noexcept {
  this->Clear();
  other.MoveTo(&value_);
  return *this;
}

RTValue& RTValue::operator=(Unicode other) noexcept {
  this->Clear();
  other.MoveTo(&value_);
  return *this;
}

RTValue::RTValue(const Any& other) {
  CopyFromCHostToCHost(&other.value_, &value_);
}

RTValue::RTValue(const RTValue& other) : RTValue(static_cast<const Any&>(other)) {
}

RTValue::RTValue(const RTView& other) : RTValue(static_cast<const Any&>(other)) {
}

String RTValue::MoveToBytes() {
  if (value_.code != TypeIndex::kRuntimeString) {
    THROW_PY_TypeError("expect 'bytes' but get '", type_name(), "'");
  }
  return String::MoveFromNoCheck(&value_);
}

String RTValue::MoveToBytesNoCheck() {
  return String::MoveFromNoCheck(&value_);
}

Unicode RTValue::MoveToUnicode() {
  if (value_.code != TypeIndex::kRuntimeUnicode) {
    THROW_PY_TypeError("expect 'bytes' but get '", type_name(), "'");
  }
  return Unicode::MoveFromNoCheck(&value_);
}

Unicode RTValue::MoveToUnicodeNoCheck() {
  return Unicode::MoveFromNoCheck(&value_);
}

void RTValue::MoveToCHost(MATXScriptAny* ret_value) noexcept {
  *ret_value = value_;
  value_.code = TypeIndex::kRuntimeNullptr;
}

RTValue RTValue::MoveFromCHost(MATXScriptAny* value) noexcept {
  RTValue ret;
  ret.value_ = *value;
  value->code = TypeIndex::kRuntimeNullptr;
  return ret;
}

RTValue RTValue::MoveFromCHost(MATXScriptAny value) noexcept {
  RTValue ret;
  ret.value_ = value;
  return ret;
}

void RTValue::CopyFromCHostToCHost(const MATXScriptAny* from, MATXScriptAny* to) {
  switch (from->code) {
    case TypeIndex::kRuntimePackedFuncHandle: {
      auto* p = new NativeFunction(*reinterpret_cast<NativeFunction*>(from->data.v_handle));
      *to = *from;
      to->data.v_handle = p;
    } break;
    case TypeIndex::kRuntimeUnicode: {
      *to = UnicodeHelper::CopyFrom(from);
    } break;
    case TypeIndex::kRuntimeString: {
      *to = StringHelper::CopyFrom(from);
    } break;
    default: {
      if (from->code >= 0) {
        static_cast<Object*>(from->data.v_handle)->IncRef();
      }
      *to = *from;
    } break;
  }
}

RTValue RTValue::CopyFromCHost(MATXScriptAny value) {
  RTValue ret;
  CopyFromCHostToCHost(&value, &ret.value_);
  return ret;
}

RTValue RTValue::CopyFromCHost(const MATXScriptAny* value) {
  RTValue ret;
  CopyFromCHostToCHost(value, &ret.value_);
  return ret;
}

void RTValue::DestroyCHost(MATXScriptAny* value) noexcept {
  switch (value->code) {
    case TypeIndex::kRuntimePackedFuncHandle: {
      delete reinterpret_cast<NativeFunction*>(value->data.v_handle);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      UnicodeHelper::Destroy(value);
    } break;
    case TypeIndex::kRuntimeString: {
      StringHelper::Destroy(value);
    } break;
    default: {
      if (value->code >= 0) {
        static_cast<Object*>(value->data.v_handle)->DecRef();
      }
    } break;
  }
  value->code = TypeIndex::kRuntimeNullptr;
}

void RTValue::CopyToCHost(MATXScriptAny* ret_value) const {
  CopyFromCHostToCHost(&value_, ret_value);
}

bool RTValue::operator==(const RTValue& other) const {
  return Any::Equal(*this, other);
}

RTValue& RTValue::operator=(RTValue&& other) noexcept {
  this->Clear();
  value_ = other.value_;
  other.value_.code = TypeIndex::kRuntimeNullptr;
  return *this;
}
RTValue& RTValue::operator=(double value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeFloat);
  value_.data.v_float64 = value;
  return *this;
}
RTValue& RTValue::operator=(std::nullptr_t value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeNullptr);
  value_.data.v_handle = value;
  return *this;
}
RTValue& RTValue::operator=(DataType dtype) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeDataType);
  value_.data.v_type = dtype;
  return *this;
}
RTValue& RTValue::operator=(int64_t value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeInteger);
  value_.data.v_int64 = value;
  return *this;
}
RTValue& RTValue::operator=(uint64_t value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeInteger);
  value_.data.v_int64 = value;
  return *this;
}
RTValue& RTValue::operator=(int32_t value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeInteger);
  value_.data.v_int64 = value;
  return *this;
}
RTValue& RTValue::operator=(uint32_t value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeInteger);
  value_.data.v_int64 = value;
  return *this;
}
RTValue& RTValue::operator=(bool value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeInteger);
  value_.data.v_int64 = value;
  return *this;
}
RTValue& RTValue::operator=(void* value) noexcept {
  this->SwitchToPOD(TypeIndex::kRuntimeOpaqueHandle);
  value_.data.v_handle = value;
  return *this;
}
RTValue& RTValue::operator=(const char* str) {
  this->Clear();
  String obj(str, std::char_traits<char>::length(str));
  obj.MoveTo(&value_);
  return *this;
}
RTValue& RTValue::operator=(const char32_t* str) {
  this->Clear();
  Unicode obj(str, std::char_traits<char32_t>::length(str));
  obj.MoveTo(&value_);
  return *this;
}
RTValue& RTValue::operator=(const RTValue& other) {
  return operator=(RTValue(other));
}

void RTValue::SwitchToPOD(int type_code) noexcept {
  if (value_.code != type_code) {
    this->Clear();
    value_.code = type_code;
  }
}

void RTValue::SwitchToObject(int type_code, ObjectPtr<Object> other) noexcept {
  if (other.data_ != nullptr) {
    this->Clear();
    value_.code = type_code;
    // move the handle out
    value_.data.v_handle = other.data_;
    other.data_ = nullptr;
  } else {
    SwitchToPOD(TypeIndex::kRuntimeNullptr);
  }
}

std::ostream& operator<<(std::ostream& out, const Any& input) {
  auto type_code = input.type_code();
  switch (type_code) {
    case TypeIndex::kRuntimeInteger: {
      out << input.As<int64_t>();
    } break;
    case TypeIndex::kRuntimeFloat: {
      // from https://github.com/python/cpython/blob/3.7/Objects/floatobject.c#L315-L330
      out << py_builtins::PyOS_double_to_string(
          input.As<double>(), 'r', 0, py_builtins::Py_DTSF_ADD_DOT_0, NULL);
    } break;
    case TypeIndex::kRuntimeOpaqueHandle: {
      out << "OpaqueHandle(" << input.As<void*>() << ")";
    } break;
    case TypeIndex::kRuntimeNullptr: {
      out << "nullptr";
    } break;
    case TypeIndex::kRuntimeString: {
      out << input.AsNoCheck<string_view>();  // for str.format
      // out << "b'" << input.operator string_view() << "'";
    } break;
    case TypeIndex::kRuntimeUnicode: {
      out << input.AsNoCheck<unicode_view>();  // for str.format
      // out << "\"" << input.operator unicode_view() << "\"";
    } break;
    case TypeIndex::kRuntimeTuple: {
      out << input.AsObjectRefNoCheck<Tuple>();
    } break;
    case TypeIndex::kRuntimeList: {
      out << input.AsObjectRefNoCheck<List>();
    } break;
    case TypeIndex::kRuntimeDict: {
      out << input.AsObjectRefNoCheck<Dict>();
    } break;
    case TypeIndex::kRuntimeSet: {
      out << input.AsObjectRefNoCheck<Set>();
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeFTSet: {
      out << input.AsObjectRefNoCheck<FTObjectBase>();
    } break;
    case TypeIndex::kRuntimeNDArray: {
      out << input.AsObjectRefNoCheck<NDArray>();
    } break;
    case TypeIndex::kRuntimeUserData: {
      out << input.AsObjectRefNoCheck<UserDataRef>();
    } break;
    case TypeIndex::kRuntimePackedFuncHandle: {
      out << "PackedFunc(" << input.ptr<void>() << ")";
    } break;
    default: {
      out << "Object(" << input.ptr<Object>() << ")";
    } break;
  }
  return out;
}

namespace py_builtins {
// copy from https://github.com/python/cpython/blob/3.8/Objects/tupleobject.c#L353-L391
#if SIZEOF_PY_UHASH_T > 4
#define _PyHASH_XXPRIME_1 ((size_t)11400714785074694791ULL)
#define _PyHASH_XXPRIME_2 ((size_t)14029467366897019727ULL)
#define _PyHASH_XXPRIME_5 ((size_t)2870177450012600261ULL)
#define _PyHASH_XXROTATE(x) ((x << 31) | (x >> 33)) /* Rotate left 31 bits */
#else
#define _PyHASH_XXPRIME_1 ((size_t)2654435761UL)
#define _PyHASH_XXPRIME_2 ((size_t)2246822519UL)
#define _PyHASH_XXPRIME_5 ((size_t)374761393UL)
#define _PyHASH_XXROTATE(x) (((x) << 13) | ((x) >> 19)) /* Rotate left 13 bits */
#endif
static size_t tuple_hash(TupleNode* v) {
  size_t i, len = v->size;
  size_t acc = _PyHASH_XXPRIME_5;
  for (auto& item : *v) {
    size_t lane = Any::Hash(item);
    if (lane == (size_t)-1) {
      return -1;
    }
    acc += lane * _PyHASH_XXPRIME_2;
    acc = _PyHASH_XXROTATE(acc);
    acc *= _PyHASH_XXPRIME_1;
  }

  /* Add input length, mangled to keep the historical value of hash(()). */
  acc += len ^ (_PyHASH_XXPRIME_5 ^ 3527539UL);

  if (acc == (size_t)-1) {
    return 1546275796;
  }
  return acc;
}
}  // namespace py_builtins

std::size_t Any::Hash(const Any& a) {
  const MATXScriptAny& value = a.value();
  switch (value.code) {
    case TypeIndex::kRuntimeNullptr: {
      return 0;
    } break;
    case TypeIndex::kRuntimeOpaqueHandle: {
      auto ptr = reinterpret_cast<uint64_t>(value.data.v_handle);
      return ScalarHash<uint64_t>()(ptr);
    } break;
    case TypeIndex::kRuntimeInteger: {
      return ScalarHash<int64_t>()(value.data.v_int64);
    } break;
    case TypeIndex::kRuntimeFloat: {
      return ScalarHash<double>()(value.data.v_float64);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return std::hash<unicode_view>()(UnicodeHelper::AsViewNoCheck(&value));
    } break;
    case TypeIndex::kRuntimeString: {
      return std::hash<string_view>()(StringHelper::AsViewNoCheck(&value));
    } break;
    case TypeIndex::kRuntimeDataType: {
      return ScalarHash<DLDataType>()(value.data.v_type);
    } break;
    case TypeIndex::kRuntimeObjectRValueRefArg: {
      MXTHROW << "TypeError: unhashable type: 'ObjectRValueRefArg'";
      return false;
    } break;
    case TypeIndex::kRuntimePackedFuncHandle: {
      MXTHROW << "TypeError: unhashable type: 'PackedFunc'";
      return false;
    } break;
    case TypeIndex::kRuntimeDLTensorHandle: {
      MXTHROW << "TypeError: unhashable type: 'DLTensorHandle'";
      return false;
    } break;
    case TypeIndex::kRuntimeContext: {
      MXTHROW << "TypeError: unhashable type: 'Context'";
      return false;
    } break;
    case TypeIndex::kMATXByteArray: {
      MXTHROW << "TypeError: unhashable type: 'ByteArray'";
      return false;
    } break;
    case TypeIndex::kRuntimeFTList:
    case TypeIndex::kRuntimeList: {
      MXTHROW << "TypeError: unhashable type: 'list'";
    } break;
    case TypeIndex::kRuntimeFTDict:
    case TypeIndex::kRuntimeDict: {
      MXTHROW << "TypeError: unhashable type: 'dict'";
    } break;
    case TypeIndex::kRuntimeFTSet:
    case TypeIndex::kRuntimeSet: {
      MXTHROW << "TypeError: unhashable type: 'set'";
    } break;
    case TypeIndex::kRuntimeTuple: {
      return py_builtins::tuple_hash(reinterpret_cast<TupleNode*>(value.data.v_handle));
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud_ref = a.AsNoCheck<UserDataRef>();
      if (ud_ref->ud_ptr->type_2_71828182846() == UserDataStructType::kUserData) {
        auto ud_ptr = ((IUserDataRoot*)(ud_ref->ud_ptr));
        auto f_table_itr = ud_ptr->function_table_2_71828182846_->find("__hash__");
        if (f_table_itr == ud_ptr->function_table_2_71828182846_->end()) {
          // ptr hash
          auto ptr = reinterpret_cast<uint64_t>(ud_ref->ud_ptr);
          return ScalarHash<uint64_t>()(ptr);
        } else {
          // call __hash__
          return ud_ref->generic_call_attr("__hash__", {}).As<int64_t>();
        }
      } else {
        auto ptr = reinterpret_cast<uint64_t>(ud_ref->ud_ptr);
        return ScalarHash<uint64_t>()(ptr);
      }
    } break;
    case TypeIndex::kRuntimeOpaqueObject: {
      auto* node_ptr = reinterpret_cast<OpaqueObjectNode*>(value.data.v_handle);
      uint64_t ptr = reinterpret_cast<uint64_t>(node_ptr->ptr);
      return ScalarHash<uint64_t>()(ptr);
    } break;
    default: {
      return ScalarHash<uint64_t>()(reinterpret_cast<uint64_t>(value.data.v_handle));
    } break;
  }
  return 0;
}

bool Any::Equal(const Any& lhs, const Any& rhs) {
  return ArithOps::eq(lhs, rhs);
}

bool Any::LessThan(const Any& lhs, const Any& rhs) {
  return ArithOps::lt(lhs, rhs);
}

bool Any::LessEqual(const Any& lhs, const Any& rhs) {
  return ArithOps::le(lhs, rhs);
}

bool Any::GreaterThan(const Any& lhs, const Any& rhs) {
  return ArithOps::gt(lhs, rhs);
}

bool Any::GreaterEqual(const Any& lhs, const Any& rhs) {
  return ArithOps::ge(lhs, rhs);
}

const RTValue None;

}  // namespace runtime
}  // namespace matxscript
