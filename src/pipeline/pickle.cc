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
#include <matxscript/pipeline/pickle.h>

#include <unordered_map>

#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/json_util.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {
namespace pickle {

/******************************************************************************
 * utf-8 json, no node info
 *****************************************************************************/

List FromJsonArray(const rapidjson::Value& val, bool use_unicode) {
  MXCHECK(val.IsArray());
  List ret;
  auto json_array = val.GetArray();
  ret.reserve(json_array.Size());
  for (auto& item : json_array) {
    ret.push_back(FromJson(item, use_unicode));
  }
  return ret;
}

Dict FromJsonDict(const rapidjson::Value& val, bool use_unicode) {
  MXCHECK(val.IsObject());
  Dict ret;
  auto json_obj = val.GetObject();
  for (auto& item : json_obj) {
    RTValue name = FromJson(item.name, use_unicode);
    RTValue value = FromJson(item.value, use_unicode);
    ret.set_item(name, value);
  }
  return ret;
}

RTValue FromJson(const rapidjson::Value& val, bool use_unicode) {
  switch (val.GetType()) {
    case rapidjson::kNullType: {
      return RTValue();
    } break;
    case rapidjson::kFalseType: {
      return RTValue(false);
    } break;
    case rapidjson::kTrueType: {
      return RTValue(true);
    } break;
    case rapidjson::kObjectType: {
      return FromJsonDict(val, use_unicode);
    } break;
    case rapidjson::kArrayType: {
      return FromJsonArray(val, use_unicode);
    } break;
    case rapidjson::kStringType: {
      if (use_unicode) {
        return String(val.GetString(), val.GetStringLength()).decode();
      } else {
        return String(val.GetString(), val.GetStringLength());
      }
    } break;
    case rapidjson::kNumberType: {
      char buf[256] = {0};
      if (val.IsDouble()) {
        return RTValue(val.GetDouble());
      } else if (val.IsInt()) {
        return RTValue(val.GetInt());
      } else if (val.IsUint()) {
        return RTValue(static_cast<int64_t>(val.GetUint()));
      } else if (val.IsInt64()) {
        return RTValue(val.GetInt64());
      } else {
        return RTValue(static_cast<int64_t>(val.GetUint64()));
      }
    } break;
    default: {
      return RTValue();
    } break;
  }
}

void ToJsonList(const List& rtv,
                rapidjson::Value& json_val,
                rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetArray();
  for (auto& item : rtv) {
    rapidjson::Value json_item;
    ToJson(item, json_item, allocator);
    json_val.PushBack(json_item, allocator);
  }
}

void ToJsonDict(const Dict& rtv,
                rapidjson::Value& json_val,
                rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetObject();
  for (auto item : rtv.items()) {
    rapidjson::Value json_k_item;
    rapidjson::Value json_v_item;
    String skey;
    switch (item.first.type_code()) {
      // TODO(wuxian): fix bool
      // case TypeIndex::kRuntimeBool:
      case TypeIndex::kRuntimeInteger:
      case TypeIndex::kRuntimeFloat:
      case TypeIndex::kRuntimeUnicode: {
        skey = Kernel_Unicode::make(item.first).encode();
      } break;
      case TypeIndex::kRuntimeNullptr: {
        skey = "null";
      } break;
      default:
        MXCHECK(false) << "keys must be str, int, float, bool or None, not "
                       << item.first.type_name();
    }
    json_k_item.SetString(skey.c_str(), skey.length(), allocator);
    ToJson(item.second, json_v_item, allocator);
    json_val.AddMember(json_k_item, json_v_item, allocator);
  }
}

void ToJson(const Any& rtv,
            rapidjson::Value& json_val,
            rapidjson::MemoryPoolAllocator<>& allocator) {
  switch (rtv.type_code()) {
    case TypeIndex::kRuntimeNullptr: {
      json_val.SetNull();
    } break;
    case TypeIndex::kRuntimeInteger: {
      json_val.SetInt64(rtv.As<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      json_val.SetDouble(rtv.As<double>());
    } break;
    case TypeIndex::kRuntimeString: {
      auto s = rtv.As<String>();
      json_val.SetString(s.data(), s.length(), allocator);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      auto u = rtv.As<Unicode>();
      auto s = u.encode();
      json_val.SetString(s.data(), s.length(), allocator);
    } break;
    case TypeIndex::kRuntimeList: {
      auto vl = rtv.AsObjectRefNoCheck<List>();
      ToJsonList(vl, json_val, allocator);
    } break;
    case TypeIndex::kRuntimeDict: {
      auto vl = rtv.AsObjectRefNoCheck<Dict>();
      ToJsonDict(vl, json_val, allocator);
    } break;
    default: {
      MXCHECK(false) << "[ToJson] unsupported runtime value type: " << rtv.type_name();
    } break;
  }
}

/******************************************************************************
 * every node is like this:
 * {
 *    "t": "str",
 *    "v": "abc",
 * }
 *****************************************************************************/
List FromJsonStructArray(const rapidjson::Value& val) {
  MXCHECK(val.IsArray());
  List ret;
  auto json_array = val.GetArray();
  ret.reserve(json_array.Size());
  for (auto& item : json_array) {
    ret.push_back(FromJsonStruct(item));
  }
  return ret;
}

Dict FromJsonStructDict(const rapidjson::Value& val) {
  MXCHECK(val.IsArray() && val.Size() % 2 == 0);
  Dict ret;
  for (auto itr = val.Begin(); itr != val.End(); itr += 1) {
    RTValue name = FromJsonStruct(*itr);
    itr += 1;
    RTValue value = FromJsonStruct(*itr);
    ret.set_item(name, value);
  }
  return ret;
}

Set FromJsonStructSet(const rapidjson::Value& val) {
  MXCHECK(val.IsArray());
  Set ret;
  auto json_array = val.GetArray();
  ret.reserve(json_array.Size());
  for (auto& item : json_array) {
    ret.emplace(FromJsonStruct(item));
  }
  return ret;
}

UserDataRef FromJsonStructUserData(const rapidjson::Value& val) {
  MXCHECK(val.IsObject());
  uint32_t tag = JSON_GET(val, "tag", Uint);
  int32_t type = JSON_GET(val, "type", Uint);
  uint32_t var_num = JSON_GET(val, "var_num", Uint);
  // std::uintptr_t ud_ptr = JSON_GET(val, "ud_ptr", Uint64);
  MXCHECK(type == UserDataStructType::kNativeData) << "only native op can be load from json";
  bool is_native_op = JSON_GET(val, "is_native_op", Bool);
  MXCHECK(is_native_op) << "only native op can be load from json";
  bool is_jit_object = JSON_GET(val, "is_jit_object", Bool);
  String native_class_name = JSON_GET(val, "native_class_name", String);
  String native_instance_name = JSON_GET(val, "native_instance_name", String);
  NativeObject* nud_ptr = new NativeObject();
  nud_ptr->is_jit_object_ = is_jit_object;
  nud_ptr->is_native_op_ = is_native_op;
  nud_ptr->native_class_name_ = native_class_name;
  nud_ptr->native_instance_name_ = native_instance_name;
  UserDataRef ret(tag, var_num, reinterpret_cast<void*>(nud_ptr), default_userdata_deleter);
  return ret;
}

Tuple FromJsonStructADT(const rapidjson::Value& val) {
  MXCHECK(val.IsObject());
  int32_t size = JSON_GET(val, "size", Uint);
  auto data = JSON_GET(val, "data", Array);
  std::vector<RTValue> fields;
  fields.reserve(data.Size());
  for (auto itr = data.Begin(); itr != data.End(); itr += 1) {
    RTValue item = FromJsonStruct(*itr);
    fields.push_back(item);
  }
  return Tuple(std::make_move_iterator(fields.begin()), std::make_move_iterator(fields.end()));
}

namespace {
template <typename T>
T AsValue(const rapidjson::Value& val);

template <>
int32_t AsValue<int32_t>(const rapidjson::Value& val) {
  return val.GetInt();
}

template <>
int64_t AsValue<int64_t>(const rapidjson::Value& val) {
  return val.GetInt64();
}

template <>
float AsValue<float>(const rapidjson::Value& val) {
  return val.GetFloat();
}

template <>
double AsValue<double>(const rapidjson::Value& val) {
  return val.GetDouble();
}

template <>
uint8_t AsValue<uint8_t>(const rapidjson::Value& val) {
  return static_cast<uint8_t>(val.GetInt());
}

template <>
uint16_t AsValue<uint16_t>(const rapidjson::Value& val) {
  return static_cast<uint16_t>(val.GetInt());
}

template <>
int8_t AsValue<int8_t>(const rapidjson::Value& val) {
  return static_cast<int8_t>(val.GetInt());
}

template <>
int16_t AsValue<int16_t>(const rapidjson::Value& val) {
  return static_cast<int16_t>(val.GetInt());
}

template <>
Half AsValue<Half>(const rapidjson::Value& val) {
  return static_cast<Half>(val.GetFloat());
}

template <typename T>
void GetNumericDataFromJson(const rapidjson::Value& val, T* data, size_t ele_num) {
  auto data_array = val.GetArray();
  size_t k = 0;
  for (auto itr = data_array.Begin(); itr != data_array.End(); itr += 1) {
    if (k >= ele_num) {
      THROW_PY_IndexError("NDArray index out of range, ", k, " >= ", ele_num);
    }
    data[k] = AsValue<T>(*itr);
    ++k;
  }
}

template <typename T, typename Allocator>
void FromValue(rapidjson::Value* ret, Allocator& allocator, const T& val) {
  ret->Set<T>(val, allocator);
}

template <>
void FromValue(rapidjson::Value* ret,
               rapidjson::MemoryPoolAllocator<>& allocator,
               const uint16_t& val) {
  ret->Set(static_cast<uint32_t>(val), allocator);
}

template <>
void FromValue(rapidjson::Value* ret,
               rapidjson::MemoryPoolAllocator<>& allocator,
               const int16_t& val) {
  ret->Set(static_cast<int32_t>(val), allocator);
}

template <>
void FromValue(rapidjson::Value* ret,
               rapidjson::MemoryPoolAllocator<>& allocator,
               const uint8_t& val) {
  ret->Set(static_cast<uint32_t>(val), allocator);
}

template <>
void FromValue(rapidjson::Value* ret,
               rapidjson::MemoryPoolAllocator<>& allocator,
               const int8_t& val) {
  ret->Set(static_cast<int32_t>(val), allocator);
}

template <>
void FromValue(rapidjson::Value* ret,
               rapidjson::MemoryPoolAllocator<>& allocator,
               const Half& val) {
  ret->Set(static_cast<float>(val), allocator);
}

}  // namespace

NDArray FromJsonStructNDArray(const rapidjson::Value& val) {
  MXCHECK(val.IsObject());
  MXCHECK(val.HasMember("dtype"));
  MXCHECK(val.HasMember("shape"));
  MXCHECK(val.HasMember("data"));

  auto dtype_str = JSON_GET(val, "dtype", String);
  DataType dtype(String2DLDataType(dtype_str));

  auto shape = JSON_GET(val, "shape", Array);

  std::vector<int64_t> shape_list;
  shape_list.reserve(shape.Size());
  for (auto itr = shape.Begin(); itr != shape.End(); itr += 1) {
    shape_list.push_back(itr->GetInt64());
  }

  DLDevice device{kDLCPU, 0};
  auto tensor = NDArray::Empty(shape_list, dtype, device);
  MATX_NDARRAY_TYPE_SWITCH(dtype, DType, {
    auto* data_ptr = tensor.Data<DType>();
    GetNumericDataFromJson<DType>(val["data"], const_cast<DType*>(data_ptr), tensor.ElementSize());
  });
  return tensor;
}

void* FromJsonStructOpaqueData(const rapidjson::Value& val) {
  MXCHECK(val.IsObject());
  std::uintptr_t user_ptr = JSON_GET(val, "user_ptr", Uint64);
  return reinterpret_cast<void*>(user_ptr);
}

RTValue FromJsonStruct(const rapidjson::Value& struct_json_node) {
  MXCHECK(struct_json_node.IsObject()) << "JsonStruct node should be always object";
  auto itr_find_t = struct_json_node.FindMember("t");
  auto itr_find_v = struct_json_node.FindMember("v");
  MXCHECK(struct_json_node.MemberCount() == 2 && itr_find_t != struct_json_node.MemberEnd() &&
          itr_find_v != struct_json_node.MemberEnd());
  auto& type = itr_find_t->value;
  auto& val = itr_find_v->value;
  MXCHECK(type.IsString());
  string_view type_str_v(type.GetString(), type.GetStringLength());
  // for compatible with versions 1.5 and earlier
  if (type_str_v == "ADT") {
    type_str_v = "Tuple";
  }
  int32_t type_code = Str2TypeIndex(type_str_v);
  switch (type_code) {
    case TypeIndex::kRuntimeInteger: {
      return RTValue(val.GetInt64());
    } break;
    case TypeIndex::kRuntimeFloat: {
      return RTValue(val.GetDouble());
    } break;
    case TypeIndex::kRuntimeNullptr: {
      return RTValue();
    } break;
    case TypeIndex::kRuntimeString: {
      if (val.GetType() == rapidjson::kStringType) {
        return String(val.GetString(), val.GetStringLength());
      } else {
        auto json_array = val.GetArray();
        String ret;
        ret.reserve(json_array.Size());
        for (auto& item : json_array) {
          ret.push_back(item.GetUint());
        }
        return ret;
      }
    } break;
    case TypeIndex::kRuntimeUnicode: {
      return String(val.GetString(), val.GetStringLength()).decode();
    } break;
    case TypeIndex::kRuntimeUserData: {
      return FromJsonStructUserData(val);
    } break;
    case TypeIndex::kRuntimeOpaqueHandle: {
      void* user_ptr = FromJsonStructOpaqueData(val);
      return RTValue(user_ptr);
    } break;
    case TypeIndex::kRuntimeList: {
      return FromJsonStructArray(val);
    } break;
    case TypeIndex::kRuntimeDict: {
      return FromJsonStructDict(val);
    } break;
    case TypeIndex::kRuntimeSet: {
      return FromJsonStructSet(val);
    } break;
    case TypeIndex::kRuntimeTuple: {
      return FromJsonStructADT(val);
    } break;
    case TypeIndex::kRuntimeNDArray: {
      return FromJsonStructNDArray(val);
    } break;
    default: {
      MXCHECK(false) << "[FromJsonStruct] unsupported runtime value type: " << type_str_v;
      return RTValue();
    } break;
  }
}

void ToJsonStructList(const List& rtv,
                      rapidjson::Value& json_val,
                      rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetArray();
  for (auto& item : rtv) {
    rapidjson::Value json_item;
    ToJsonStruct(item, json_item, allocator);
    json_val.PushBack(json_item, allocator);
  }
}

void ToJsonStructSet(const Set& rtv,
                     rapidjson::Value& json_val,
                     rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetArray();
  for (auto& item : rtv) {
    rapidjson::Value json_item;
    ToJsonStruct(item, json_item, allocator);
    json_val.PushBack(json_item, allocator);
  }
}

void ToJsonStructDict(const Dict& rtv,
                      rapidjson::Value& json_val,
                      rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetArray();
  for (auto item : rtv.items()) {
    rapidjson::Value json_k_item;
    rapidjson::Value json_v_item;
    ToJsonStruct(item.first, json_k_item, allocator);
    ToJsonStruct(item.second, json_v_item, allocator);
    json_val.PushBack(json_k_item, allocator);
    json_val.PushBack(json_v_item, allocator);
  }
}

void ToJsonStructUserData(const UserDataRef& rtv,
                          rapidjson::Value& json_val,
                          rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetObject();
  JsonUtil::Set(&json_val, "tag", rtv->tag, allocator);
  JsonUtil::Set(&json_val, "type", rtv->ud_ptr->type_2_71828182846(), allocator);
  JsonUtil::Set(&json_val, "class_name", rtv->ud_ptr->ClassName_2_71828182846(), allocator);
  JsonUtil::Set(&json_val, "var_num", rtv->var_num, allocator);
  json_val.AddMember("ud_ptr", reinterpret_cast<uint64_t>(rtv->ud_ptr), allocator);
  MXCHECK(rtv->ud_ptr->type_2_71828182846() == UserDataStructType::kNativeData)
      << "[Class: " << rtv->ud_ptr->ClassName_2_71828182846()
      << "] [type: " << rtv->ud_ptr->type_2_71828182846()
      << "] does not support serialization. Please check whether it is used in the __init__ function of an op or as a constant symbol of the pipeline!!!";
  auto* nud_ptr = dynamic_cast<NativeObject*>(rtv->ud_ptr);
  MXCHECK(nud_ptr->is_native_op_)
      << "[Class: " << rtv->ud_ptr->ClassName_2_71828182846()
      << "] [type: " << rtv->ud_ptr->type_2_71828182846()
      << "] does not support serialization. Please check whether it is used in the __init__ function of an op or as a constant symbol of the pipeline!!!";
  JsonUtil::Set(&json_val, "is_native_op", nud_ptr->is_native_op_, allocator);
  JsonUtil::Set(&json_val, "is_jit_object", nud_ptr->is_jit_object_, allocator);
  JsonUtil::Set(&json_val, "native_class_name", nud_ptr->native_class_name_.view(), allocator);
  JsonUtil::Set(
      &json_val, "native_instance_name", nud_ptr->native_instance_name_.view(), allocator);
}

void ToJsonStructADT(const Tuple& rtv,
                     rapidjson::Value& json_val,
                     rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetObject();
  JsonUtil::Set(&json_val, "size", rtv.size(), allocator);
  rapidjson::Value data_item;
  data_item.SetArray();
  for (size_t i = 0; i < rtv.size(); ++i) {
    rapidjson::Value json_item;
    ToJsonStruct(rtv.get_item(i), json_item, allocator);
    data_item.PushBack(json_item, allocator);
  }
  JsonUtil::Set(&json_val, "data", data_item, allocator);
}

void ToJsonStructNDArray(const NDArray& rtv,
                         rapidjson::Value& json_val,
                         rapidjson::MemoryPoolAllocator<>& allocator) {
  MXCHECK(rtv.IsContiguous()) << "only contiguous ndarray supports serialization.";
  json_val.SetObject();
  JsonUtil::Set(&json_val, "dtype", rtv.DTypeUnicode().encode().c_str(), allocator);

  rapidjson::Value shape_item;
  shape_item.SetArray();
  auto shape = rtv.Shape();
  for (int64_t dim : shape) {
    shape_item.PushBack(dim, allocator);
  }
  JsonUtil::Set(&json_val, "shape", shape_item, allocator);

  rapidjson::Value data_item;
  data_item.SetArray();
  int64_t elem_size = rtv.ElementSize();

  MATX_NDARRAY_TYPE_SWITCH(rtv->dtype, DType, {
    const DType* data = rtv.Data<DType>();
    for (int64_t i = 0; i < elem_size; ++i) {
      rapidjson::Value j_data;
      FromValue(&j_data, allocator, data[i]);
      data_item.PushBack(std::move(j_data), allocator);
    }
  });
  JsonUtil::Set(&json_val, "data", data_item, allocator);
}

void ToJsonStructOpaqueData(void* user_ptr,
                            rapidjson::Value& json_val,
                            rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetObject();
  json_val.AddMember("user_ptr", reinterpret_cast<uint64_t>(user_ptr), allocator);
}

void ToJsonStructPackedFunc(void* ptr,
                            rapidjson::Value& json_val,
                            rapidjson::MemoryPoolAllocator<>& allocator) {
  json_val.SetObject();
  json_val.AddMember("ptr", reinterpret_cast<uint64_t>(ptr), allocator);
}

void ToJsonStruct(const Any& rtv,
                  rapidjson::Value& json_val,
                  rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value json_val_v;
  auto type_str = rtv.type_name();
  rapidjson::Value json_val_t;
  json_val_t.SetString(type_str.data(), type_str.size(), allocator);
  switch (rtv.type_code()) {
    case TypeIndex::kRuntimeNullptr: {
      json_val_v.SetNull();
    } break;
    case TypeIndex::kRuntimeInteger: {
      json_val_v.SetInt64(rtv.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      json_val_v.SetDouble(rtv.AsNoCheck<double>());
    } break;
    case TypeIndex::kRuntimeString: {
      auto s = rtv.AsNoCheck<string_view>();
      if (std::any_of(s.begin(), s.end(), [](char c) {
            return static_cast<unsigned char>(c) >= 127 || static_cast<unsigned char>(c) == 0;
          })) {
        json_val_v.SetArray();
        const unsigned char* d = reinterpret_cast<const unsigned char*>(s.data());
        for (size_t i = 0; i < s.length(); ++i) {
          rapidjson::Value json_item;
          json_item.SetUint(d[i]);
          json_val_v.PushBack(json_item, allocator);
        }
      } else {
        json_val_v.SetString(s.data(), s.length(), allocator);
      }
    } break;
    case TypeIndex::kRuntimeUnicode: {
      auto u = rtv.AsNoCheck<unicode_view>();
      auto s = UnicodeHelper::Encode(u);
      json_val_v.SetString(s.data(), s.length(), allocator);
    } break;
    case TypeIndex::kRuntimeUserData: {
      auto ud = rtv.AsObjectRefNoCheck<UserDataRef>();
      ToJsonStructUserData(ud, json_val_v, allocator);
    } break;
    case TypeIndex::kRuntimeOpaqueHandle: {
      void* user_ptr = rtv.AsNoCheck<void*>();
      ToJsonStructOpaqueData(user_ptr, json_val_v, allocator);
    } break;
    case TypeIndex::kRuntimePackedFuncHandle: {
      void* user_ptr = rtv.ptr<void>();
      ToJsonStructPackedFunc(user_ptr, json_val_v, allocator);
    } break;
    case TypeIndex::kRuntimeList: {
      auto vl = rtv.AsObjectRefNoCheck<List>();
      ToJsonStructList(vl, json_val_v, allocator);
    } break;
    case TypeIndex::kRuntimeSet: {
      auto vl = rtv.AsObjectRefNoCheck<Set>();
      ToJsonStructSet(vl, json_val_v, allocator);
    } break;
    case TypeIndex::kRuntimeDict: {
      auto vl = rtv.AsObjectRefNoCheck<Dict>();
      ToJsonStructDict(vl, json_val_v, allocator);
    } break;
    case TypeIndex::kRuntimeNDArray: {
      auto vl = rtv.AsObjectRefNoCheck<NDArray>();
      ToJsonStructNDArray(vl, json_val_v, allocator);
    } break;
    case TypeIndex::kRuntimeTuple: {
      auto vl = rtv.AsObjectRefNoCheck<Tuple>();
      ToJsonStructADT(vl, json_val_v, allocator);
    } break;
    default: {
      MXTHROW << "[ToJson] unsupported runtime value type: " << rtv.type_name();
    } break;
  }
  json_val.SetObject();
  json_val.AddMember("t", json_val_t, allocator);
  json_val.AddMember("v", json_val_v, allocator);
}

static constexpr const char* MATX4_SERIALIZE_VERSION = "v1.0";

String Serialize(const Any& value) {
  rapidjson::Document doc;
  doc.SetObject();
  JsonUtil::Set(&doc, "version", MATX4_SERIALIZE_VERSION, doc.GetAllocator());
  rapidjson::Value js_val;
  ToJsonStruct(value, js_val, doc.GetAllocator());
  JsonUtil::Set(&doc, "doc", js_val, doc.GetAllocator());
  return JsonUtil::ToString(&doc, false);
}

RTValue DeSerialize(const string_view& str) {
  rapidjson::Document doc;
  ::matxscript::runtime::JsonUtil::FromString(str, doc);
  MXCHECK(doc.HasMember("version") && doc.HasMember("doc"))
      << "Serialized json object should have key `version` and `data` ";
  String version = JSON_GET(doc, "version", String);
  MXCHECK(version == MATX4_SERIALIZE_VERSION) << "Serialization doesn't match, supported "
                                              << MATX4_SERIALIZE_VERSION << " ,but got " << version;
  RTValue v = FromJsonStruct(doc["doc"]);
  return v;
}

}  // namespace pickle
}  // namespace runtime
}  // namespace matxscript
