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
#include <matxscript/runtime/json_util.h>

#include <fstream>

#include <rapidjson/error/en.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/writer.h>

#include <matxscript/runtime/logging.h>
namespace matxscript {
namespace runtime {
namespace JsonUtil {

/*********************************************************************
 * Json Value serialization and deserialization
 *********************************************************************/
bool FromString(string_view json_str, rapidjson::Document& doc) {
  constexpr unsigned flag = rapidjson::kParseNanAndInfFlag;
  if (doc.Parse<flag>(json_str.data(), json_str.size()).HasParseError()) {
    MXTHROW << "Error(offset " << doc.GetErrorOffset()
            << "): " << GetParseError_En(doc.GetParseError());
    return false;
  }
  return true;
}

bool FromFile(string_view filepath, rapidjson::Document& doc) {
  std::ifstream ifs(filepath.data(), std::ios::in | std::ios::binary);
  if (!ifs) {
    MXTHROW << "Can't open the file. Please check " << filepath;
  }
  ifs.seekg(0, std::ios::end);
  auto length = static_cast<std::size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> buffer;
  buffer.reset(new char[length + 1]);
  buffer.get()[length] = '\0';
  ifs.read(buffer.get(), length);
  ifs.close();
  return FromString(buffer.get(), doc);
}

String ToString(const rapidjson::Value* val, bool pretty, bool escape, int indent) {
  switch (val->GetType()) {
    case rapidjson::kNullType:
      return String();
    case rapidjson::kFalseType:
      return String("false");
    case rapidjson::kTrueType:
      return String("true");
    case rapidjson::kObjectType:
    case rapidjson::kArrayType: {
      const unsigned flag = rapidjson::kWriteValidateEncodingFlag | rapidjson::kWriteNanAndInfFlag;
      rapidjson::StringBuffer buffer(nullptr, 10485760);
      if (escape) {
        if (pretty) {
          rapidjson::PrettyWriter<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::ASCII<>>
              writer(buffer);
          writer.SetIndent(' ', indent);
          // writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
          MXCHECK(val->Accept(writer)) << "ToString failed! val type: " << val->GetType();
        } else {
          rapidjson::Writer<rapidjson::StringBuffer,
                            rapidjson::UTF8<>,
                            rapidjson::ASCII<>,
                            rapidjson::CrtAllocator,
                            flag>
              writer(buffer);
          MXCHECK(val->Accept(writer)) << "ToString failed! val type: " << val->GetType();
        }

      } else {
        if (pretty) {
          rapidjson::PrettyWriter<rapidjson::StringBuffer, rapidjson::UTF8<>, rapidjson::UTF8<>>
              writer(buffer);
          writer.SetIndent(' ', 2);
          // writer.SetFormatOptions(rapidjson::kFormatSingleLineArray);
          MXCHECK(val->Accept(writer)) << "ToString failed! val type: " << val->GetType();
        } else {
          rapidjson::Writer<rapidjson::StringBuffer,
                            rapidjson::UTF8<>,
                            rapidjson::UTF8<>,
                            rapidjson::CrtAllocator,
                            flag>
              writer(buffer);
          MXCHECK(val->Accept(writer)) << "ToString failed! val type: " << val->GetType();
        }
      }
      return String(buffer.GetString(), buffer.GetSize());
    }
    case rapidjson::kStringType:
      return String(val->GetString(), val->GetStringLength());
    case rapidjson::kNumberType: {
      char buf[256] = {0};
      if (val->IsDouble()) {
        if (sizeof(uintptr_t) != sizeof(uint64_t)) {
          snprintf(buf, sizeof(buf), "%.6g", val->GetDouble());
        } else {
          snprintf(buf, sizeof(buf), "%.16g", val->GetDouble());
        }
        return String(buf);
      } else if (val->IsInt()) {
        snprintf(buf, sizeof(buf), "%d", val->GetInt());
        return String(buf);
      } else if (val->IsUint()) {
        snprintf(buf, sizeof(buf), "%u", val->GetUint());
        return String(buf);
      } else if (val->IsInt64()) {
        snprintf(buf, sizeof(buf), "%lld", val->GetInt64());
        return String(buf);
      } else {
        snprintf(buf, sizeof(buf), "%llu", val->GetUint64());
        return String(buf);
      }
    }
    default:
      break;
  }
  return String();
}

/*********************************************************************
 * Json Array Add
 *********************************************************************/
void Add(rapidjson::Value* obj, string_view value, rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value obj_val(value.data(), value.size(), allocator);
  obj->PushBack(obj_val, allocator);
}

void Add(rapidjson::Value* obj,
         const rapidjson::Value& value,
         rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value obj_val(value, allocator);
  obj->PushBack(obj_val, allocator);
}

void Add(rapidjson::Value* obj,
         rapidjson::Value&& value,
         rapidjson::MemoryPoolAllocator<>& allocator) {
  obj->PushBack(std::move(value), allocator);
}

/*********************************************************************
 * Json Object Set Function
 *********************************************************************/
void Set(rapidjson::Value* obj,
         rapidjson::Value& key,
         rapidjson::Value& value,
         rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value::MemberIterator itr = obj->FindMember(key);
  if (itr == obj->MemberEnd()) {
    obj->AddMember(key, value, allocator);
  } else {
    itr->value = value;
  }
}

void Set(rapidjson::Value* obj,
         const char* key,
         string_view value,
         rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value obj_key(key, allocator);
  rapidjson::Value obj_val(value.data(), value.size(), allocator);
  return Set(obj, obj_key, obj_val, allocator);
}

void Set(rapidjson::Value* obj,
         const char* key,
         const char* value,
         rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value obj_key(key, allocator);
  rapidjson::Value obj_val(value, allocator);
  return Set(obj, obj_key, obj_val, allocator);
}

void Set(rapidjson::Value* obj,
         const char* key,
         rapidjson::Value& value,
         rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value obj_key(key, allocator);
  return Set(obj, obj_key, value, allocator);
}

}  // namespace JsonUtil
}  // namespace runtime
}  // namespace matxscript
