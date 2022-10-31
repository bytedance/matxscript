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

#include <cstdio>
#include <stdexcept>

#include <matxscript/runtime/logging.h>
#ifndef RAPIDJSON_ASSERT
#define RAPIDJSON_ASSERT(x) MXCHECK(x)
#endif
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/file_util.h>
#include <rapidjson/document.h>

/**
 * util class for rapidjson.
 */
namespace matxscript {
namespace runtime {
namespace JsonUtil {

/*********************************************************************
 * Json Value serialization and deserialization
 *********************************************************************/
extern bool FromString(string_view json_str, rapidjson::Document& doc);

extern bool FromFile(string_view filepath, rapidjson::Document& doc);

extern String ToString(const rapidjson::Value* val,
                       bool pretty = false,
                       bool escape = false,
                       int indent = 2);

/*********************************************************************
 * Json Array Add
 *********************************************************************/
template <class T_POD, typename = std::enable_if<std::is_pod<T_POD>::value>>
static inline void Add(rapidjson::Value* obj,
                       T_POD value,
                       rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value obj_val(value);
  obj->PushBack(obj_val, allocator);
}

extern void Add(rapidjson::Value* obj,
                string_view value,
                rapidjson::MemoryPoolAllocator<>& allocator);

extern void Add(rapidjson::Value* obj,
                rapidjson::Value& value,
                rapidjson::MemoryPoolAllocator<>& allocator);

/*********************************************************************
 * Json Object Set Function
 *********************************************************************/
extern void Set(rapidjson::Value* obj,
                rapidjson::Value& key,
                rapidjson::Value& value,
                rapidjson::MemoryPoolAllocator<>& allocator);

template <class T_POD, typename = typename std::enable_if<std::is_pod<T_POD>::value>::type>
static inline void Set(rapidjson::Value* obj,
                       const char* key,
                       T_POD value,
                       rapidjson::MemoryPoolAllocator<>& allocator) {
  rapidjson::Value obj_val(value);
  rapidjson::Value obj_key(key, strlen(key), allocator);
  return Set(obj, obj_key, obj_val, allocator);
}

extern void Set(rapidjson::Value* obj,
                const char* key,
                string_view value,
                rapidjson::MemoryPoolAllocator<>& allocator);

extern void Set(rapidjson::Value* obj,
                const char* key,
                const char* value,
                rapidjson::MemoryPoolAllocator<>& allocator);

extern void Set(rapidjson::Value* obj,
                const char* key,
                rapidjson::Value& value,
                rapidjson::MemoryPoolAllocator<>& allocator);

/*********************************************************************
 * Json Object Get Function
 *********************************************************************/
// just use JSON_GET and JSON_GET_OPTIONAL
#ifndef JSON_GET
#define JSON_GET(value, member, TYPE)                                      \
  [&]() {                                                                  \
    MXCHECK((value).HasMember((member))) << #value " must define " member; \
    MXCHECK((value)[(member)].Is##TYPE()) << member " must be a " #TYPE;   \
    return (value)[(member)].Get##TYPE();                                  \
  }()
#endif

#ifndef JSON_GET_OPTIONAL
#define JSON_GET_OPTIONAL(value, member, TYPE, def)                      \
  [&]() {                                                                \
    if (!(value).HasMember((member)))                                    \
      return def;                                                        \
    MXCHECK((value)[(member)].Is##TYPE()) << member " must be a " #TYPE; \
    return (value)[(member)].Get##TYPE();                                \
  }()
#endif

}  // namespace JsonUtil
}  // namespace runtime
}  // namespace matxscript
