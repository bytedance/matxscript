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

// clang-format off
#include <rapidjson/document.h>
#include <matxscript/runtime/logging.h>
#ifndef RAPIDJSON_ASSERT
#define RAPIDJSON_ASSERT(x) CHECK(x)
#endif
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/container.h>
// clang-format on

namespace matxscript {
namespace runtime {
namespace pickle {

/******************************************************************************
 * utf-8 json, no node info
 *****************************************************************************/
MATX_DLL RTValue FromJson(const rapidjson::Value& val, bool use_unicode = false);

MATX_DLL void ToJson(const Any& value,
                     rapidjson::Value& json_val,
                     rapidjson::MemoryPoolAllocator<>& allocator);

MATX_DLL inline rapidjson::Document ToJson(const Any& value) {
  rapidjson::Document doc;
  ToJson(value, doc, doc.GetAllocator());
  return doc;
}

/******************************************************************************
 * every node is like this:
 * {
 *    "t": "str",
 *    "v": "abc",
 * }
 *****************************************************************************/
// user_code is only valid for opaque handle
MATX_DLL RTValue FromJsonStruct(const rapidjson::Value& val);
MATX_DLL void ToJsonStruct(const Any& value,
                           rapidjson::Value& json_val,
                           rapidjson::MemoryPoolAllocator<>& allocator);

MATX_DLL inline rapidjson::Document ToJsonStruct(const Any& value) {
  rapidjson::Document doc;
  ToJsonStruct(value, doc, doc.GetAllocator());
  return doc;
}

MATX_DLL String Serialize(const Any& value);
MATX_DLL RTValue DeSerialize(const string_view& str);

}  // namespace pickle
}  // namespace runtime
}  // namespace matxscript
