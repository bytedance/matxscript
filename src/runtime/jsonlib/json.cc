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
#include <matxscript/runtime/jsonlib/json.h>

#include <rapidjson/document.h>

#include <matxscript/pipeline/pickle.h>
#include <matxscript/runtime/json_util.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

RTValue json_loads(string_view s) {
  rapidjson::Document doc;
  JsonUtil::FromString(s, doc);
  return pickle::FromJson(doc, true);
}

RTValue json_load(const File& fp) {
  rapidjson::Document doc;
  // TODO(wuxian): refactor JsonUtil or abandon it
  JsonUtil::FromFile(fp.path().data(), doc);
  return pickle::FromJson(doc, true);
}

Unicode json_dumps(const Any& obj, int indent, bool ensure_ascii) {
  rapidjson::Document doc;
  pickle::ToJson(obj, doc, doc.GetAllocator());
  if (indent == -1) {
    return JsonUtil::ToString(&doc, false, ensure_ascii).decode();
  } else {
    return JsonUtil::ToString(&doc, true, ensure_ascii, indent).decode();
  }
}

// RTValue json_dump(const RTValue& obj, const File& fp) {
// TODO(wuxian): Need a file class that supports write operations
// }

MATXSCRIPT_REGISTER_GLOBAL("runtime.JsonDumps").set_body_typed(json_dumps);

}  // namespace runtime
}  // namespace matxscript
