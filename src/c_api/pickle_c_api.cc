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
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <matxscript/pipeline/pickle.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/json_util.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {
namespace pickle {

static Unicode ToJsonStructStr(const Any& rtv) {
  rapidjson::Document doc;
  rapidjson::StringBuffer buffer;
  ToJsonStruct(rtv, doc, doc.GetAllocator());
  buffer.Clear();
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  return String(buffer.GetString()).decode();
}

static RTValue FromJsonStructStr(const string_view& bytes) {
  rapidjson::Document doc;
  MXCHECK(JsonUtil::FromString(bytes, doc));
  return FromJsonStruct(doc);
}

MATXSCRIPT_REGISTER_GLOBAL("runtime.pickle_ToJsonStruct").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[runtime.pickle.ToJsonStruct] Expect 1 arguments but get "
                            << args.size();
  return ToJsonStructStr(args[0]);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.pickle_FromJsonStruct").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[runtime.pickle.FromJsonStruct] Expect 1 arguments but get "
                            << args.size();
  String v = UnicodeHelper::Encode(args[0].As<unicode_view>());
  return FromJsonStructStr(v);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Serialize").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[runtime.pickle.Serialize] Expect 1 arguments but get "
                            << args.size();
  return Serialize(args[0]).decode();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.DeSerialize").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 1) << "[runtime.pickle.DeSerialize] Expect 1 arguments but get "
                            << args.size();
  auto v = UnicodeHelper::Encode(args[0].As<unicode_view>());
  return DeSerialize(v);
});

}  // namespace pickle
}  // namespace runtime
}  // namespace matxscript
