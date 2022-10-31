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
#include <matxscript/ir/adt.h>
#include <matxscript/ir/type.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using namespace ::matxscript::runtime;

MATXSCRIPT_REGISTER_GLOBAL("ir.Type_GetRuntimeTypeCode").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[runtime.RTValue_GetTypeCode] Expect 1 arguments but get "
                             << args.size();
  int64_t type_code = INT16_MIN;
  Type t = args[0].As<Type>();
  if (IsPrimType(t)) {
    auto node = t.as<PrimTypeNode>();
    switch (node->dtype.code()) {
      case kDLUInt:
      case kDLInt: {
        type_code = TypeIndex::kRuntimeInteger;
      } break;
      case kDLBfloat:
      case kDLFloat: {
        type_code = TypeIndex::kRuntimeFloat;
      } break;

      default: {
        type_code = TypeIndex::kRuntimeOpaqueHandle;
      } break;
    }
  } else if (IsStringType(t)) {
    type_code = TypeIndex::kRuntimeString;
  } else if (IsUnicodeType(t)) {
    type_code = TypeIndex::kRuntimeUnicode;
  } else if (IsListType(t)) {
    type_code = TypeIndex::kRuntimeList;
  } else if (IsDictType(t)) {
    type_code = TypeIndex::kRuntimeDict;
  } else if (IsSetType(t)) {
    type_code = TypeIndex::kRuntimeSet;
  } else if (IsTrieType(t)) {
    type_code = TypeIndex::kRuntimeTrie;
  } else if (IsFileType(t)) {
    type_code = TypeIndex::kRuntimeFile;
  } else if (IsIteratorType(t)) {
    type_code = TypeIndex::kRuntimeIterator;
  } else if (IsUserDataType(t) || t->IsInstance<ClassTypeNode>()) {
    type_code = TypeIndex::kRuntimeUserData;
  }
  return type_code;
});

}  // namespace ir
}  // namespace matxscript
