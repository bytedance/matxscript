// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
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
#include <matxscript/ir/_base/object_hash.h>
#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

size_t ObjectHash::operator()(const ObjectRef& a) const {
  const Object* obj_ptr = a.get();
  switch (a->type_index()) {
    case runtime::TypeIndex::kRuntimeStringRef: {
      auto node = reinterpret_cast<const StringNode*>(obj_ptr);
      return runtime::BytesHash(node->data_container.data(), node->data_container.size());
    } break;
    default: {
      return runtime::ObjectPtrHash()(a);
    } break;
  }
}

MATXSCRIPT_REGISTER_GLOBAL("runtime.ObjectRefHash").set_body_typed([](ObjectRef obj) {
  return static_cast<int64_t>(ObjectHash()(obj));
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ObjectPtrHash").set_body_typed([](ObjectRef obj) {
  return static_cast<int64_t>(runtime::ObjectPtrHash()(obj));
});

}  // namespace ir
}  // namespace matxscript
