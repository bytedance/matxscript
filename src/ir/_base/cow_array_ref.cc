// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 * https://github.com/apache/tvm/blob/v0.7/include/tvm/runtime/container.h
 * with changes applied:
 * - rename namespace
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
#include <matxscript/ir/_base/cow_array_ref.h>
#include <matxscript/ir/_base/string_ref.h>

#include <matxscript/ir/_base/reflection.h>
#include <matxscript/ir/_base/repr_printer.h>
#include <matxscript/ir/_base/structural_equal.h>
#include <matxscript/ir/_base/structural_hash.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Array container
 *****************************************************************************/

struct ArrayNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduce(const ArrayNode* key, SHashReducer hash_reduce) {
    hash_reduce(static_cast<uint64_t>(key->size()));
    for (size_t i = 0; i < key->size(); ++i) {
      hash_reduce(key->at(i));
    }
  }

  static bool SEqualReduce(const ArrayNode* lhs, const ArrayNode* rhs, SEqualReducer equal) {
    if (lhs->size() != rhs->size())
      return false;
    for (size_t i = 0; i < lhs->size(); ++i) {
      if (!equal(lhs->at(i), rhs->at(i)))
        return false;
    }
    return true;
  }
};

MATXSCRIPT_REGISTER_OBJECT_TYPE(ArrayNode);
MATXSCRIPT_REGISTER_REFLECTION_VTABLE(ArrayNode, ArrayNodeTrait)
    .set_creator([](const String&) -> ObjectPtr<Object> { return make_object<ArrayNode>(); });

MATXSCRIPT_REGISTER_GLOBAL("runtime.Array").set_body([](PyArgs args) -> RTValue {
  std::vector<ObjectRef> data;
  for (int i = 0; i < args.size(); ++i) {
    if (args[i].type_code() == TypeIndex::kRuntimeNullptr) {
      data.push_back(ObjectRef(nullptr));
    } else if (args[i].type_code() >= 0) {
      data.push_back(args[i].As<ObjectRef>());
    } else {
      MXCHECK(StringRef::CanConvertFrom(args[i]))
          << "[runtime.Array] not supported item type_code: " << args[i].type_code();
      data.push_back(args[i].As<StringRef>());
    }
  }
  return Array<ObjectRef>(data);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ArrayGetItem").set_body([](PyArgs args) -> RTValue {
  int64_t i = args[1].As<int64_t>();
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  MXCHECK(ptr->IsInstance<ArrayNode>());
  auto* n = static_cast<const ArrayNode*>(ptr);
  MXCHECK_LT(static_cast<size_t>(i), n->size()) << "out of bound of array";
  return n->at(i);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ArraySize").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  MXCHECK(ptr->IsInstance<ArrayNode>());
  return static_cast<int64_t>(static_cast<const ArrayNode*>(ptr)->size());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.ArrayContains").set_body([](PyArgs args) -> RTValue {
  ObjectRef item =
      StringRef::CanConvertFrom(args[1]) ? args[1].As<StringRef>() : args[1].As<ObjectRef>();
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  MXCHECK(ptr->IsInstance<ArrayNode>());
  auto* n = static_cast<const ArrayNode*>(ptr);
  bool result = false;
  for (auto i = 0; i < n->size(); ++i) {
    if (ObjectEqual()(item, n->at(i))) {
      result = true;
      break;
    }
  }
  return result;
});

// Container printer
MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ArrayNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ArrayNode*>(node.get());
      p->stream << '[';
      for (size_t i = 0; i < op->size(); ++i) {
        if (i != 0) {
          p->stream << ", ";
        }
        p->Print(op->at(i));
      }
      p->stream << ']';
    });

}  // namespace runtime
}  // namespace matxscript
