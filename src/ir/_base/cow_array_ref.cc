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
#include <matxscript/ir/_base/structural_equal.h>
#include <matxscript/ir/_base/structural_hash.h>
#include <matxscript/ir/printer/doc.h>
#include <matxscript/ir/printer/ir_docsifier.h>
#include <matxscript/runtime/functor.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace ir {

using runtime::PyArgs;
using runtime::RTValue;

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
    if (equal.IsPathTracingEnabled()) {
      return SEqualReduceTraced(lhs, rhs, equal);
    }

    if (lhs->size() != rhs->size())
      return false;
    for (size_t i = 0; i < lhs->size(); ++i) {
      if (!equal(lhs->at(i), rhs->at(i)))
        return false;
    }
    return true;
  }

 private:
  static bool SEqualReduceTraced(const ArrayNode* lhs,
                                 const ArrayNode* rhs,
                                 const SEqualReducer& equal) {
    uint32_t min_size = std::min(lhs->size(), rhs->size());
    const ObjectPathPair& array_paths = equal.GetCurrentObjectPaths();

    for (uint32_t index = 0; index < min_size; ++index) {
      ObjectPathPair element_paths = {array_paths->lhs_path->ArrayIndex(index),
                                      array_paths->rhs_path->ArrayIndex(index)};
      if (!equal(lhs->at(index), rhs->at(index), element_paths)) {
        return false;
      }
    }

    if (lhs->size() == rhs->size()) {
      return true;
    }

    // If the array length is mismatched, don't report it immediately.
    // Instead, defer the failure until we visit all children.
    //
    // This is for human readability. For example, say we have two sequences
    //
    //    (1)     a b c d e f g h i j k l m
    //    (2)     a b c d e g h i j k l m
    //
    // If we directly report a mismatch at the end of the array right now,
    // the user will see that array (1) has an element `m` at index 12 but array (2)
    // has no index 12 because it's too short:
    //
    //    (1)     a b c d e f g h i j k l m
    //                                    ^error here
    //    (2)     a b c d e g h i j k l m
    //                                    ^ error here
    //
    // This is not very helpful. Instead, if we defer reporting this mismatch until all elements
    // are fully visited, we can be much more helpful with pointing out the location:
    //
    //    (1)     a b c d e f g h i j k l m
    //                      ^
    //                   error here
    //
    //    (2)     a b c d e g h i j k l m
    //                      ^
    //                  error here
    if (equal->IsFailDeferralEnabled()) {
      if (lhs->size() > min_size) {
        equal->DeferFail({array_paths->lhs_path->ArrayIndex(min_size),
                          array_paths->rhs_path->MissingArrayElement(min_size)});
      } else {
        equal->DeferFail({array_paths->lhs_path->MissingArrayElement(min_size),
                          array_paths->rhs_path->ArrayIndex(min_size)});
      }
      // Can return `true` pretending that everything is good since we have deferred the failure.
      return true;
    }
    return false;
  }
};

MATXSCRIPT_REGISTER_OBJECT_TYPE(ArrayNode);
MATXSCRIPT_REGISTER_REFLECTION_VTABLE(ArrayNode, ArrayNodeTrait)
    .set_creator([](const runtime::String&) -> ObjectPtr<Object> {
      return runtime::make_object<ArrayNode>();
    });

MATXSCRIPT_REGISTER_GLOBAL("runtime.Array").set_body([](PyArgs args) -> RTValue {
  std::vector<ObjectRef> data;
  for (int i = 0; i < args.size(); ++i) {
    if (args[i].type_code() == runtime::TypeIndex::kRuntimeNullptr) {
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
using namespace ::matxscript::ir::printer;
MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Array<ObjectRef>>(  //
        "",
        [](Array<ObjectRef> array, ObjectPath p, IRDocsifier d) -> Doc {
          int n = array.size();
          Array<ExprDoc> results;
          results.reserve(n);
          for (int i = 0; i < n; ++i) {
            results.push_back(d->AsDoc<ExprDoc>(array[i], p->ArrayIndex(i)));
          }
          return ListDoc(results);
        });

}  // namespace ir
}  // namespace matxscript
