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
#include <matxscript/ir/_base/cow_map_ref.h>
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
 * Map container
 *****************************************************************************/

struct MapNodeTrait {
  static constexpr const std::nullptr_t VisitAttrs = nullptr;

  static void SHashReduceForOMap(const MapNode* key, SHashReducer hash_reduce) {
    // SHash's var handling depends on the determinism of traversal.
    // NOTE: only book-keep the mapped hash keys.
    // This resolves common use cases where we want to store
    // Map<Var, Value> where Var is defined in the function
    // parameters.
    using KV = std::pair<size_t, ObjectRef>;
    std::vector<KV> temp;
    for (const auto& kv : *key) {
      size_t hashed_value;
      if (hash_reduce->LookupHashedValue(kv.first, &hashed_value)) {
        temp.emplace_back(hashed_value, kv.second);
      }
    }
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(), [](const KV& lhs, const KV& rhs) {
      return lhs.first < rhs.first;
    });
    // add size to the hash
    hash_reduce(static_cast<uint64_t>(key->size()));
    // hash the content
    for (size_t i = 0; i < temp.size();) {
      size_t k = i + 1;
      for (; k < temp.size() && temp[k].first == temp[i].first; ++k) {
      }
      // ties are rare, but we need to skip them to make the hash determinsitic
      if (k == i + 1) {
        hash_reduce->SHashReduceHashedValue(temp[i].first);
        hash_reduce(temp[i].second);
      }
      i = k;
    }
  }

  static void SHashReduceForSMap(const MapNode* key, SHashReducer hash_reduce) {
    // NOTE: only book-keep the mapped hash keys.
    // This resolves common use cases where we want to store
    // Map<Var, Value> where Var is defined in the function
    // parameters.
    using KV = std::pair<StringRef, ObjectRef>;
    std::vector<KV> temp;
    for (const auto& kv : *key) {
      temp.push_back(std::make_pair(Downcast<StringRef>(kv.first), kv.second));
    }
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(), [](const KV& lhs, const KV& rhs) {
      return lhs.first < rhs.first;
    });
    // NOTE: we won't have ties
    // add size to the hash after sorting.
    hash_reduce(static_cast<uint64_t>(key->size()));
    // hash the content
    for (size_t i = 0; i < temp.size(); ++i) {
      hash_reduce(temp[i].first);
      hash_reduce(temp[i].second);
    }
  }

  static void SHashReduce(const MapNode* key, SHashReducer hash_reduce) {
    bool is_str_map = std::all_of(key->begin(), key->end(), [](const auto& v) {
      return v.first->template IsInstance<StringNode>();
    });
    if (is_str_map) {
      SHashReduceForSMap(key, hash_reduce);
    } else {
      SHashReduceForOMap(key, hash_reduce);
    }
  }

  static bool SEqualReduceForOMap(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    for (const auto& kv : *lhs) {
      // Only allow equal checking if the keys are already mapped
      // This resolves common use cases where we want to store
      // Map<Var, Value> where Var is defined in the function
      // parameters.
      ObjectRef rhs_key = equal->MapLhsToRhs(kv.first);
      if (!rhs_key.defined())
        return false;
      auto it = rhs->find(rhs_key);
      if (it == rhs->end())
        return false;
      if (!equal(kv.second, it->second))
        return false;
    }
    return true;
  }

  static bool SEqualReduceForSMap(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    for (const auto& kv : *lhs) {
      auto it = rhs->find(kv.first);
      if (it == rhs->end())
        return false;
      if (!equal(kv.second, it->second))
        return false;
    }
    return true;
  }

  static bool SEqualReduce(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    if (rhs->size() != lhs->size())
      return false;
    if (rhs->size() == 0)
      return true;
    bool ls = std::all_of(lhs->begin(), lhs->end(), [](const auto& v) {
      return v.first->template IsInstance<StringNode>();
    });
    bool rs = std::all_of(rhs->begin(), rhs->end(), [](const auto& v) {
      return v.first->template IsInstance<StringNode>();
    });
    if (ls != rs) {
      return false;
    }
    return (ls && rs) ? SEqualReduceForSMap(lhs, rhs, equal) : SEqualReduceForOMap(lhs, rhs, equal);
  }
};

MATXSCRIPT_REGISTER_OBJECT_TYPE(MapNode);
MATXSCRIPT_REGISTER_REFLECTION_VTABLE(MapNode, MapNodeTrait)
    .set_creator([](const String&) -> ObjectPtr<Object> { return MapNode::Empty(); });

MATXSCRIPT_REGISTER_GLOBAL("runtime.Map").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size() % 2, 0);
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> data;
  for (int i = 0; i < args.size(); i += 2) {
    ObjectRef k =
        StringRef::CanConvertFrom(args[i]) ? args[i].As<StringRef>() : args[i].As<ObjectRef>();
    ObjectRef v = args[i + 1].As<ObjectRef>();
    data.emplace(std::move(k), std::move(v));
  }
  return Map<ObjectRef, ObjectRef>(std::move(data));
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.MapSize").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  MXCHECK(ptr->IsInstance<MapNode>());
  auto* n = static_cast<const MapNode*>(ptr);
  return static_cast<int64_t>(n->size());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.MapGetItem").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  MXCHECK(ptr->IsInstance<MapNode>());

  auto* n = static_cast<const MapNode*>(ptr);
  auto it = n->find(StringRef::CanConvertFrom(args[1]) ? args[1].As<StringRef>()
                                                       : args[1].As<ObjectRef>());
  MXCHECK(it != n->end()) << "cannot find the corresponding key in the Map";
  return (*it).second;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.MapCount").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  MXCHECK(ptr->IsInstance<MapNode>());
  const MapNode* n = static_cast<const MapNode*>(ptr);
  int64_t cnt = n->count(StringRef::CanConvertFrom(args[1]) ? args[1].As<StringRef>()
                                                            : args[1].As<ObjectRef>());
  return cnt;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.MapItems").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  auto* n = static_cast<const MapNode*>(ptr);
  Array<ObjectRef> rkvs;
  for (const auto& kv : *n) {
    if (kv.first->IsInstance<StringNode>()) {
      rkvs.push_back(Downcast<StringRef>(kv.first));
    } else {
      rkvs.push_back(kv.first);
    }
    rkvs.push_back(kv.second);
  }
  return std::move(rkvs);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.MapKeys").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  auto* n = static_cast<const MapNode*>(ptr);
  Array<ObjectRef> keys;
  for (const auto& kv : *n) {
    if (kv.first->IsInstance<StringNode>()) {
      keys.push_back(Downcast<StringRef>(kv.first));
    } else {
      keys.push_back(kv.first);
    }
  }
  return keys;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.MapValues").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args[0].type_code(), 0);
  Object* ptr = static_cast<Object*>(args[0].value().data.v_handle);
  auto* n = static_cast<const MapNode*>(ptr);
  Array<ObjectRef> values;
  for (const auto& kv : *n) {
    values.push_back(kv.second);
  }
  return std::move(values);
});

MATXSCRIPT_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<MapNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const MapNode*>(node.get());
      p->stream << '{';
      for (auto it = op->begin(); it != op->end(); ++it) {
        if (it != op->begin()) {
          p->stream << ", ";
        }
        if (it->first->IsInstance<StringNode>()) {
          p->stream << '\"' << Downcast<StringRef>(it->first) << "\": ";
        } else {
          p->Print(it->first);
          p->stream << ": ";
        }
        p->Print(it->second);
      }
      p->stream << '}';
    });

MATX_DLL constexpr uint64_t DenseMapNode::kNextProbeLocation[];

}  // namespace runtime
}  // namespace matxscript
