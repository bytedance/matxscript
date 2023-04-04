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
    using KV = std::pair<uint64_t, ObjectRef>;
    std::vector<KV> temp;
    for (const auto& kv : *key) {
      uint64_t hashed_value;
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
      // ties are rare, but we need to skip them to make the hash deterministic
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
      temp.push_back(std::make_pair(runtime::Downcast<StringRef>(kv.first), kv.second));
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

  static bool IsStringMap(const MapNode* map) {
    return std::all_of(map->begin(), map->end(), [](const auto& v) {
      return v.first->template IsInstance<StringNode>();
    });
  }

  static bool SEqualReduceTracedForOMap(const MapNode* lhs,
                                        const MapNode* rhs,
                                        const SEqualReducer& equal) {
    const ObjectPathPair& map_paths = equal.GetCurrentObjectPaths();

    std::vector<const Object*> seen_rhs_keys;

    // First, check that every key from `lhs` is also in `rhs`,
    // and their values are mapped to each other.
    for (const auto& kv : *lhs) {
      ObjectPath lhs_path = map_paths->lhs_path->MapValue(kv.first);

      ObjectRef rhs_key = equal->MapLhsToRhs(kv.first);
      if (!rhs_key.defined()) {
        equal.RecordMismatchPaths({lhs_path, map_paths->rhs_path->MissingMapEntry()});
        return false;
      }

      auto it = rhs->find(rhs_key);
      if (it == rhs->end()) {
        equal.RecordMismatchPaths({lhs_path, map_paths->rhs_path->MissingMapEntry()});
        return false;
      }

      if (!equal(kv.second, it->second, {lhs_path, map_paths->rhs_path->MapValue(it->first)})) {
        return false;
      }

      seen_rhs_keys.push_back(it->first.get());
    }

    std::sort(seen_rhs_keys.begin(), seen_rhs_keys.end());

    // Second, check that we have visited every `rhs` key when iterating over `lhs`.
    for (const auto& kv : *rhs) {
      if (!std::binary_search(seen_rhs_keys.begin(), seen_rhs_keys.end(), kv.first.get())) {
        equal.RecordMismatchPaths(
            {map_paths->lhs_path->MissingMapEntry(), map_paths->rhs_path->MapValue(kv.first)});
        return false;
      }
    }

    MXCHECK(lhs->size() == rhs->size());
    return true;
  }

  static bool SEqualReduceTracedForSMap(const MapNode* lhs,
                                        const MapNode* rhs,
                                        const SEqualReducer& equal) {
    const ObjectPathPair& map_paths = equal.GetCurrentObjectPaths();

    // First, check that every key from `lhs` is also in `rhs`, and their values are equal.
    for (const auto& kv : *lhs) {
      ObjectPath lhs_path = map_paths->lhs_path->MapValue(kv.first);
      auto it = rhs->find(kv.first);
      if (it == rhs->end()) {
        equal.RecordMismatchPaths({lhs_path, map_paths->rhs_path->MissingMapEntry()});
        return false;
      }

      if (!equal(kv.second, it->second, {lhs_path, map_paths->rhs_path->MapValue(it->first)})) {
        return false;
      }
    }

    // Second, make sure every key from `rhs` is also in `lhs`.
    for (const auto& kv : *rhs) {
      ObjectPath rhs_path = map_paths->rhs_path->MapValue(kv.first);
      if (!lhs->count(kv.first)) {
        equal.RecordMismatchPaths({map_paths->lhs_path->MissingMapEntry(), rhs_path});
        return false;
      }
    }

    MXCHECK(lhs->size() == rhs->size());
    return true;
  }

  static bool SEqualReduceTraced(const MapNode* lhs,
                                 const MapNode* rhs,
                                 const SEqualReducer& equal) {
    if (IsStringMap(lhs)) {
      return SEqualReduceTracedForSMap(lhs, rhs, equal);
    } else {
      return SEqualReduceTracedForOMap(lhs, rhs, equal);
    }
  }

  static bool SEqualReduce(const MapNode* lhs, const MapNode* rhs, SEqualReducer equal) {
    if (equal.IsPathTracingEnabled()) {
      return SEqualReduceTraced(lhs, rhs, equal);
    }

    if (rhs->size() != lhs->size())
      return false;
    if (rhs->size() == 0)
      return true;
    bool ls = IsStringMap(lhs);
    bool rs = IsStringMap(rhs);
    if (ls != rs) {
      return false;
    }
    return (ls && rs) ? SEqualReduceForSMap(lhs, rhs, equal) : SEqualReduceForOMap(lhs, rhs, equal);
  }
};

MATXSCRIPT_REGISTER_OBJECT_TYPE(MapNode);
MATXSCRIPT_REGISTER_REFLECTION_VTABLE(MapNode, MapNodeTrait)
    .set_creator([](const runtime::String&) -> ObjectPtr<Object> { return MapNode::Empty(); });

MATXSCRIPT_REGISTER_GLOBAL("runtime.Map").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size() % 2, 0);
  std::unordered_map<ObjectRef, ObjectRef, runtime::ObjectPtrHash, runtime::ObjectPtrEqual> data;
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
      rkvs.push_back(runtime::Downcast<StringRef>(kv.first));
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
      keys.push_back(runtime::Downcast<StringRef>(kv.first));
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

MATX_DLL constexpr uint64_t DenseMapNode::kNextProbeLocation[];

using namespace ::matxscript::ir::printer;
MATXSCRIPT_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<Map<ObjectRef, ObjectRef>>(  //
        "",
        [](Map<ObjectRef, ObjectRef> dict, ObjectPath p, IRDocsifier d) -> Doc {
          using POO = std::pair<ObjectRef, ObjectRef>;
          std::vector<POO> items{dict.begin(), dict.end()};
          bool is_str_map = true;
          for (const auto& kv : items) {
            if (!kv.first.as<StringNode>()) {
              is_str_map = false;
              break;
            }
          }
          if (is_str_map) {
            std::sort(items.begin(), items.end(), [](const POO& lhs, const POO& rhs) {
              return runtime::Downcast<StringRef>(lhs.first) <
                     runtime::Downcast<StringRef>(rhs.first);
            });
          } else {
            std::sort(items.begin(), items.end(), [](const POO& lhs, const POO& rhs) {
              return lhs.first.get() < rhs.first.get();
            });
          }
          int n = dict.size();
          Array<ExprDoc> ks;
          Array<ExprDoc> vs;
          ks.reserve(n);
          vs.reserve(n);
          for (int i = 0; i < n; ++i) {
            ks.push_back(d->AsDoc<ExprDoc>(items[i].first, p->MissingMapEntry()));
            vs.push_back(d->AsDoc<ExprDoc>(items[i].second, p->MapValue(items[i].first)));
          }
          return DictDoc(ks, vs);
        });

}  // namespace ir
}  // namespace matxscript
