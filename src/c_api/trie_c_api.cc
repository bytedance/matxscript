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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Trie container
 *****************************************************************************/
MATXSCRIPT_REGISTER_GLOBAL("runtime.Trie").set_body([](PyArgs args) -> RTValue {
  MXCHECK_LE(args.size(), 1) << "[runtime.Trie] Expect 0 or 1 arguments but get " << args.size();
  if (args.size() > 0) {
    MXCHECK(args[0].IsObjectRef<Dict>())
        << "[runtime.Trie] Expect arguments[0] is Dict type but get: type_code="
        << args[0].type_code() << " name=" << TypeIndex2Str(args[0].type_code());
    Dict d = args[0].As<Dict>();
    std::map<string_view, int64_t> dic;
    std::vector<String> ukeys;
    ukeys.reserve(d.size());
    for (auto& kv : d.items()) {
      MXCHECK(kv.first.IsString() || kv.first.IsUnicode())
          << "[runtime.Trie] Expect arguments[0] is dict<str, int>, but get key mismatch: "
          << kv.first.type_name();
      MXCHECK(kv.second.type_code() == TypeIndex::kRuntimeInteger)
          << "[runtime.Trie] Expect arguments[0] is dict<str, int>, but get value mismatch: "
          << kv.second.type_name();
      int64_t index = kv.second.As<int64_t>();
      if (kv.first.type_code() == TypeIndex::kRuntimeString) {
        auto node = kv.first.As<string_view>();
        dic.emplace(node, index);
      } else {
        auto view = kv.first.As<unicode_view>();
        ukeys.push_back(UTF8Encode(view));
        dic.emplace(ukeys.back(), index);
      }
    }
    return Trie(dic);
  }
  return Trie();
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Trie_Update").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() == 2 || args.size() == 3)
      << "[runtime.Trie_Update] Expect 2 or 3 arguments but get " << args.size();
  MXCHECK(args[0].IsObjectRef<Trie>())
      << "[runtime.Trie_Update] Expect arguments[0] is Trie, but get: "
      << TypeIndex2Str(args[0].type_code());
  MXCHECK(args[1].IsString() || args[1].IsUnicode())
      << "[runtime.Trie_Update] Expect arguments[1] is str, but get: "
      << TypeIndex2Str(args[1].type_code());
  auto* trie_node = args[0].ptr<TrieNode>();
  int64_t index = -1;
  if (args.size() == 3) {
    MXCHECK(args[2].type_code() == TypeIndex::kRuntimeInteger)
        << "[runtime.Trie_Update] Expect arguments[2] is int, but get: "
        << TypeIndex2Str(args[2].type_code());
    index = args[2].As<int64_t>();
  }
  if (args[1].IsString()) {
    trie_node->Update(args[1].As<string_view>(), index);
  } else {
    trie_node->Update(args[1].As<unicode_view>(), index);
  }
  return None;
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Trie_PrefixSearch").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[runtime.Trie_PrefixSearch] Expect 3 arguments but get "
                             << args.size();
  MXCHECK(args[0].IsObjectRef<Trie>())
      << "[runtime.Trie_PrefixSearch] Expect arguments[0] is Trie, but get: "
      << TypeIndex2Str(args[0].type_code());
  MXCHECK(args[1].IsString() || args[1].IsUnicode())
      << "[runtime.Trie_PrefixSearch] Expect arguments[1] is str, but get: "
      << TypeIndex2Str(args[1].type_code());
  MXCHECK(args[2].type_code() == TypeIndex::kRuntimeInteger)
      << "[runtime.Trie_PrefixSearch] Expect arguments[2] is int, but get: "
      << TypeIndex2Str(args[2].type_code());
  auto* trie_node = args[0].ptr<TrieNode>();
  int64_t pos = args[2].As<int64_t>();
  RTValue w = args[1].As<RTValue>();
  return trie_node->prefix_search(w, pos);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Trie_PrefixSearchAll").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 3) << "[runtime.Trie_PrefixSearchAll] Expect 3 arguments but get "
                             << args.size();
  MXCHECK(args[0].IsObjectRef<Trie>())
      << "[runtime.Trie_PrefixSearchAll] Expect arguments[0] is Trie, but get: "
      << TypeIndex2Str(args[0].type_code());
  auto* trie_node = args[0].ptr<TrieNode>();
  int64_t pos = args[2].As<int64_t>();
  RTValue w = args[1].As<RTValue>();
  return trie_node->prefix_search_all(w, pos);
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Trie_Save").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[runtime.Trie_Save] Expect 2 arguments but get " << args.size();
  MXCHECK(args[0].IsObjectRef<Trie>())
      << "[runtime.Trie_Save] Expect arguments[0] is Trie, but get: "
      << TypeIndex2Str(args[0].type_code());
  auto* trie_node = args[0].ptr<TrieNode>();
  return trie_node->save(args[1].As<Unicode>());
});

MATXSCRIPT_REGISTER_GLOBAL("runtime.Trie_Load").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 2) << "[runtime.Trie_Load] Expect 2 arguments but get " << args.size();
  MXCHECK(args[0].IsObjectRef<Trie>())
      << "[runtime.Trie_Load] Expect arguments[0] is Trie, but get: "
      << TypeIndex2Str(args[0].type_code());
  auto* trie_node = args[0].ptr<TrieNode>();
  return trie_node->load(args[1].As<Unicode>());
});

}  // namespace runtime
}  // namespace matxscript
