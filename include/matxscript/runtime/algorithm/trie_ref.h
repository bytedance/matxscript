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

#include <initializer_list>
#include <map>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/container/unicode_view.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class TrieNode;
class List;

class Trie : public ObjectRef {
 public:
  using ContainerType = TrieNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 public:
  Trie();
  explicit Trie(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }
  explicit Trie(const std::map<string_view, int64_t>& dic);
  // allow copy and assign
  Trie(const Trie& other) noexcept = default;
  Trie(Trie&& other) noexcept = default;
  Trie& operator=(const Trie& other) noexcept = default;
  Trie& operator=(Trie&& other) noexcept = default;

  // sugar
  TrieNode* get();
  const TrieNode* get() const {
    return const_cast<Trie*>(this)->get();
  }
  const TrieNode* operator->() const {
    return get();
  }
  TrieNode* operator->() {
    return get();
  }

  // public method
  void update(const string_view& w, int64_t val = -1) const;
  void update(const unicode_view& w, int64_t val = -1) const;
  void update(const Any& w, int64_t val = -1) const;
  int64_t PrefixSearch(const string_view& w, int64_t* val = nullptr) const;
  int64_t PrefixSearch(const unicode_view& w, int64_t* val = nullptr) const;
  Tuple prefix_search(const string_view& w, int64_t pos = 0) const;
  Tuple prefix_search(const unicode_view& w, int64_t pos = 0) const;
  Tuple prefix_search(const Any& w, int64_t pos = 0) const;
  List prefix_search_all(const string_view& w, int64_t pos = 0) const;
  List prefix_search_all(const unicode_view& w, int64_t pos = 0) const;
  List prefix_search_all(const Any& w, int64_t pos = 0) const;
  int save(const unicode_view& file_path) const;
  int load(const unicode_view& file_path) const;
};

template <>
bool IsConvertible<Trie>(const Object* node);

template <>
MATXSCRIPT_ALWAYS_INLINE Trie Any::As<Trie>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeTrie);
  return Trie(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE Trie Any::AsNoCheck<Trie>() const {
  return Trie(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

std::ostream& operator<<(std::ostream& os, Trie const& n);

}  // namespace runtime
}  // namespace matxscript
