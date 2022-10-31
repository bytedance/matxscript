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
#include <matxscript/runtime/algorithm/trie_ref.h>

#include <matxscript/runtime/algorithm/trie_private.h>
#include <matxscript/runtime/container/list_ref.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/global_type_index.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

template <>
bool IsConvertible<Trie>(const Object* node) {
  return node ? node->IsInstance<Trie::ContainerType>() : Trie::_type_is_nullable;
}

Trie::Trie() {
  data_ = make_object<TrieNode>();
}

Trie::Trie(const std::map<string_view, int64_t>& dic) {
  data_ = make_object<TrieNode>(dic);
}

TrieNode* Trie::get() {
  MX_DPTR(Trie);
  return d;
}

void Trie::update(const string_view& w, int64_t val) const {
  MX_CHECK_DPTR(Trie);
  return d->Update(w, val);
}

void Trie::update(const unicode_view& w, int64_t val) const {
  MX_CHECK_DPTR(Trie);
  return d->Update(w, val);
}

void Trie::update(const Any& w, int64_t val) const {
  MX_CHECK_DPTR(Trie);
  return d->update(w, val);
}

int64_t Trie::PrefixSearch(const string_view& w, int64_t* val) const {
  MX_CHECK_DPTR(Trie);
  return d->PrefixSearch(w, val);
}

int64_t Trie::PrefixSearch(const unicode_view& w, int64_t* val) const {
  MX_CHECK_DPTR(Trie);
  return d->PrefixSearch(w, val);
}

Tuple Trie::prefix_search(const string_view& w, int64_t pos) const {
  MX_CHECK_DPTR(Trie);
  return d->prefix_search(w, pos);
}

Tuple Trie::prefix_search(const unicode_view& w, int64_t pos) const {
  MX_CHECK_DPTR(Trie);
  return d->prefix_search(w, pos);
}

Tuple Trie::prefix_search(const Any& w, int64_t pos) const {
  MX_CHECK_DPTR(Trie);
  return d->prefix_search(w, pos);
}

List Trie::prefix_search_all(const string_view& w, int64_t pos) const {
  MX_CHECK_DPTR(Trie);
  return d->prefix_search_all(w, pos);
}

List Trie::prefix_search_all(const unicode_view& w, int64_t pos) const {
  MX_CHECK_DPTR(Trie);
  return d->prefix_search_all(w, pos);
}

List Trie::prefix_search_all(const Any& w, int64_t pos) const {
  MX_CHECK_DPTR(Trie);
  return d->prefix_search_all(w, pos);
}

int Trie::save(const unicode_view& file_path) const {
  MX_CHECK_DPTR(Trie);
  return d->save(file_path);
}

int Trie::load(const unicode_view& file_path) const {
  MX_CHECK_DPTR(Trie);
  return d->load(file_path);
}

std::ostream& operator<<(std::ostream& os, Trie const& n) {
  os << "Trie(addr: " << n.get() << ")";
  return os;
}

}  // namespace runtime
}  // namespace matxscript
