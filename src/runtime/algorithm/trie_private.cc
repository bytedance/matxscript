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
#include <matxscript/runtime/algorithm/trie_private.h>

#include <unordered_map>

#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

// constructor
TrieNode::TrieNode() {
  trie_ = std::make_unique<cedar_t>();
}

TrieNode::TrieNode(const std::map<string_view, int64_t>& dic) {
  trie_ = std::make_unique<cedar_t>();
  if (dic.empty())
    return;
  std::vector<const char*> key;
  std::vector<size_t> key_len;
  std::vector<int> values;
  key.reserve(dic.size());
  key_len.reserve(dic.size());
  values.reserve(dic.size());
  for (const auto& it : dic) {
    key.push_back(it.first.data());
    key_len.push_back(it.first.size());
    values.push_back(it.second);
  }
  auto rc =
      trie_->build(key.size(), const_cast<const char**>(&key[0]), key_len.data(), values.data());
  MXCHECK_EQ(rc, 0) << "build trie failed!!!";
}

void TrieNode::Update(const string_view& w, int64_t val) {
  trie_->update(w.data(), w.size(), (int32_t)val);
}

void TrieNode::Update(const unicode_view& w, int64_t val) {
  auto u8s = UTF8Encode(w.data(), w.size());
  return Update(u8s.view(), val);
}

int64_t TrieNode::PrefixSearch(const string_view& w, int64_t* val) const {
  if (trie_ == nullptr) {
    return 0;
  }
  constexpr int kResultSize = 64;
  cedar_t::result_pair_type trie_results[kResultSize];
  const int num_nodes = trie_->commonPrefixSearch(w.data(), trie_results, kResultSize, w.size());

  if (num_nodes == 0) {
    return 0;
  }
  int64_t mblen = 0;
  for (int i = 0; i < num_nodes; ++i) {
    if (mblen < (int64_t)trie_results[i].length) {
      mblen = trie_results[i].length;
      if (val) {
        *val = trie_results[i].value;
      }
    }
  }
  return mblen;
}

int64_t TrieNode::PrefixSearch(const unicode_view& w, int64_t* val) const {
  auto u8s = UTF8Encode(w.data(), w.size());
  int64_t bytes_len = PrefixSearch(u8s, val);
  return UTF8CharCounts(u8s.data(), bytes_len);
}

std::vector<std::pair<int64_t, int64_t>> TrieNode::PrefixSearchAll(const string_view& w) const {
  std::vector<std::pair<int64_t, int64_t>> ret;
  if (trie_ == nullptr) {
    return ret;
  }
  constexpr int kResultSize = 64;
  cedar_t::result_pair_type trie_results[kResultSize];
  const int num_nodes = trie_->commonPrefixSearch(w.data(), trie_results, kResultSize, w.size());

  if (num_nodes == 0) {
    return ret;
  }
  for (int i = 0; i < num_nodes; ++i) {
    ret.push_back({trie_results[i].length, trie_results[i].value});
  }
  return ret;
}

std::vector<std::pair<int64_t, int64_t>> TrieNode::PrefixSearchAll(const unicode_view& w) const {
  auto u8s = UTF8Encode(w.data(), w.size());
  auto ret = PrefixSearchAll(u8s);

  // Traverse the utf8 string and generate a mapping from utf8 len to unicode count
  std::unordered_map<int64_t, int64_t> len2count;
  const char* start = u8s.data();
  const char* end = u8s.data() + u8s.size();
  int64_t count = 0, length = 0;
  while (start < end) {
    int char_length = OneCharLen(start);
    start += char_length;
    length += char_length;
    ++count;
    len2count[length] = count;
  }

  // Replace utf8 length with unicode length
  for (auto it = ret.begin(); it != ret.end();) {
    auto count_it = len2count.find(it->first);
    if (count_it == len2count.end()) {
      it = ret.erase(it);
    } else {
      it->first = count_it->second;
      ++it;
    }
  }
  return ret;

  //
}

void TrieNode::update(const string_view& w, int64_t val) {
  Update(w, val);
}

void TrieNode::update(const unicode_view& w, int64_t val) {
  Update(w, val);
}

void TrieNode::update(const Any& w, int64_t val) {
  switch (w.type_code()) {
    case TypeIndex::kRuntimeString: {
      update(w.AsNoCheck<string_view>(), val);
    } break;
    case TypeIndex::kRuntimeUnicode: {
      update(w.AsNoCheck<unicode_view>(), val);
    } break;
    default: {
      THROW_PY_TypeError("Trie.update first arg must be str or bytes, not ", w.type_name());
    } break;
  }
}

Tuple TrieNode::prefix_search(const string_view& w, int64_t pos) const {
  int64_t val = -1;
  pos = slice_index_correction(pos, w.size());
  int64_t mblen = PrefixSearch(w.substr(pos), &val);
  return Tuple::dynamic(mblen, val);
}

Tuple TrieNode::prefix_search(const unicode_view& w, int64_t pos) const {
  int64_t val = -1;
  pos = slice_index_correction(pos, w.size());
  int64_t mblen = PrefixSearch(w.substr(pos), &val);
  return Tuple::dynamic(mblen, val);
}

Tuple TrieNode::prefix_search(const Any& w, int64_t pos) const {
  if (w.type_code() == TypeIndex::kRuntimeUnicode) {
    return prefix_search(w.AsNoCheck<unicode_view>(), pos);
  } else if (w.type_code() == TypeIndex::kRuntimeString) {
    return prefix_search(w.AsNoCheck<string_view>(), pos);
  } else {
    return Tuple::dynamic(0, -1);
  }
}

List TrieNode::prefix_search_all(const string_view& w, int64_t pos) const {
  pos = slice_index_correction(pos, w.size());
  auto res = PrefixSearchAll(w.substr(pos));
  List ret;
  for (auto& item : res) {
    ret.push_back(Tuple::dynamic(item.first, item.second));
  }
  return ret;
}

List TrieNode::prefix_search_all(const unicode_view& w, int64_t pos) const {
  pos = slice_index_correction(pos, w.size());
  auto res = PrefixSearchAll(w.substr(pos));
  List ret;
  for (auto& item : res) {
    ret.push_back(Tuple::dynamic(item.first, item.second));
  }
  return ret;
}

List TrieNode::prefix_search_all(const Any& w, int64_t pos) const {
  if (w.type_code() == TypeIndex::kRuntimeUnicode) {
    return prefix_search_all(w.AsNoCheck<unicode_view>(), pos);
  } else if (w.type_code() == TypeIndex::kRuntimeString) {
    return prefix_search_all(w.AsNoCheck<string_view>(), pos);
  } else {
    return List();
  }
}

int TrieNode::save(const unicode_view& file_path) const {
  return trie_->save(UTF8Encode(file_path).c_str());
}

int TrieNode::load(const unicode_view& file_path) const {
  return trie_->open(UTF8Encode(file_path).c_str());
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(TrieNode);

}  // namespace runtime
}  // namespace matxscript
