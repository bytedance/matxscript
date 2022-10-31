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
#include <matxscript/runtime/algorithm/prefix_matcher.h>

#include <memory>
#include <set>
#include <string>

#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

PrefixMatcher::PrefixMatcher(const std::set<std::string>& dic) {
  if (dic.empty())
    return;
  std::vector<const char*> key;
  key.reserve(dic.size());
  for (const auto& it : dic) {
    key.push_back(it.data());
  }
  trie_ = std::make_unique<cedar_t>();
  MXCHECK_EQ(0, trie_->build(key.size(), const_cast<const char**>(&key[0]), nullptr, nullptr));
}

int PrefixMatcher::PrefixMatch(const char* w, size_t w_len, bool* found) const {
  if (trie_ == nullptr) {
    if (found) {
      *found = false;
    }
    return std::min<int>(w_len, OneCharLen(w));
  }
  constexpr int kResultSize = 64;
  cedar_t::result_pair_type trie_results[kResultSize];
  const int num_nodes = trie_->commonPrefixSearch(w, trie_results, kResultSize, w_len);

  if (found) {
    *found = (num_nodes > 0);
  }
  if (num_nodes == 0) {
    return std::min<int>(w_len, OneCharLen(w));
  }
  int mblen = 0;
  for (int i = 0; i < num_nodes; ++i) {
    mblen = std::max<int>(trie_results[i].length, mblen);
  }
  return mblen;
}

int PrefixMatcher::PrefixMatch(const std::string& w, bool* found) const {
  return PrefixMatch(w.data(), w.length(), found);
}

int PrefixMatcher::PrefixSearch(const char* w, size_t w_len) const {
  if (trie_ == nullptr) {
    return 0;
  }
  constexpr int kResultSize = 64;
  cedar_t::result_pair_type trie_results[kResultSize];
  const int num_nodes = trie_->commonPrefixSearch(w, trie_results, kResultSize, w_len);

  if (num_nodes == 0) {
    return 0;
  }
  int mblen = 0;
  for (int i = 0; i < num_nodes; ++i) {
    mblen = std::max<int>(trie_results[i].length, mblen);
  }
  return mblen;
}

}  // namespace runtime
}  // namespace matxscript
