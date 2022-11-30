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
#include <matxscript/runtime/algorithm/prefix_mapping.h>

#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/utf8_util.h>

namespace matxscript {
namespace runtime {

PrefixMapping::PrefixMapping(const std::map<String, int>& dic) {
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

int PrefixMapping::PrefixSearch(const char* w, size_t w_len, int* val) const {
  if (trie_ == nullptr) {
    return 0;
  }
  constexpr int kResultSize = 64;
  cedar_t::result_pair_type trie_results[kResultSize];
  const int num_nodes = trie_->commonPrefixSearch(w, trie_results, kResultSize, w_len);

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

}  // namespace runtime
}  // namespace matxscript
