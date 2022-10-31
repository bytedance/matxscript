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
#include <gtest/gtest.h>
#include <matxscript/runtime/algorithm/trie_ref.h>
#include "matxscript/runtime/logging.h"

namespace matxscript {
namespace runtime {

TEST(Trie, PrefixMatch) {
  std::map<string_view, int64_t> test_data{
      {"hello world", 0},
      {"happy", 1},
      {"world", 2},
      {"hi", 3},
      {"worldhi", 4},
  };

  Trie trie(test_data);
  std::string data("world");
  int64_t index = -1;
  int len = trie.PrefixSearch(data, &index);
  ASSERT_EQ(index, 2);
  ASSERT_EQ(len, data.size());
  ASSERT_EQ(0, trie.PrefixSearch("je", &index));
  data = std::string("happy") + "world";
  auto match_len = trie.PrefixSearch(data);
  ASSERT_EQ(match_len, 5);
}

}  // namespace runtime
}  // namespace matxscript
