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
#include <matxscript/runtime/container.h>

namespace matxscript {
namespace runtime {

TEST(Dict, Constructor) {
  Dict a{{"hello", "world"}};
  std::vector<Dict::value_type> b{{"hello", "world"}};
  auto new_dict2 = Dict(a.item_begin(), a.item_end());
  std::cout << new_dict2 << std::endl;
  EXPECT_EQ(a, new_dict2);
  auto new_dict3 = Dict(b.begin(), b.end());
  std::cout << new_dict3 << std::endl;
  EXPECT_EQ(a, new_dict3);
}

TEST(Dict, len) {
  Dict d{{1, 2}, {Unicode(U"hello"), 3}};
  ASSERT_EQ(d.size(), 2);
}

TEST(Dict, equal) {
  Dict d1{{1, 2}, {Unicode(U"hello"), 3}};
  ASSERT_EQ(d1, d1);
  Dict d2{{1, 2}, {Unicode(U"hello"), 3}};
  ASSERT_EQ(d1, d2);
  Dict d3{{1, 2}, {Unicode(U"hello"), 4}};
  ASSERT_NE(d1, d3);
}

TEST(Dict, contains) {
  Dict d{{1, 2}, {Unicode(U"hello"), 3}, {3, 4}};
  ASSERT_TRUE(d.contains(1));
  ASSERT_TRUE(d.contains(Unicode(U"hello")));
  ASSERT_TRUE(d.contains(U"hello"));
  ASSERT_TRUE(d.contains(unicode_view(U"hello")));
  ASSERT_FALSE(d.contains(42));
}

TEST(Dict, get_item) {
  Dict d{{1, 2}, {Unicode(U"hello"), 3}, {3, Unicode(U"matx4")}};
  ASSERT_EQ(d[1], 2);
  ASSERT_EQ(d[Unicode(U"hello")], 3);
  ASSERT_EQ(d.get_item(Unicode(U"hello")), 3);
  ASSERT_EQ(d.get_item(U"hello"), 3);
  ASSERT_EQ(d.get_item(unicode_view(U"hello")), 3);
  ASSERT_EQ(d[3], Unicode(U"matx4"));
  ASSERT_NE(d[3], Unicode(U"matx3"));
  ASSERT_NE(d[1], 4);
}

TEST(Dict, set_item) {
  Dict d{{1, 2}, {Unicode(U"hello"), 3}, {3, Unicode(U"matx4")}};
  d[2] = 3;
  ASSERT_EQ(d[2], 3);
  d[Unicode(U"hello")] = Unicode(U"world");
  ASSERT_EQ(d[Unicode(U"hello")], Unicode(U"world"));
  d.set_item(U"hello unicode", U"world");
  d.set_item("hello bytes", "world");
  ASSERT_EQ(d[Unicode(U"hello unicode")], Unicode(U"world"));
  ASSERT_EQ(d["hello bytes"], String("world"));
  ASSERT_NE(d[U"hello bytes"], String("world"));
  ASSERT_NE(d[U"hello bytes"], Unicode(U"world"));
}

TEST(Dict, clear) {
  Dict d{{1, 2}, {Unicode(U"hello"), 3}, {3, Unicode(U"matx4")}};
  d.clear();
  ASSERT_EQ(d.size(), 0);
}

}  // namespace runtime
}  // namespace matxscript
