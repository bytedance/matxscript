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
#include "matxscript/runtime/logging.h"

namespace matxscript {
namespace runtime {

TEST(Set, Constructor) {
  Set a{"hello", "world"};
  std::vector<Set::value_type> b{"hello", "world"};
  auto new_a2 = Set(a.begin(), a.end());
  std::cout << new_a2 << std::endl;
  EXPECT_EQ(a, new_a2);
  auto new_a3 = Set(b.begin(), b.end());
  std::cout << new_a3 << std::endl;
  EXPECT_EQ(a, new_a3);
}

TEST(set, len) {
  Set s1{1, 2, 3};
  ASSERT_EQ(s1.size(), 3);
}

TEST(set, equal) {
  Set s1{1, Unicode(U"hello"), 42};
  Set s2{1, Unicode(U"hello"), 42};
  ASSERT_EQ(s1, s2);
  ASSERT_EQ(s2, s2);
  s2.add(42);
  ASSERT_EQ(s1, s2);
  s2.add(10);
  ASSERT_NE(s1, s2);
}

TEST(set, add) {
  Set s;
  s.add(1);
  ASSERT_EQ(s.size(), 1);
  s.add(1);
  ASSERT_EQ(s.size(), 1);
  ASSERT_TRUE(s.contains(1));
  Unicode str1{U"hello"};
  s.add(str1);
  ASSERT_EQ(s.size(), 2);
  ASSERT_TRUE(s.contains(str1));
  str1 += U" world";
  s.add(str1);
  ASSERT_TRUE(s.contains(str1));
  ASSERT_EQ(s.size(), 3);
}

TEST(set, contains) {
  Unicode str1{U"Hello"};
  Set s1;
  s1.add(str1);
  ASSERT_TRUE(s1.contains(str1));
  str1 += U" world";
  ASSERT_FALSE(s1.contains(str1));
  Unicode str2{U"Hello"};
  ASSERT_TRUE(s1.contains(str2));
}

TEST(Set, clear) {
  Set s1{1, 2, 3};
  s1.clear();
  ASSERT_EQ(s1.size(), 0);
}

}  // namespace runtime
}  // namespace matxscript
