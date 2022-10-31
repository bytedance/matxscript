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

TEST(Tuple, len) {
  auto tup = Tuple::dynamic(1, 2.5f, "a", List({1, 2}));
  ASSERT_EQ(tup.size(), 4);
}

TEST(Tuple, iterator) {
  auto tup = Tuple::dynamic(1, 2.5f, "a", List({1, 2}));
  for (auto& item : tup) {
    std::cout << item << std::endl;
  }
}

TEST(Tuple, hash) {
  auto tup = Tuple::dynamic(1, "a");
  auto hash = std::hash<RTValue>()(tup);
  std::cout << hash << std::endl;
}

TEST(Tuple, equal) {
  auto tup1 = Tuple::dynamic(1, "a", List({1, 2}));
  auto tup2 = Tuple::dynamic(1, "a", List({1, 2}));
  EXPECT_EQ(tup1, tup2);
}

TEST(Tuple, AsDictKey) {
  Dict t;
  auto tup1 = Tuple::dynamic(1, "a");
  auto tup2 = Tuple::dynamic(1, "a");
  t[tup1] = 1;
  EXPECT_TRUE(t.contains(RTView(tup2)));
}

}  // namespace runtime
}  // namespace matxscript
