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
#include <matxscript/runtime/ft_container.h>
#include "matxscript/runtime/logging.h"

namespace matxscript {
namespace runtime {

TEST(FTSet, Constructor) {
  FTSet<String> a{"hello", "world"};
  std::vector<FTSet<String>::value_type> b{"hello", "world"};
  auto new_a2 = FTSet<String>(a.begin(), a.end());
  std::cout << new_a2 << std::endl;
  EXPECT_EQ(a, new_a2);
  auto new_a3 = FTSet<String>(b.begin(), b.end());
  std::cout << new_a3 << std::endl;
  EXPECT_EQ(a, new_a3);
}

TEST(FTSet, RTValue) {
  FTSet<String> ft_cons1{"hello", "world"};
  RTValue any_v1 = ft_cons1;
  EXPECT_EQ(any_v1, ft_cons1);
  RTValue any_v2(ft_cons1);
  EXPECT_TRUE(any_v2.IsObjectRef<FTObjectBase>());
  std::cout << any_v2 << std::endl;
  FTSet<String> ft_cons2(any_v2.As<FTSet<String>>());
  EXPECT_EQ(any_v2, ft_cons2);
  FTSet<String> ft_cons3 = any_v2.As<FTSet<String>>();
  EXPECT_EQ(any_v2, ft_cons3);
  EXPECT_ANY_THROW((any_v2.As<FTSet<Unicode>>()));
  FTSet<RTValue> generic;
}

TEST(FTSet, len) {
  FTSet<int> s1{1, 2, 3};
  ASSERT_EQ(s1.size(), 3);
}

TEST(FTSet, equal) {
  {
    FTSet<int> s1{1, 42};
    FTSet<int> s2{1, 42};
    ASSERT_EQ(s1, s2);
    ASSERT_EQ(s2, s2);
    s2.add(42);
    ASSERT_EQ(s1, s2);
    s2.add(10);
    ASSERT_NE(s1, s2);
  }
  {
    FTSet<RTValue> s1{1, 42};
    FTSet<RTValue> s2{1, 42};
    ASSERT_EQ(s1, s2);
    ASSERT_EQ(s2, s2);
    s2.add(42);
    ASSERT_EQ(s1, s2);
    s2.add(10);
    ASSERT_NE(s1, s2);
  }
  {
    FTSet<RTValue> s1{1, 42};
    FTSet<int64_t> s2{1, 42};
    ASSERT_EQ(s1, s2);
    ASSERT_EQ(s2, s2);
    s2.add(42);
    ASSERT_EQ(s1, s2);
    s2.add(10);
    // TODO: fix eq
    // ASSERT_NE(s1, s2);
  }
}

TEST(FTSet, add) {
  {
    FTSet<int> s;
    s.add(1);
    ASSERT_EQ(s.size(), 1);
    s.add(1);
    ASSERT_EQ(s.size(), 1);
    ASSERT_TRUE(s.contains(1));
    EXPECT_ANY_THROW(s.add(RTValue("hi")));
  }
  {
    FTSet<RTValue> s;
    s.add(1);
    ASSERT_EQ(s.size(), 1);
    s.add("hi");
    ASSERT_EQ(s.size(), 2);
    ASSERT_TRUE(s.contains(1));
    ASSERT_TRUE(s.contains("hi"));
  }
}

TEST(FTSet, contains) {
  {
    FTSet<int64_t> s1;
    s1.add(1);
    ASSERT_TRUE(s1.contains(1));
    ASSERT_TRUE(s1.contains(1.0));
  }
  {
    Unicode str1{U"Hello"};
    FTSet<Unicode> s1;
    s1.add(str1);
    ASSERT_TRUE(s1.contains(str1));
    ASSERT_TRUE(s1.contains(unicode_view(str1)));
    str1 += U" world";
    ASSERT_FALSE(s1.contains(str1));
    Unicode str2{U"Hello"};
    ASSERT_TRUE(s1.contains(str2));
    EXPECT_FALSE(s1.contains(1));
    EXPECT_FALSE(s1.contains(1.0));
    EXPECT_FALSE(s1.contains(true));
    EXPECT_FALSE(s1.contains(str2.encode()));
  }
  {
    Unicode str1{U"Hello"};
    FTSet<RTValue> s1;
    s1.add(str1);
    ASSERT_TRUE(s1.contains(str1));
    str1 += U" world";
    ASSERT_FALSE(s1.contains(str1));
    Unicode str2{U"Hello"};
    ASSERT_TRUE(s1.contains(str2));
    EXPECT_FALSE(s1.contains(1));
    EXPECT_FALSE(s1.contains(1.0));
    EXPECT_FALSE(s1.contains(true));
    EXPECT_FALSE(s1.contains(str2.encode()));
  }
}

TEST(FTSet, clear) {
  FTSet<int> s1{1, 2, 3};
  s1.clear();
  ASSERT_EQ(s1.size(), 0);
}

TEST(FTSet, difference) {
  {
    FTSet<int> s1{1, 2, 3};
    ASSERT_FALSE(s1.difference({FTSet<RTValue>{1}}).contains(1));
    ASSERT_FALSE(s1.difference({FTSet<RTValue>{1}, FTSet<RTValue>{1}}).contains(1));
    ASSERT_FALSE(s1.difference({FTSet<RTValue>{1}, FTSet<RTValue>{2}}).contains(2));
    ASSERT_FALSE(s1.difference({FTSet<RTValue>{1}, FTSet<RTValue>{2}.iter()}).contains(2));
  }
  {
    FTSet<RTValue> s1{1, 2, 3};
    ASSERT_FALSE(s1.difference({FTSet<int>{1}}).contains(1));
    ASSERT_FALSE(s1.difference({FTSet<int>{1}, FTSet<int>{1}}).contains(1));
    ASSERT_FALSE(s1.difference({FTSet<RTValue>{1}, FTSet<int>{2}}).contains(2));
    ASSERT_FALSE(s1.difference({FTSet<RTValue>{1}, FTSet<int>{2}.iter()}).contains(2));
  }
}

TEST(FTSet, difference_update) {
  {
    FTSet<int> s1{1, 2, 3};
    s1.difference_update({FTSet<RTValue>{1}});
    ASSERT_FALSE(s1.contains(1));
    EXPECT_NO_THROW(s1.difference_update({FTSet<Unicode>{U"hi"}}));
  }
  {
    FTSet<RTValue> s1{1, 2, 3};
    s1.difference_update({FTSet<int64_t>{1}});
    ASSERT_FALSE(s1.contains(1));
    EXPECT_NO_THROW(s1.difference_update({FTSet<Unicode>{U"hi"}}));
  }
}

TEST(FTSet, set_union) {
  {
    FTSet<int> s1{1, 2, 3};
    auto s2 = s1.set_union({FTSet<RTValue>{4}});
    ASSERT_TRUE(s1.contains(4));
    EXPECT_ANY_THROW(s1.set_union({FTSet<Unicode>{U"hi"}}));
  }
  {
    FTSet<RTValue> s1{1, 2, 3};
    auto s2 = s1.set_union({FTSet<RTValue>{4}});
    ASSERT_TRUE(s1.contains(4));
    EXPECT_NO_THROW(s1.set_union({FTSet<Unicode>{U"hi"}}));
  }
}

TEST(FTSet, discard) {
  {
    FTSet<int> s1{1, 2, 3};
    s1.discard(1);
    ASSERT_FALSE(s1.contains(1));
    EXPECT_NO_THROW(s1.discard(Unicode{U"hi"}));
    // TypeError: unhashable type: 'set'
    EXPECT_ANY_THROW(s1.discard(FTSet<Unicode>{U"hi"}));
  }
  {
    FTSet<RTValue> s1{1, 2, 3};
    s1.discard(1);
    ASSERT_FALSE(s1.contains(1));
    EXPECT_NO_THROW(s1.discard(Unicode{U"hi"}));
    // TypeError: unhashable type: 'set'
    EXPECT_ANY_THROW(s1.discard(FTSet<Unicode>{U"hi"}));
  }
}

TEST(FTSet, update) {
  {
    FTSet<int> s1{1, 2, 3};
    s1.update({FTSet<RTValue>{4}});
    ASSERT_TRUE(s1.contains(4));
    EXPECT_ANY_THROW(s1.update({FTSet<Unicode>{U"hi"}}));
  }
  {
    FTSet<RTValue> s1{1, 2, 3};
    s1.update({FTSet<RTValue>{4}});
    ASSERT_TRUE(s1.contains(4));
    EXPECT_NO_THROW(s1.update({FTSet<Unicode>{U"hi"}}));
    ASSERT_TRUE(s1.contains(U"hi"));
  }
}

}  // namespace runtime
}  // namespace matxscript
