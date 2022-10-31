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
#include <matxscript/runtime/container/native_func_private.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/function.h>

namespace matxscript {
namespace runtime {

TEST(FTList, Constructor) {
  FTList<String> a{"hello", "world"};
  std::vector<String> b{"hello", "world"};
  auto new_a2 = FTList<String>(a.begin(), a.end());
  std::cout << new_a2 << std::endl;
  EXPECT_EQ(a, new_a2);
  auto new_a3 = FTList<String>(b.begin(), b.end());
  std::cout << new_a3 << std::endl;
  EXPECT_EQ(a, new_a3);

  {
    // test bool
    FTList<bool> b{true, false};
    std::cout << b << std::endl;
  }
}

TEST(FTList, RTValue) {
  FTList<String> ft_cons1{"hello", "world"};
  RTValue any_v1 = ft_cons1;
  EXPECT_EQ(any_v1, ft_cons1);
  RTValue any_v2(ft_cons1);
  std::cout << any_v2 << std::endl;
  EXPECT_TRUE(any_v2.IsObjectRef<FTObjectBase>());
  FTList<String> ft_cons2(any_v2.As<FTList<String>>());
  EXPECT_EQ(any_v2, ft_cons2);
  FTList<String> ft_cons3 = any_v2.As<FTList<String>>();
  EXPECT_EQ(any_v2, ft_cons3);
  EXPECT_ANY_THROW(any_v2.As<FTList<FTList<String>>>());
  FTList<RTValue> generic;
}

TEST(FTList, len) {
  FTList<Unicode> list1{Unicode(U"hello"), Unicode(U"world")};
  ASSERT_EQ(list1.size(), 2);
}

TEST(FTList, equal) {
  FTList<int64_t> list1{1, 2, 3};
  FTList<int64_t> list2{1, 2, 3};
  FTList<int64_t> list3{1, 2};
  FTList<Unicode> list4{U""};
  FTList<RTValue> list5{1, 2, 3};
  FTList<RTValue> list6{1, U"", ""};
  ASSERT_EQ(list1, list2);
  ASSERT_NE(list1, list3);
  ASSERT_NE(list3, list4);
  ASSERT_EQ(list1, list5);
  ASSERT_NE(list1, list6);
}

TEST(FTList, contains) {
  {
    FTList<int64_t> list1{1, 2, 3};
    EXPECT_TRUE(list1.contains(1));
    EXPECT_TRUE(list1.contains(1.0));
    EXPECT_FALSE(list1.contains(4));
    EXPECT_FALSE(list1.contains(""));
    EXPECT_TRUE(list1.contains(RTValue(1)));
    EXPECT_FALSE(list1.contains(RTValue(4)));
    EXPECT_FALSE(list1.contains(RTValue("")));
  }
  {
    FTList<RTValue> list1{1, 2, 3};
    EXPECT_TRUE(list1.contains(1));
    EXPECT_FALSE(list1.contains(4));
    EXPECT_FALSE(list1.contains(""));
    EXPECT_TRUE(list1.contains(RTValue(1)));
    EXPECT_FALSE(list1.contains(RTValue(4)));
    EXPECT_FALSE(list1.contains(RTValue("")));
  }
}

TEST(FTList, get_item) {
  {
    FTList<Unicode> list1{Unicode(U"hello"), Unicode(U"world")};
    ASSERT_EQ(list1.get_item(0), Unicode(U"hello"));
    ASSERT_NE(list1.get_item(1), Unicode(U"xx"));
    EXPECT_THROW(list1.get_item(2), std::exception);
  }
  {
    FTList<RTValue> list1{Unicode(U"hello"), Unicode(U"world")};
    ASSERT_EQ(list1.get_item(0), Unicode(U"hello"));
    ASSERT_NE(list1.get_item(1), Unicode(U"xx"));
    EXPECT_THROW(list1.get_item(2), std::exception);
  }
}

TEST(FTList, set_item) {
  {
    FTList<Unicode> list2{Unicode(U"hello"), Unicode(U"world")};
    list2.set_item(1, U"haha");
    EXPECT_ANY_THROW(list2.set_item(1, RTValue("haha")));
    EXPECT_EQ(list2.get_item(1), U"haha");
  }
  {
    FTList<RTValue> list2{Unicode(U"hello"), Unicode(U"world")};
    list2.set_item(1, U"haha");
    EXPECT_EQ(list2.get_item(1), U"haha");
    EXPECT_NO_THROW(list2.set_item(1, RTValue("haha")));
  }
}

TEST(FTList, get_slice) {
  FTList<Unicode> list1{Unicode(U"hello"), Unicode(U"world")};
  FTList<Unicode> list2{Unicode(U"hello"), Unicode(U"world"), U"5"};
  ASSERT_EQ(list2.get_slice(0, 2), list1);
  ASSERT_EQ(list2.get_slice(0, 3, 2), (FTList<Unicode>{Unicode(U"hello"), U"5"}));
  ASSERT_NE(list2.get_slice(0, 3, 2), list1);
}

TEST(FTList, set_slice) {
  {
    FTList<int> list1{1, 2, 5};
    FTList<int> list2{0, 0, 5};
    list2.set_slice(0, 2, FTList<int>{1, 2});
    ASSERT_EQ(list1, list2);
  }
  {
    FTList<RTValue> list1{1, 2, 5};
    FTList<RTValue> list2{0, 0, 5};
    list2.set_slice(0, 2, FTList<RTValue>{1, 2});
    ASSERT_EQ(list1, list2);
  }
  {
    FTList<int> list1{1, 2, 5};
    FTList<int> list2{0, 0, 5};
    list2.set_slice(0, 2, FTList<RTValue>{1, 2});
    ASSERT_EQ(list1, list2);
  }
  {
    FTList<int> list1{1, 2, 5};
    EXPECT_ANY_THROW(list1.set_slice(0, 2, FTList<RTValue>{"", 2}));
  }
  {
    FTList<RTValue> list1{1, 2, 5};
    FTList<RTValue> list2{0, 0, 5};
    list2.set_slice(0, 2, FTList<int>{1, 2});
    ASSERT_EQ(list1, list2);
  }
}

TEST(FTList, append) {
  {
    FTList<Unicode> list1{Unicode(U"hello"), Unicode(U"world")};
    list1.append(U"xx");
    ASSERT_EQ(list1.size(), 3);
    ASSERT_EQ(list1.get_item(2), U"xx");
  }
  {
    FTList<RTValue> list1{Unicode(U"hello"), Unicode(U"world")};
    list1.append(U"xx");
    ASSERT_EQ(list1.size(), 3);
    ASSERT_EQ(list1.get_item(2), U"xx");
  }
}

TEST(FTList, extend) {
  {
    FTList<int> list1{1, 2};
    FTList<int> list2{3};
    list1.extend(list2);
    ASSERT_EQ(list1, (FTList<int>{1, 2, 3}));
    ASSERT_EQ(list1, (FTList<RTValue>{1, 2, 3}));
  }
  {
    FTList<RTValue> list1{1, 2};
    FTList<RTValue> list2{3};
    list1.extend(list2);
    ASSERT_EQ(list1, (FTList<RTValue>{1, 2, 3}));
    ASSERT_EQ(list1, (FTList<int>{1, 2, 3}));
  }
  {
    FTList<RTValue> list1{1, 2};
    FTList<int> list2{3};
    list1.extend(list2);
    ASSERT_EQ(list1, (FTList<RTValue>{1, 2, 3}));
    ASSERT_EQ(list1, (FTList<int>{1, 2, 3}));
  }
  {
    FTList<int> list1{1, 2};
    FTList<RTValue> list2{3};
    list1.extend(list2);
    ASSERT_EQ(list1, (FTList<RTValue>{1, 2, 3}));
    ASSERT_EQ(list1, (FTList<int>{1, 2, 3}));
  }
}

TEST(FTList, repeat) {
  {
    FTList<int> list1{1, 1};
    FTList<int> list2{1};
    ASSERT_EQ(list1.repeat(3), list2.repeat(6));
  }
  {
    FTList<RTValue> list1{1, 1};
    FTList<RTValue> list2{1};
    ASSERT_EQ(list1.repeat(3), list2.repeat(6));
  }
}

TEST(FTList, repeat_one) {
  {
    // int
    auto repeat_value = 1;
    auto repeat_num = 6;
    auto list1 = FTList<int>::repeat_one(repeat_value, repeat_num);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i], repeat_value);
    }
  }
  {
    // int
    auto repeat_value = 1.0;
    auto repeat_num = 6;
    auto list1 = FTList<float>::repeat_one(repeat_value, repeat_num);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_FLOAT_EQ(list1[i], repeat_value);
    }
  }

  {
    // String
    auto* repeat_value = "hello";
    auto repeat_num = 6;
    String lvalue(repeat_value);
    auto list1 = FTList<String>::repeat_one(lvalue, repeat_num);
    EXPECT_EQ(lvalue, repeat_value);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i], repeat_value);
    }
  }

  {
    // String
    auto* repeat_value = "hello";
    auto repeat_num = 6;
    String rvalue(repeat_value);
    auto list1 = FTList<String>::repeat_one(std::move(rvalue), repeat_num);
    EXPECT_TRUE(rvalue.empty());
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i], repeat_value);
    }
  }

  {
    // Unicode
    auto* repeat_value = U"hello";
    auto repeat_num = 6;
    Unicode lvalue(repeat_value);
    auto list1 = FTList<Unicode>::repeat_one(lvalue, repeat_num);
    EXPECT_EQ(lvalue, repeat_value);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i], repeat_value);
    }
  }

  {
    // Unicode
    auto* repeat_value = U"hello";
    auto repeat_num = 6;
    Unicode rvalue(repeat_value);
    auto list1 = FTList<Unicode>::repeat_one(std::move(rvalue), repeat_num);
    EXPECT_TRUE(rvalue.empty());
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i], repeat_value);
    }
  }
}

TEST(FTList, repeat_many) {
  {
    std::initializer_list<int> repeat_value = {1, 2};
    auto repeat_num = 6;
    auto list1 = FTList<int>::repeat_many(repeat_value, repeat_num);
    EXPECT_EQ(list1.size(), repeat_num * repeat_value.size());
    for (auto i = 0; i < repeat_num; ++i) {
      auto init_len = repeat_value.size();
      for (auto k = 0; k < init_len; ++k) {
        auto& init_v = *(repeat_value.begin() + k);
        EXPECT_EQ(list1[i * init_len + k], init_v);
      }
    }
  }
  {
    std::initializer_list<String> repeat_value = {"Hello", "World"};
    auto repeat_num = 6;
    auto list1 = FTList<String>::repeat_many(repeat_value, repeat_num);
    EXPECT_EQ(list1.size(), repeat_num * repeat_value.size());
    for (auto i = 0; i < repeat_num; ++i) {
      auto init_len = repeat_value.size();
      for (auto k = 0; k < init_len; ++k) {
        auto& init_v = *(repeat_value.begin() + k);
        EXPECT_EQ(list1[i * init_len + k], init_v);
      }
    }
  }
  {
    std::initializer_list<Unicode> repeat_value = {U"Hello", U"World"};
    auto repeat_num = 6;
    auto list1 = FTList<Unicode>::repeat_many(repeat_value, repeat_num);
    EXPECT_EQ(list1.size(), repeat_num * repeat_value.size());
    for (auto i = 0; i < repeat_num; ++i) {
      auto init_len = repeat_value.size();
      for (auto k = 0; k < init_len; ++k) {
        auto& init_v = *(repeat_value.begin() + k);
        EXPECT_EQ(list1[i * init_len + k], init_v);
      }
    }
  }
}

TEST(FTList, resize) {
  FTList<int> list;
  ASSERT_EQ(list.size(), 0);
  list.resize(5);
  ASSERT_EQ(list.size(), 5);
}

TEST(FTList, reserve) {
  FTList<int> list;
  EXPECT_NO_THROW(list.reserve(256));
}

TEST(FTList, capacity) {
  FTList<int> list;
  EXPECT_NO_THROW(list.capacity());
}

TEST(FTList, pop) {
  {
    FTList<int> list1{1, 2};
    EXPECT_EQ(list1.pop(), 2);
  }
  {
    FTList<int> list1{1, 2};
    EXPECT_EQ(list1.pop(0), 1);
  }
  {
    FTList<RTValue> list1{1, 2};
    EXPECT_EQ(list1.pop(), 2);
  }
  {
    FTList<RTValue> list1{1, 2};
    EXPECT_EQ(list1.pop(0), 1);
  }
}

TEST(FTList, remove) {
  {
    FTList<int> list1{1, 2};
    list1.remove(2);
    EXPECT_EQ(list1, FTList<int>{1});
    EXPECT_EQ(list1, FTList<RTValue>{1});
  }
  {
    FTList<int> list1{1, 2};
    list1.remove(RTValue(2));
    EXPECT_EQ(list1, FTList<int>{1});
    EXPECT_EQ(list1, FTList<RTValue>{1});
  }
  {
    FTList<RTValue> list1{1, 2};
    list1.remove(2);
    EXPECT_EQ(list1, FTList<int>{1});
    EXPECT_EQ(list1, FTList<RTValue>{1});
  }
  {
    FTList<RTValue> list1{1, 2};
    list1.remove(RTValue(2));
    EXPECT_EQ(list1, FTList<int>{1});
    EXPECT_EQ(list1, FTList<RTValue>{1});
  }
  {
    FTList<int> list1{1, 2};
    EXPECT_ANY_THROW(list1.remove(3));
    EXPECT_ANY_THROW(list1.remove(RTValue(3)));
    EXPECT_ANY_THROW(list1.remove(RTValue("")));
    EXPECT_ANY_THROW(list1.remove(""));
  }
  {
    FTList<RTValue> list1{1, 2};
    EXPECT_ANY_THROW(list1.remove(3));
    EXPECT_ANY_THROW(list1.remove(RTValue(3)));
    EXPECT_ANY_THROW(list1.remove(RTValue("")));
    EXPECT_ANY_THROW(list1.remove(""));
  }
}

TEST(FTList, clear) {
  {
    FTList<int> list1{1, 2};
    list1.clear();
    EXPECT_EQ(list1, FTList<int>{});
  }
  {
    FTList<RTValue> list1{1, 2};
    list1.clear();
    EXPECT_EQ(list1, FTList<RTValue>{});
  }
}

TEST(FTList, reverse) {
  {
    FTList<int> list1{1, 2};
    list1.reverse();
    EXPECT_EQ(list1, FTList<int>({2, 1}));
  }
  {
    FTList<RTValue> list1{1, 2};
    list1.reverse();
    EXPECT_EQ(list1, FTList<RTValue>({2, 1}));
  }
}

TEST(FTList, count) {
  {
    FTList<int> list1{1, 1, 2};
    EXPECT_EQ(list1.count(1), 2);
    EXPECT_EQ(list1.count(2), 1);
    EXPECT_EQ(list1.count(RTValue(1)), 2);
    EXPECT_EQ(list1.count(RTValue(2)), 1);
  }
  {
    FTList<RTValue> list1{1, 1, 2};
    EXPECT_EQ(list1.count(1), 2);
    EXPECT_EQ(list1.count(2), 1);
    EXPECT_EQ(list1.count(RTValue(1)), 2);
    EXPECT_EQ(list1.count(RTValue(2)), 1);
  }
}

TEST(FTList, sort) {
  FTList<int64_t> list;
  int64_t start = 0;
  int64_t end = 10;
  for (int64_t i = start; i < end; ++i) {
    list.append(i);
  }

  list.sort(true);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_EQ(list.get_item(i), end - start - i - 1);
  }

  list.sort(false);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_EQ(list.get_item(i), i);
  }

  auto creator = [](void* data) -> void* { return new (data) NativeFuncUserData(); };
  auto deleter = [](ILightUserData* data) { ((NativeFuncUserData*)(data))->~NativeFuncUserData(); };
  NativeFunction native_func = [](PyArgs args) { return RTValue(-args[0].AsNoCheck<int64_t>()); };
  auto func = UserDataRef(0, 0, sizeof(NativeFuncUserData), creator, deleter, nullptr);
  ((NativeFuncUserData*)(func->ud_ptr))->__call__ = &native_func;

  list.sort(RTView(func), false);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_EQ(list.get_item(i), end - start - i - 1);
  }

  list.sort(RTView(func), true);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_EQ(list.get_item(i), i);
  }

  func = UserDataRef(nullptr);
}

}  // namespace runtime
}  // namespace matxscript
