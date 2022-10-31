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
#include <matxscript/runtime/container/native_func_private.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/function.h>

namespace matxscript {
namespace runtime {

TEST(List, Constructor) {
  List a{"hello", "world"};
  std::vector<List::value_type> b{"hello", "world"};
  auto new_a2 = List(a.begin(), a.end());
  std::cout << new_a2 << std::endl;
  EXPECT_EQ(a, new_a2);
  auto new_a3 = List(b.begin(), b.end());
  std::cout << new_a3 << std::endl;
  EXPECT_EQ(a, new_a3);
}

TEST(List, len) {
  List list1{Unicode(U"hello"), Unicode(U"world")};
  ASSERT_EQ(list1.size(), 2);
  List list2{list1, Unicode(U"hello"), Unicode(U"world")};
  ASSERT_EQ(list2.size(), 3);
}

TEST(List, equal) {
  List list1{1, 2, 3};
  List list2{1, 2, 3};
  List list3{1, 2};
  ASSERT_EQ(list1, list2);
  ASSERT_NE(list1, list3);
}

TEST(List, get_item) {
  List list1{Unicode(U"hello"), Unicode(U"world")};
  List list2{list1, Unicode(U"hello"), Unicode(U"world"), 5};
  ASSERT_EQ(list2.get_item(0).As<List>(), list1);
  ASSERT_EQ(list2.get_item(1), Unicode(U"hello"));
  ASSERT_EQ(list2.get_item(3).As<int>(), 5);
  ASSERT_NE(list2.get_item(1), Unicode(U"world"));
  ASSERT_NE(list1, list2);
  ASSERT_EQ(list1, list1);
}

TEST(List, set_item) {
  List list1{Unicode(U"hello"), 42};
  List list2{Unicode(U"hello"), Unicode(U"world")};
  list2.set_item(1, 42);
  ASSERT_EQ(list1, list2);
  list2.set_item(0, 42);
  ASSERT_NE(list1, list2);
}

TEST(List, get_slice) {
  List list1{Unicode(U"hello"), Unicode(U"world")};
  List list2{Unicode(U"hello"), Unicode(U"world"), 5};
  ASSERT_EQ(list2.get_slice(0, 2), list1);
  ASSERT_EQ(list2.get_slice(0, 3, 2), (List{Unicode(U"hello"), 5}));
  ASSERT_NE(list2.get_slice(0, 3, 2), list1);
}

TEST(List, set_slice) {
  List list1{1, 2, 5};
  List list2{Unicode(U"hello"), Unicode(U"world"), 5};
  list2.set_slice(0, 2, List{1, 2});
  ASSERT_EQ(list1, list2);
}

TEST(List, append) {
  List list1{Unicode(U"hello"), Unicode(U"world")};
  List list2{Unicode(U"hello"), Unicode(U"world"), 5};
  List list3{Unicode(U"hello"), Unicode(U"world"), 5, list2};
  list1.append(5);
  ASSERT_EQ(list1, list2);
  list1.append(list2);
  ASSERT_EQ(list1, list3);
  list3.append(list3);
  ASSERT_EQ(list3, list3);
}

TEST(List, extend) {
  List list1{1, 2};
  List list2{3};
  list1.extend(list2);
  ASSERT_EQ(list1, (List{1, 2, 3}));
}

TEST(List, repeat) {
  List list1{1, 1};
  List list2{1};
  ASSERT_EQ(list1.repeat(3), list2.repeat(6));
}

TEST(List, repeat_one) {
  {
    // int
    auto repeat_value = 1;
    auto repeat_num = 6;
    RTValue lvalue(repeat_value);
    List list1 = List::repeat_one(lvalue, repeat_num);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i].As<int64_t>(), repeat_value);
    }
  }
  {
    // float
    auto repeat_value = 1.0;
    auto repeat_num = 6;
    List list1 = List::repeat_one(repeat_value, repeat_num);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_FLOAT_EQ(list1[i].As<double>(), repeat_value);
    }
  }

  {
    // String
    auto* repeat_value = "hello";
    auto repeat_num = 6;
    RTValue lvalue(repeat_value);
    List list1 = List::repeat_one(lvalue, repeat_num);
    EXPECT_EQ(lvalue.As<string_view>(), repeat_value);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i].As<string_view>(), repeat_value);
    }
  }

  {
    // String
    auto* repeat_value = "hello";
    auto repeat_num = 6;
    RTValue rvalue(repeat_value);
    List list1 = List::repeat_one(std::move(rvalue), repeat_num);
    EXPECT_TRUE(rvalue.is_nullptr());
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i].As<string_view>(), repeat_value);
    }
  }

  {
    // Unicode
    auto* repeat_value = U"hello";
    auto repeat_num = 6;
    RTValue lvalue(repeat_value);
    List list1 = List::repeat_one(lvalue, repeat_num);
    EXPECT_EQ(lvalue.As<unicode_view>(), repeat_value);
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i].As<unicode_view>(), repeat_value);
    }
  }

  {
    // Unicode
    auto* repeat_value = U"hello";
    auto repeat_num = 6;
    RTValue rvalue(repeat_value);
    List list1 = List::repeat_one(std::move(rvalue), repeat_num);
    EXPECT_TRUE(rvalue.is_nullptr());
    EXPECT_EQ(list1.size(), repeat_num);
    for (auto i = 0; i < repeat_num; ++i) {
      EXPECT_EQ(list1[i].As<unicode_view>(), repeat_value);
    }
  }
}

TEST(List, repeat_many) {
  std::initializer_list<List::value_type> repeat_value = {"Hello", U"World", 3};
  auto repeat_num = 6;
  List list1 = List::repeat_many(repeat_value, repeat_num);
  EXPECT_EQ(list1.size(), repeat_num * repeat_value.size());
  for (auto i = 0; i < repeat_num; ++i) {
    auto init_len = repeat_value.size();
    for (auto k = 0; k < init_len; ++k) {
      auto& init_v = *(repeat_value.begin() + k);
      EXPECT_TRUE(Any::Equal(list1[i * init_len + k], init_v));
    }
  }
}

TEST(List, resize) {
  List list;
  ASSERT_EQ(list.size(), 0);
  list.resize(5);
  ASSERT_EQ(list.size(), 5);
}

TEST(List, sort) {
  List list;
  int64_t start = 0;
  int64_t end = 10;
  for (int64_t i = start; i < end; ++i) {
    list.append(i);
  }

  list.sort(true);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_TRUE(Any::Equal(list[i], RTView(end - start - i - 1)));
  }

  list.sort(false);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_TRUE(Any::Equal(list[i], RTView(i)));
  }

  auto creator = [](void* data) -> void* { return new (data) NativeFuncUserData(); };
  auto deleter = [](ILightUserData* data) { ((NativeFuncUserData*)(data))->~NativeFuncUserData(); };
  NativeFunction native_func = [](PyArgs args) { return RTValue(-args[0].AsNoCheck<int64_t>()); };
  auto func = UserDataRef(0, 0, sizeof(NativeFuncUserData), creator, deleter, nullptr);
  ((NativeFuncUserData*)(func->ud_ptr))->__call__ = &native_func;

  list.sort(RTView(func), false);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_TRUE(Any::Equal(list[i], RTView(end - start - i - 1)));
  }

  list.sort(RTView(func), true);
  for (int64_t i = start; i < end; ++i) {
    EXPECT_TRUE(Any::Equal(list[i], RTView(i)));
  }

  func = UserDataRef(nullptr);
}

}  // namespace runtime
}  // namespace matxscript
