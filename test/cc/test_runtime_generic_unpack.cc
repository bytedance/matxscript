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
#include <matxscript/runtime/generic/generic_unpack.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(Generic, kernel_builtins_len) {
  {
    Unicode a(U"abc");
    ASSERT_EQ(kernel_builtins_len(a), 3);
    Unicode b(U"\u6d4b\u8bd5\u96c6");
    ASSERT_EQ(kernel_builtins_len(b), 3);
  }
  {
    String a("abc");
    ASSERT_EQ(kernel_builtins_len(a), 3);
    String b("\u6d4b\u8bd5\u96c6");
    ASSERT_EQ(kernel_builtins_len(b), strlen("\u6d4b\u8bd5\u96c6"));
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    List b({1, Unicode(U"WORLD"), String("WORLD")});
    ASSERT_EQ(kernel_builtins_len(a), 3);
    ASSERT_EQ(kernel_builtins_len(b), 3);
  }
  {
    Dict a({{1, Unicode(U"HELLO")}, {String("HELLO"), String("World")}});
    ASSERT_EQ(kernel_builtins_len(a), 2);
  }
  {
    Set a({1, Unicode(U"HELLO"), String("HELLO")});
    Set b({1, Unicode(U"HELLO"), String("WORLD")});
    Set c({1, Unicode(U"HELLO"), Unicode(U"HELLO")});
    ASSERT_EQ(kernel_builtins_len(a), 3);
    ASSERT_EQ(kernel_builtins_len(b), 3);
    ASSERT_EQ(kernel_builtins_len(c), 2);
  }
  {
    auto d = std::make_pair(1, 2);
    EXPECT_EQ(kernel_builtins_len(d), 2);
  }
  {
    auto d = std::make_tuple(1, 2, 3);
    EXPECT_EQ(kernel_builtins_len(d), 3);
  }
}

TEST(Generic, kernel_builtins_unpack) {
  {
    Unicode a(U"abc");
    RTValue r = kernel_builtins_unpack<0, RTValue>(a);
    Unicode typed_r = r.As<Unicode>();
    ASSERT_EQ(typed_r, Unicode(U"a"));
    ASSERT_THROW((kernel_builtins_unpack<100, Unicode>(a)), Error);
  }
  {
    auto d = std::make_pair(1, 2);
    int64_t l1 = kernel_builtins_unpack<0, int64_t>(d);
    int64_t l2 = kernel_builtins_unpack<1, int64_t>(d);
    EXPECT_EQ(l1, 1);
    EXPECT_EQ(l2, 2);
  }
  {
    auto d = std::make_tuple(1, 2);
    int64_t l1 = kernel_builtins_unpack<0, int64_t>(d);
    int64_t l2 = kernel_builtins_unpack<1, int64_t>(d);
    EXPECT_EQ(l1, 1);
    EXPECT_EQ(l2, 2);
  }
}

}  // namespace runtime
}  // namespace matxscript
