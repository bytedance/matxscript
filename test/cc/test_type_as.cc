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
#include <iostream>

#include <gtest/gtest.h>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/type_helper_macros.h>

namespace matxscript {
namespace runtime {

extern std::string NormalizeError(std::string err_msg);

TEST(TypeAs, StackTraceWithPyInfo) {
  try {
    RTValue a;
    String message;
    message.append("File \"").append("abc.py").append("\"");
    message.append(", line ").append("10");
    message.append(", in ").append("my_mock_func").append("\n");
    message.append("   ");
    MATXSCRIPT_TYPE_AS_WITH_PY_INFO(a, int64_t, message.data());
  } catch (const std::runtime_error& e) {
    auto ret = NormalizeError(e.what());
    std::cout << ret << std::endl;
  }
}

TEST(TypeAs, MoveString) {
  constexpr char data[] = "Hello, World";
  RTValue d = data;
  auto s1 = MATXSCRIPT_TYPE_AS(d, String);
  EXPECT_EQ(s1, String(data));
  EXPECT_TRUE(d.Is<String>());
  EXPECT_EQ(d.As<String>(), String(data));
  auto s2 = MATXSCRIPT_TYPE_AS(std::move(d), String);
  EXPECT_EQ(s2, String(data));
  EXPECT_TRUE(d.is_nullptr());
}

TEST(TypeAs, MoveUnicode) {
  constexpr char32_t data[] = U"Hello, World";
  RTValue d = data;
  auto s1 = MATXSCRIPT_TYPE_AS(d, Unicode);
  EXPECT_EQ(s1, Unicode(data));
  EXPECT_TRUE(d.Is<Unicode>());
  EXPECT_EQ(d.As<Unicode>(), Unicode(data));
  auto s2 = MATXSCRIPT_TYPE_AS(std::move(d), Unicode);
  EXPECT_EQ(s2, Unicode(data));
  EXPECT_TRUE(d.is_nullptr());
}

TEST(TypeAs, MoveObject) {
  List obj{RTValue(U"hello")};
  RTValue d(obj);
  auto t1 = MATXSCRIPT_TYPE_AS(d, List);
  EXPECT_EQ(t1, obj);
  EXPECT_TRUE(d.Is<List>());
  EXPECT_EQ(d.As<List>(), obj);
  auto s2 = MATXSCRIPT_TYPE_AS(std::move(d), List);
  EXPECT_EQ(s2, obj);
  EXPECT_TRUE(d.is_nullptr());
}

}  // namespace runtime
}  // namespace matxscript
