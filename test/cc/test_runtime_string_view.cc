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
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(StringView, ConstChars) {
  string_view a = "hello";
  EXPECT_EQ(a, "hello");
  EXPECT_NE(a, "hh");
  string_view b("hello");
  EXPECT_EQ(a, b);
  string_view c("");
  EXPECT_EQ(c, "");
  EXPECT_NE(c, "hh");
  EXPECT_NE(c, a);
  std::cout << a << std::endl;
}

TEST(StringView, std_string) {
  std::string std_s("hello");
  string_view a = "hello";
  EXPECT_EQ(a, std_s);
  a = std_s;
  EXPECT_EQ(a, std_s);
  std::string new_std_s(a.data(), a.size());
  EXPECT_EQ(new_std_s, std_s);
  std::cout << a << std::endl;
}

TEST(StringView, LargeConverter) {
  String raw;
  raw.resize(4096, 'a');
  string_view raw_view = raw;
  String copy1 = raw;
  String copy2 = raw_view;
  EXPECT_TRUE(copy1.data() == copy2.data());
  RTView any_copy1 = raw;
  RTView any_copy2 = raw_view;
  EXPECT_TRUE(any_copy1.value().data.v_str_store.v_ml.bytes ==
              any_copy2.value().data.v_str_store.v_ml.bytes);
  string_view from_any1 = any_copy1.As<string_view>();
  string_view from_any2 = any_copy2.As<string_view>();
  EXPECT_TRUE(from_any1.data() == from_any2.data());
  RTValue val_copy1 = raw;
  RTValue val_copy2 = raw_view;
  EXPECT_TRUE(val_copy1.value().data.v_str_store.v_ml.bytes ==
              val_copy2.value().data.v_str_store.v_ml.bytes);
  string_view from_val1 = val_copy1.As<string_view>();
  string_view from_val2 = val_copy2.As<string_view>();
  EXPECT_TRUE(from_val1.data() == from_val2.data());
}

TEST(StringView, MediumConverter) {
  String raw;
  raw.resize(128, 'a');
  string_view raw_view = raw;
  String copy1 = raw;
  String copy2 = raw_view;
  EXPECT_TRUE(copy1.data() != copy2.data());
  EXPECT_TRUE(copy1 == copy2);
  RTView any_copy1 = raw;
  RTView any_copy2 = raw_view;
  EXPECT_TRUE(any_copy1.value().data.v_str_store.v_ml.bytes ==
              any_copy2.value().data.v_str_store.v_ml.bytes);
  string_view from_any1 = any_copy1.As<string_view>();
  string_view from_any2 = any_copy2.As<string_view>();
  EXPECT_TRUE(from_any1.data() == from_any2.data());
  EXPECT_TRUE(from_any1 == from_any2);
  RTValue val_copy1 = raw;
  RTValue val_copy2 = raw_view;
  EXPECT_TRUE(val_copy1.value().data.v_str_store.v_ml.bytes !=
              val_copy2.value().data.v_str_store.v_ml.bytes);
  string_view from_val1 = val_copy1.As<string_view>();
  string_view from_val2 = val_copy2.As<string_view>();
  EXPECT_TRUE(from_val1 == from_val2);
  EXPECT_TRUE(from_val1.data() != from_val2.data());
}

TEST(StringView, SmallConverter) {
  String raw;
  raw.resize(1, 'a');
  string_view raw_view = raw;
  String copy1 = raw;
  String copy2 = raw_view;
  EXPECT_TRUE(copy1.data() != copy2.data());
  EXPECT_TRUE(copy1 == copy2);
  RTView any_copy1 = raw;
  RTView any_copy2 = raw_view;
  EXPECT_TRUE(any_copy1.value().data.v_str_store.v_small_bytes !=
              any_copy2.value().data.v_str_store.v_small_bytes);
  string_view from_any1 = any_copy1.As<string_view>();
  string_view from_any2 = any_copy2.As<string_view>();
  EXPECT_TRUE(from_any1.data() != from_any2.data());
  EXPECT_TRUE(from_any1 == from_any2);
  RTValue val_copy1 = raw;
  RTValue val_copy2 = raw_view;
  EXPECT_TRUE(val_copy1.value().data.v_str_store.v_small_bytes !=
              val_copy2.value().data.v_str_store.v_small_bytes);
  string_view from_val1 = val_copy1.As<string_view>();
  string_view from_val2 = val_copy2.As<string_view>();
  EXPECT_TRUE(from_val1 == from_val2);
  EXPECT_TRUE(from_val1.data() != from_val2.data());
}

}  // namespace runtime
}  // namespace matxscript
