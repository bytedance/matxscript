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
#include <string>

#include <gtest/gtest.h>
#include <matxscript/runtime/container/string_core.h>
#include <matxscript/runtime/container/string_view.h>

namespace matxscript {
namespace runtime {

TEST(StringCore, Constructor) {
  // small init
  std::string small_data("hell");
  string_core<char> sc_small(small_data.data(), small_data.size());
  std::cout << sc_small.data() << std::endl;
  EXPECT_EQ(sc_small.size(), small_data.size());

  // small copy
  string_core<char> sc_small_copy(sc_small);
  std::cout << sc_small_copy.data() << std::endl;
  EXPECT_EQ(sc_small_copy.size(), small_data.size());

  // small move
  string_core<char> sc_small_move(std::move(sc_small));
  EXPECT_EQ(sc_small.size(), 0);
  EXPECT_EQ(sc_small.data()[0], '\0');
  std::cout << sc_small_move.data() << std::endl;
  EXPECT_EQ(sc_small_move.size(), small_data.size());

  // medium init
  std::string medium_data;
  medium_data.resize(126, 'm');
  string_core<char> sc_medium(medium_data.data(), medium_data.size());
  std::cout << sc_medium.data() << std::endl;
  EXPECT_EQ(sc_medium.size(), medium_data.size());

  // medium copy
  string_core<char> sc_medium_copy(sc_medium);
  std::cout << sc_medium_copy.data() << std::endl;
  EXPECT_EQ(sc_medium_copy.size(), medium_data.size());

  // medium move
  string_core<char> sc_medium_move(std::move(sc_medium));
  EXPECT_EQ(sc_medium.size(), 0);
  EXPECT_EQ(sc_medium.data()[0], '\0');
  std::cout << sc_medium_move.data() << std::endl;
  EXPECT_EQ(sc_medium_move.size(), medium_data.size());

  // large init
  std::string large_data;
  large_data.resize(510, 'l');
  string_core<char> sc_large(large_data.data(), large_data.size());
  std::cout << sc_large.data() << std::endl;
  EXPECT_EQ(sc_large.size(), large_data.size());

  // large copy
  string_core<char> sc_large_copy(sc_large);
  std::cout << sc_large_copy.data() << std::endl;
  EXPECT_EQ(sc_large_copy.size(), large_data.size());

  // large move
  string_core<char> sc_large_move(std::move(sc_large));
  EXPECT_EQ(sc_large.size(), 0);
  EXPECT_EQ(sc_large.data()[0], '\0');
  std::cout << sc_large_move.data() << std::endl;
  EXPECT_EQ(sc_large_move.size(), sc_large_move.size());
}

TEST(StringCore, MemberFunctions) {
  std::string small_data("hell");
  string_core<char> sc(small_data.data(), small_data.size());
  sc.push_back('o');
  EXPECT_EQ(sc.size(), 5);
  EXPECT_EQ(string_view(sc.data()), "hello");

  auto data_small_ptr = sc.data();
  sc.reserve(125);
  EXPECT_GE(sc.capacity(), 125);
  auto data_medium_ptr = sc.data();
  EXPECT_EQ(string_view(sc.data()), "hello");
  EXPECT_NE(data_small_ptr, data_medium_ptr);
  sc.reserve(259);
  EXPECT_GE(sc.capacity(), 259);
  EXPECT_EQ(string_view(sc.data()), "hello");
  auto data_large_ptr = sc.data();
  EXPECT_NE(data_large_ptr, data_small_ptr);
  EXPECT_NE(data_large_ptr, data_medium_ptr);

  sc.reserve(1024);
  EXPECT_EQ(string_view(sc.data()), "hello");
  EXPECT_NE(data_large_ptr, sc.data());

  sc.reserve(32);
  EXPECT_EQ(string_view(sc.data()), "hello");
  EXPECT_NE(data_large_ptr, sc.data());

  sc.reserve(4);
  EXPECT_EQ(string_view(sc.data()), "hello");
  EXPECT_NE(data_large_ptr, sc.data());
}

}  // namespace runtime
}  // namespace matxscript
