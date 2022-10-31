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
#include <matxscript/runtime/container/builtins_map.h>
#include <matxscript/runtime/container/builtins_zip.h>
#include <matxscript/runtime/container/generic_zip.h>

namespace matxscript {
namespace runtime {

TEST(Builtins, zip_list) {
  List a = {"hello", "world", 123};
  List b = {"hi", "haha", 1, 2, 3};

  auto zip_cons = builtins_zip(a, b);
  for (auto iter = zip_cons.begin(); iter != zip_cons.end(); ++iter) {
    auto item = *iter;
    std::cout << std::get<0>(item) << std::endl;
    std::cout << std::get<1>(item) << std::endl;
  }

  auto g_zip_cons = generic_builtins_zip(a.iter(), b.iter());
  bool has_next = g_zip_cons.HasNext();
  while (has_next) {
    std::tuple<RTValue, RTValue> item = g_zip_cons.Next(&has_next);
    std::cout << std::get<0>(item) << std::endl;
    std::cout << std::get<1>(item) << std::endl;
  }
}

TEST(Builtins, zip_str) {
  const Unicode& a = U"a a a a";
  const Unicode& b = U"b b b b";

  auto zip_cons = builtins_zip(a, b);
  for (auto iter = zip_cons.begin(); iter != zip_cons.end(); ++iter) {
    auto item = *iter;
    std::cout << std::get<0>(item) << std::endl;
    std::cout << std::get<1>(item) << std::endl;
  }

  auto g_zip_cons = generic_builtins_zip(a.iter(), b.iter());
  bool has_next = g_zip_cons.HasNext();
  while (has_next) {
    std::tuple<RTValue, RTValue> item = g_zip_cons.Next(&has_next);
    std::cout << std::get<0>(item) << std::endl;
    std::cout << std::get<1>(item) << std::endl;
  }
}

TEST(Builtins, map_list) {
  List a = {"hello", "world"};
  List b = {"hi", "haha"};

  // TODO: fixme
  /* auto func = [](const String& a, const String& b) -> String { return a + " " + b; };

  auto map_cons = builtins_map(func, a, b);
  List ret(map_cons.begin(), map_cons.end());
  std::cout << ret << std::endl;
  EXPECT_EQ(ret, List({"hello hi", "world haha"})); */
}

TEST(Builtins, map_str) {
  const Unicode& a = U"a";
  const Unicode& b = U"b";

  // TODO: fixme
  /* auto func = [](const Unicode& a, const Unicode& b) -> Unicode { return a + U" " + b; };

  auto map_cons = builtins_map(func, a, b);
  List ret(map_cons.begin(), map_cons.end());
  std::cout << ret << std::endl;
  EXPECT_EQ(ret, List({U"a b"})); */
}

}  // namespace runtime
}  // namespace matxscript
