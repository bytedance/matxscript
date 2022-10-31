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
#include <matxscript/ir/_base/string_ref.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/runtime_value.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(RTValue, Constructor) {
  RTValue generic;
  {
    // number
    RTValue int_v(1);
    RTValue float_v(1.0);
    RTValue bool_v(false);
    int_v = 1;
    float_v = 0.2;
    bool_v = true;
    generic = 1;
    generic = 1.1;
    generic = false;
  }
  {
    // container
    RTValue list_v(List{1, 2});
    RTValue set_v(Set{1, 2});
    RTValue dict_v(Dict{{1, 1}, {"he", 2}});
    RTValue u_v(Unicode(U"jk"));
    RTValue s_v(String("jk"));
    list_v = List{3, 2};
    set_v = Set{4, 2};
    dict_v = Dict{{"kk", 10}};
    u_v = Unicode();
    s_v = String();
    generic = list_v;
    generic = set_v;
    generic = set_v;
    generic = String("sokula");
    generic = Unicode(U"sokula");
  }
}

TEST(RTValue, Cast) {
  RTValue int_v(1);
  RTValue float_v(1.0);
  RTValue bool_v(false);
  RTValue list_v(List{1, 2});
  RTValue set_v(Set{1, 2});
  RTValue dict_v(Dict{{1, 1}, {"he", 2}});
  RTValue u_v(Unicode(U"jk"));
  RTValue s_v(String("jk"));

  int64_t int_nv = int_v.As<int64_t>();
  double float_nv = float_v.As<double>();
  bool bool_nv = bool_v.As<bool>();

  List list_nv = list_v.As<List>();
  list_nv = list_v.As<List>();
  Set set_nv = set_v.As<Set>();
  set_nv = set_v.As<Set>();
  Dict dict_nv = dict_v.As<Dict>();
  dict_nv = dict_v.As<Dict>();

  Unicode unicode_nv = u_v.As<Unicode>();
  unicode_nv = u_v.As<Unicode>();
  String string_nv = s_v.As<String>();
  string_nv = s_v.As<String>();

  RTValue s_v2(StringRef("jk"));
  StringRef string_nv2 = s_v2.As<StringRef>();
  EXPECT_EQ(string_nv2.use_count(), 2);
  string_nv2 = s_v2.MoveToObjectRef<StringRef>();
  // move self
  EXPECT_EQ(string_nv2.use_count(), 1);

  RTValue s_v3(StringRef("jk"));
  StringRef string_nv3;
  string_nv3 = s_v3.MoveToObjectRef<StringRef>();
  EXPECT_EQ(string_nv3.use_count(), 1);
}

MATXSCRIPT_NO_INLINE List test_runtime_value_view(const List& u) {
  EXPECT_EQ(u.use_count(), 1);
  return u;
}

TEST(RTValue, ObjectView) {
  RTValue raw(List{U"hello"});
  auto ret = test_runtime_value_view(raw.AsObjectView<List>().data());
  EXPECT_EQ(ret, raw.AsObjectRef<List>());
}

}  // namespace runtime
}  // namespace matxscript
