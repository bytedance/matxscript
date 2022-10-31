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

namespace matxscript {
namespace runtime {

TEST(FTDict, Constructor) {
  FTDict<String, String> a{{"hello", "world"}};
  std::vector<FTDict<String, String>::value_type> b{{"hello", "world"}};
  auto new_dict2 = FTDict<String, String>(a.item_begin(), a.item_end());
  std::cout << new_dict2 << std::endl;
  EXPECT_EQ(a, new_dict2);
  auto new_dict3 = FTDict<String, String>(b.begin(), b.end());
  std::cout << new_dict3 << std::endl;
  EXPECT_EQ(a, new_dict3);
  auto new_dict4 = FTDict<String, String>(b);
  std::cout << new_dict4 << std::endl;
  EXPECT_EQ(a, new_dict4);
  FTDict<String, int> c;
  std::cout << c << std::endl;
}

TEST(FTDict, RTValue) {
  FTDict<String, String> ft_cons1{{"hello", "world"}};
  RTValue any_v1 = ft_cons1;
  EXPECT_EQ(any_v1, ft_cons1);
  RTValue any_v2(ft_cons1);
  std::cout << any_v2 << std::endl;
  EXPECT_TRUE(any_v2.IsObjectRef<FTObjectBase>());
  FTDict<String, String> ft_cons2(any_v2.As<FTDict<String, String>>());
  EXPECT_EQ(any_v2, ft_cons2);
  FTDict<String, String> ft_cons3 = any_v2.As<FTDict<String, String>>();
  EXPECT_EQ(any_v2, ft_cons3);
  EXPECT_ANY_THROW((any_v2.As<FTDict<String, Unicode>>()));
  FTDict<RTValue, RTValue> generic;
}

TEST(FTDict, iterator) {
  FTDict<int, String> d1{{1, String("hello")}, {2, String("matx4")}};
  FTDict<int, String>::const_iterator iter_begin = d1.begin();
  FTDict<int, String>::const_iterator iter_end = d1.end();
  EXPECT_EQ(2, std::distance(iter_begin, iter_end));
}

TEST(FTDict, len) {
  FTDict<int, String> d1{{1, String("hello")}, {2, String("matx4")}};
  ASSERT_EQ(d1.size(), 2);
  FTDict<String, int> c;
  ASSERT_EQ(c.size(), 0);
}

TEST(FTDict, equal) {
  FTDict<int, String> d0;  // empty dict
  FTDict<String, int> b0;
  ASSERT_EQ(d0, b0);                                                   // is it right?
  FTDict<int, String> d1{{1, String("hello")}, {2, String("matx4")}};  // same type
  FTDict<int, String> d2{{1, String("hello")}, {2, String("matx4")}};
  ASSERT_NE(d0, d1);
  ASSERT_EQ(d1, d1);
  ASSERT_EQ(d1, d2);
  FTDict<int, String> d3{{1, String("hello")}, {3, String("matx4")}};
  ASSERT_NE(d1, d3);
  FTDict<String, int> d4{{String("hello"), 1}, {String("matx4"), 2}};  // different type
  ASSERT_NE(d1, d4);
  // TODO: fixme
  // FTDict<String, int> d5{{RTValue("hello"), 1}, {RTValue("matx4"), 2}};  // RTValue
  // ASSERT_NE(d1, d5);
  // FTDict<int, String> d6{{RTValue(1), String("hello")}, {RTValue(2), String("matx4")}};
  // ASSERT_EQ(d1, d6);
}

TEST(FTDict, contains) {
  FTDict<int, String> d0;  // empty dict
  ASSERT_FALSE(d0.contains(1));
  ASSERT_FALSE(d0.contains("uu"));
  FTDict<int, String> d1{{1, String("hello")}, {2, String("matx4")}};
  ASSERT_TRUE(d1.contains(1));  // same type
  ASSERT_FALSE(d1.contains(42));
  ASSERT_FALSE(d1.contains(1.3));  // different type
  ASSERT_FALSE(d1.contains("a"));
  ASSERT_TRUE(d1.contains(true));
  ASSERT_FALSE(d1.contains(string_view("a")));
  ASSERT_FALSE(d1.contains(String("a")));
  ASSERT_FALSE(d1.contains(Unicode(U"a")));
  ASSERT_FALSE(d1.contains(RTValue("a")));  // RTValue
  ASSERT_FALSE(d1.contains(RTValue(U"a")));
  ASSERT_TRUE(d1.contains(RTValue(1)));

  FTDict<String, int> d2{{String("hello"), 1}, {String("matx4"), 1}};
  ASSERT_TRUE(d2.contains(String("hello")));  // same type
  ASSERT_FALSE(d2.contains(String("world")));
  ASSERT_FALSE(d2.contains(1.3));  // different type
  ASSERT_FALSE(d2.contains(1));
  ASSERT_FALSE(d2.contains(true));
  ASSERT_TRUE(d2.contains(string_view("hello")));
  ASSERT_FALSE(d2.contains(unicode_view(U"a")));
  ASSERT_FALSE(d2.contains(Unicode(U"hello")));
  ASSERT_TRUE(d2.contains(RTValue("hello")));  // RTValue
  ASSERT_FALSE(d2.contains(RTValue(U"hello")));
  ASSERT_FALSE(d2.contains(RTValue(1)));
}

TEST(FTDict, get_item) {
  FTDict<int, String> d0;  // empty dict
  FTDict<int, String> d{{1, String("hello")}, {2, String("matx4")}};
  ASSERT_EQ(d[1], "hello");           // same type
  ASSERT_EQ(d[RTValue(1)], "hello");  // RTValue
  ASSERT_ANY_THROW(d[RTValue("a")]);
  ASSERT_ANY_THROW(d[RTValue(U"a")]);

  ASSERT_EQ(d.get_item(1), "hello");  // same type
  ASSERT_ANY_THROW(d.get_item(3));
  ASSERT_ANY_THROW(d.get_item("a"));  // different type
  ASSERT_ANY_THROW(d.get_item(1.3));
  ASSERT_ANY_THROW(d.get_item(String("a")));
  ASSERT_ANY_THROW(d.get_item(string_view("a")));
  ASSERT_ANY_THROW(d.get_item(Unicode(U"a")));
  ASSERT_ANY_THROW(d.get_item(unicode_view(U"a")));
  ASSERT_EQ(d.get_item(RTValue(1)), "hello");  // RTValue
  ASSERT_ANY_THROW(d.get_item(RTValue("a")));
  ASSERT_ANY_THROW(d.get_item(RTValue(U"a")));
}

TEST(FTDict, get_default) {
  FTDict<int, String> d0;  // empty dict
  EXPECT_EQ(d0.get_default(1), None);
  EXPECT_EQ(d0.get_default(1, "aa"), "aa");
  EXPECT_EQ(d0.get_default(1, U"aa"), U"aa");
  FTDict<int, String> d1{{1, "aa"}, {2, "bb"}};  // same type
  EXPECT_EQ(d1.get_default(1), "aa");
  EXPECT_EQ(d1.get_default(3), None);
  EXPECT_EQ(d1.get_default(1, "cc"), "aa");
  EXPECT_EQ(d1.get_default(1, U"cc"), "aa");
  EXPECT_EQ(d1.get_default(3, "cc"), "cc");
  EXPECT_EQ(d1.get_default(3, U"cc"), U"cc");
  EXPECT_EQ(d1.get_default("1"), None);  // different type
  EXPECT_EQ(d1.get_default("1", "dd"), "dd");
  EXPECT_EQ(d1.get_default("1", U"dd"), U"dd");
  EXPECT_EQ(d1.get_default(RTValue(1)), "aa");  // RTValue
  EXPECT_EQ(d1.get_default(RTValue(1), "cc"), "aa");
  EXPECT_EQ(d1.get_default(RTValue(3), "cc"), "cc");
  EXPECT_EQ(d1.get_default(RTValue("1")), None);
  EXPECT_EQ(d1.get_default(RTValue("1"), "cc"), "cc");
}

TEST(FTDict, pop) {
  FTDict<int, String> d0;  // empty dict
  ASSERT_ANY_THROW(d0.pop(1));
  EXPECT_EQ(d0.pop(1, "aa"), "aa");
  EXPECT_EQ(d0.pop(1, U"aa"), U"aa");
  FTDict<int, String> d1{{0, "hh"}, {1, "aa"}, {2, "bb"}, {4, "kk"}};  // same type
  EXPECT_EQ(d1.pop(1), "aa");
  ASSERT_ANY_THROW(d1.pop(3));
  EXPECT_EQ(d1.pop(0, "cc"), "hh");
  EXPECT_EQ(d1.pop(0, U"cc"), U"cc");
  EXPECT_EQ(d1.pop(3, "cc"), "cc");
  EXPECT_EQ(d1.pop(3, U"cc"), U"cc");
  ASSERT_ANY_THROW(d1.pop("1"));  // different type
  EXPECT_EQ(d1.pop("1", "dd"), "dd");
  EXPECT_EQ(d1.pop("1", U"dd"), U"dd");
  EXPECT_EQ(d1.pop(RTValue(4)), "kk");  // RTValue
  EXPECT_EQ(d1.pop(RTValue(4), "cc"), "cc");
  EXPECT_EQ(d1.pop(RTValue(3), "cc"), "cc");
  ASSERT_ANY_THROW(d1.pop(RTValue("1")));
  EXPECT_EQ(d1.pop(RTValue("1"), "cc"), "cc");
}

TEST(FTDict, set_item) {
  FTDict<int, String> d0;  // empty dict
  d0[0] = "hello";
  ASSERT_EQ(d0[0], "hello");
  FTDict<int, Unicode> d{{1, Unicode(U"hello")}, {2, Unicode(U"matx4")}};
  d[2] = U"3";  // same type
  ASSERT_EQ(d[2], U"3");
  d[3] = Unicode(U"world");
  ASSERT_EQ(d[3], Unicode(U"world"));
  d.set_item(4, U"world");
  ASSERT_EQ(d[4], Unicode(U"world"));
  d[RTValue(5)] = U"5";  // RTValue
  ASSERT_EQ(d[5], U"5");
  // d[6] = RTValue(U"aa");
  // ASSERT_EQ(d[6], U"aa");
  ASSERT_ANY_THROW(d[RTValue(U"a")] = U"aa");
  // ASSERT_ANY_THROW(d[RTValue("a")] = "aa");
  // ASSERT_ANY_THROW(d[7] = RTValue("aa"));
}

TEST(FTDict, clear) {
  FTDict<int, Unicode> d{{1, Unicode(U"hello")}, {2, Unicode(U"matx4")}};
  FTDict<int, Unicode> d0;
  d.clear();
  ASSERT_EQ(d.size(), 0);
  ASSERT_EQ(d, d0);
}

}  // namespace runtime
}  // namespace matxscript