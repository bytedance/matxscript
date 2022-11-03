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
#include <float.h>
#include <limits.h>

#include <gtest/gtest.h>
#include <matxscript/runtime/container/ft_dict.h>
#include <matxscript/runtime/container/ft_list.h>
#include <matxscript/runtime/container/ft_set.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(KernelConstructor, bool) {
  {
    int32_t i32_max = INT32_MAX;
    bool a = Kernel_bool::make(i32_max);
    ASSERT_EQ(a, true);

    int32_t i32_min = INT32_MIN;
    int64_t b = Kernel_bool::make(i32_min);
    ASSERT_EQ(b, true);

    int64_t c = Kernel_bool::make(RTValue(i32_max));
    ASSERT_EQ(c, true);

    int32_t i32_0 = 0;
    bool d = Kernel_bool::make(i32_0);
    ASSERT_EQ(d, false);

    int32_t c_0 = Kernel_bool::make(RTValue(i32_0));
    bool e = Kernel_bool::make(c_0);
    ASSERT_EQ(e, false);
  }
  {
    List l1{1, 2, 3};
    bool a = Kernel_bool::make(l1);
    ASSERT_EQ(a, true);

    List l2{};
    bool b = Kernel_bool::make(l2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(l1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(l2));
    ASSERT_EQ(d, false);
  }
  {
    Set s1{1, 2, 3};
    bool a = Kernel_bool::make(s1);
    ASSERT_EQ(a, true);

    Set s2{};
    bool b = Kernel_bool::make(s2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(s1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(s2));
    ASSERT_EQ(d, false);
  }
  {
    Dict d1{{1, 2}, {3, 4}};
    bool a = Kernel_bool::make(d1);
    ASSERT_EQ(a, true);

    Dict d2{};
    bool b = Kernel_bool::make(d2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(d1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(d2));
    ASSERT_EQ(d, false);
  }
  {
    Tuple t1{1, 2, 3};
    bool a = Kernel_bool::make(t1);
    ASSERT_EQ(a, true);

    bool c = Kernel_bool::make(RTValue(t1));
    ASSERT_EQ(c, true);
  }
  {
    FTList<int> l1{1, 2, 3};
    bool a = Kernel_bool::make(l1);
    ASSERT_EQ(a, true);

    FTList<int> l2{};
    bool b = Kernel_bool::make(l2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(l1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(l2));
    ASSERT_EQ(d, false);
  }
  {
    FTSet<int> s1{1, 2, 3};
    bool a = Kernel_bool::make(s1);
    ASSERT_EQ(a, true);

    FTSet<int> s2{};
    bool b = Kernel_bool::make(s2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(s1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(s2));
    ASSERT_EQ(d, false);
  }
  {
    FTDict<int, int> d1{{1, 2}, {3, 4}};
    bool a = Kernel_bool::make(d1);
    ASSERT_EQ(a, true);

    FTDict<int, int> d2{};
    bool b = Kernel_bool::make(d2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(d1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(d2));
    ASSERT_EQ(d, false);
  }
  {
    String s1{"123"};
    bool a = Kernel_bool::make(s1);
    ASSERT_EQ(a, true);

    String s2{""};
    bool b = Kernel_bool::make(s2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(s1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(s2));
    ASSERT_EQ(d, false);

    bool e = Kernel_bool::make(s1.view());
    ASSERT_EQ(e, true);

    bool f = Kernel_bool::make(s2.view());
    ASSERT_EQ(f, false);

    bool g = Kernel_bool::make(RTValue(s1.view()));
    ASSERT_EQ(e, true);

    bool h = Kernel_bool::make(RTValue(s2.view()));
    ASSERT_EQ(f, false);
  }
  {
    Unicode s1(U"123");
    bool a = Kernel_bool::make(s1);
    ASSERT_EQ(a, true);

    Unicode s2(U"");
    bool b = Kernel_bool::make(s2);
    ASSERT_EQ(b, false);

    bool c = Kernel_bool::make(RTValue(s1));
    ASSERT_EQ(c, true);

    bool d = Kernel_bool::make(RTValue(s2));
    ASSERT_EQ(d, false);

    bool e = Kernel_bool::make(s1.view());
    ASSERT_EQ(e, true);

    bool f = Kernel_bool::make(s2.view());
    ASSERT_EQ(f, false);

    bool g = Kernel_bool::make(RTValue(s1.view()));
    ASSERT_EQ(e, true);

    bool h = Kernel_bool::make(RTValue(s2.view()));
    ASSERT_EQ(f, false);
  }
}
TEST(KernelConstructor, int64_t) {
  {
    int32_t i32_max = INT32_MAX;
    int64_t a = Kernel_int64_t::make(i32_max);
    ASSERT_EQ(a, INT32_MAX);

    int32_t i32_min = INT32_MIN;
    int64_t b = Kernel_int64_t::make(i32_min);
    ASSERT_EQ(b, INT32_MIN);

    int64_t c = Kernel_int64_t::make(RTValue(i32_max));
    ASSERT_EQ(c, INT32_MAX);
  }
  {
    int64_t i64_max = INT64_MAX;
    int64_t a = Kernel_int64_t::make(i64_max);
    ASSERT_EQ(a, INT64_MAX);

    int64_t i64_min = INT64_MIN;
    int64_t b = Kernel_int64_t::make(i64_min);
    ASSERT_EQ(b, INT64_MIN);

    int64_t c = Kernel_int64_t::make(RTValue(i64_max));
    ASSERT_EQ(c, INT64_MAX);
  }
  {
    float d32_max = FLT_MAX;
    int64_t a = Kernel_int64_t::make(d32_max);
    ASSERT_EQ(a, static_cast<int64_t>(d32_max));

    float d32_min = FLT_MIN;
    int64_t b = Kernel_int64_t::make(d32_min);
    ASSERT_EQ(b, static_cast<int64_t>(d32_min));

    // int64_t c = Kernel_int64_t::make(RTValue(d32_max));
    // ASSERT_EQ(c, static_cast<int64_t>(FLT_MAX));
  }
  {
    double d64_max = DBL_MAX;
    int64_t a = Kernel_int64_t::make(d64_max);
    ASSERT_EQ(a, static_cast<int64_t>(d64_max));

    double d64_min = DBL_MIN;
    int64_t b = Kernel_int64_t::make(d64_min);
    ASSERT_EQ(b, static_cast<int64_t>(d64_min));

    // int64_t c = Kernel_int64_t::make(RTValue(d64_max));
    // ASSERT_EQ(c, static_cast<int64_t>(DBL_MAX));
  }
  {
    bool positive = true;
    int64_t a = Kernel_int64_t::make(positive);
    ASSERT_EQ(a, 1);

    bool negtive = false;
    int64_t b = Kernel_int64_t::make(negtive);
    ASSERT_EQ(b, 0);

    int64_t c = Kernel_int64_t::make(RTValue(positive));
    ASSERT_EQ(c, 1);
  }
  {
    String s1("123");
    int64_t a = Kernel_int64_t::make(s1);
    ASSERT_EQ(a, 123);

    String s2("0123");
    int64_t b = Kernel_int64_t::make(s2);
    ASSERT_EQ(b, 123);

    String s3("-123");
    int64_t c = Kernel_int64_t::make(s3);
    ASSERT_EQ(c, -123);

    String s4("-0123");
    int64_t d = Kernel_int64_t::make(s4);
    ASSERT_EQ(d, -123);

    int64_t e = Kernel_int64_t::make(RTValue(s1));
    ASSERT_EQ(e, 123);
  }
  {
    Unicode u1(U"123");
    int64_t a = Kernel_int64_t::make(u1);
    ASSERT_EQ(a, 123);

    Unicode u2(U"0123");
    int64_t b = Kernel_int64_t::make(u2);
    ASSERT_EQ(b, 123);

    Unicode u3(U"-123");
    int64_t c = Kernel_int64_t::make(u3);
    ASSERT_EQ(c, -123);

    Unicode u4(U"-0123");
    int64_t d = Kernel_int64_t::make(u4);
    ASSERT_EQ(d, -123);

    int64_t e = Kernel_int64_t::make(RTValue(u1));
    ASSERT_EQ(e, 123);
  }
}

TEST(KernelConstructor, double) {
  {
    int32_t i32_max = INT32_MAX;
    double a = Kernel_double::make(i32_max);
    EXPECT_DOUBLE_EQ(a, static_cast<double>(INT32_MAX));

    int32_t i32_min = INT32_MIN;
    double b = Kernel_double::make(i32_min);
    EXPECT_DOUBLE_EQ(b, static_cast<double>(INT32_MIN));

    double c = Kernel_double::make(RTValue(i32_max));
    EXPECT_DOUBLE_EQ(c, static_cast<double>(INT32_MAX));
  }
  {
    int64_t i64_max = INT64_MAX;
    double a = Kernel_double::make(i64_max);
    EXPECT_DOUBLE_EQ(a, static_cast<double>(INT64_MAX));

    int64_t i64_min = INT64_MIN;
    double b = Kernel_double::make(i64_min);
    EXPECT_DOUBLE_EQ(b, static_cast<double>(INT64_MIN));

    double c = Kernel_double::make(RTValue(i64_max));
    EXPECT_DOUBLE_EQ(c, static_cast<double>(INT64_MAX));
  }
  {
    float d32_max = FLT_MAX;
    double a = Kernel_double::make(d32_max);
    EXPECT_DOUBLE_EQ(a, FLT_MAX);

    float d32_min = FLT_MIN;
    double b = Kernel_double::make(d32_min);
    EXPECT_DOUBLE_EQ(b, FLT_MIN);

    double c = Kernel_double::make(RTValue(d32_max));
    EXPECT_DOUBLE_EQ(c, FLT_MAX);
  }
  {
    double d64_max = DBL_MAX;
    double a = Kernel_double::make(d64_max);
    EXPECT_DOUBLE_EQ(a, DBL_MAX);

    double d64_min = DBL_MIN;
    double b = Kernel_double::make(d64_min);
    EXPECT_DOUBLE_EQ(b, DBL_MIN);

    double c = Kernel_double::make(RTValue(d64_max));
    EXPECT_DOUBLE_EQ(c, DBL_MAX);
  }
  {
    bool positive = true;
    double a = Kernel_double::make(positive);
    EXPECT_DOUBLE_EQ(a, 1.0);

    bool negtive = false;
    double b = Kernel_double::make(negtive);
    EXPECT_DOUBLE_EQ(b, 0.0);

    double c = Kernel_double::make(RTValue(positive));
    EXPECT_DOUBLE_EQ(c, 1.0);
  }
  {
    String s1(".2");
    double a = Kernel_double::make(s1);
    EXPECT_DOUBLE_EQ(a, 0.2);

    String s2(".0");
    double b = Kernel_double::make(s2);
    EXPECT_DOUBLE_EQ(b, 0.0);

    String s3("2.0");
    double c = Kernel_double::make(s3);
    EXPECT_DOUBLE_EQ(c, 2.0);

    String s4("2.5");
    double d = Kernel_double::make(s4);
    EXPECT_DOUBLE_EQ(d, 2.5);

    String s5("02.5");
    double e = Kernel_double::make(s5);
    EXPECT_DOUBLE_EQ(e, 2.5);

    String s6("2");
    double f = Kernel_double::make(s6);
    EXPECT_DOUBLE_EQ(f, 2.0);

    String s7("-2.5");
    double g = Kernel_double::make(s7);
    EXPECT_DOUBLE_EQ(g, -2.5);

    String s8("-2");
    double h = Kernel_double::make(s8);
    EXPECT_DOUBLE_EQ(h, -2.0);

    double i = Kernel_double::make(RTValue(s1));
    EXPECT_DOUBLE_EQ(i, 0.2);
  }
  {
    Unicode u1(U".2");
    double a = Kernel_double::make(u1);
    EXPECT_DOUBLE_EQ(a, 0.2);

    Unicode u2(U".0");
    double b = Kernel_double::make(u2);
    EXPECT_DOUBLE_EQ(b, 0.0);

    Unicode u3(U"2.0");
    double c = Kernel_double::make(u3);
    EXPECT_DOUBLE_EQ(c, 2.0);

    Unicode u4(U"2.5");
    double d = Kernel_double::make(u4);
    EXPECT_DOUBLE_EQ(d, 2.5);

    Unicode u5(U"02.5");
    double e = Kernel_double::make(u5);
    EXPECT_DOUBLE_EQ(e, 2.5);

    Unicode u6(U"2");
    double f = Kernel_double::make(u6);
    EXPECT_DOUBLE_EQ(f, 2.0);

    Unicode u7(U"-2.5");
    double g = Kernel_double::make(u7);
    EXPECT_DOUBLE_EQ(g, -2.5);

    Unicode u8(U"-2");
    double h = Kernel_double::make(u8);
    EXPECT_DOUBLE_EQ(h, -2.0);

    double i = Kernel_double::make(RTValue(u1));
    EXPECT_DOUBLE_EQ(i, 0.2);
  }
}

TEST(KernelConstructor, List) {
  {
    List a = Kernel_List::make();
    ASSERT_TRUE(a.empty());
  }
  {
    Set s{1, 2, 3};
    List a = Kernel_List::make(s);
    ASSERT_TRUE(a.contains(1));
    ASSERT_TRUE(a.contains(2));
    ASSERT_TRUE(a.contains(3));
  }
  {
    Set s{1, 2, 3};
    List a = Kernel_List::make(RTValue(s));
    ASSERT_TRUE(a.contains(1));
    ASSERT_TRUE(a.contains(2));
    ASSERT_TRUE(a.contains(3));
  }
  {
    List s{1, 2, 3};
    List a = Kernel_List::make(s);
    ASSERT_EQ(a, s);
  }
  {
    List s{1, 2, 3};
    List a = Kernel_List::make(RTValue(s));
    ASSERT_EQ(a, s);
  }
}

TEST(KernelConstructor, Dict) {
  Dict a{{"hello", "world"}};
  std::vector<Dict::value_type> b{{"hello", "world"}};
  std::cout << std::distance(a.item_begin(), a.item_end()) << std::endl;
  auto new_dict = Kernel_Dict::make(a);
  std::cout << new_dict << std::endl;
  EXPECT_EQ(a, new_dict);
}

TEST(KernelConstructor, Set) {
  {
    Set a = Kernel_Set::make();
    ASSERT_TRUE(a.empty());
  }
  {
    List l{1, 2, 3};
    Set a = Kernel_Set::make(l);
    ASSERT_EQ(a, Set({1, 2, 3}));
  }
  {
    List l{1, 2, 3};
    Set a = Kernel_Set::make(RTValue(l));
    ASSERT_EQ(a, Set({1, 2, 3}));
  }
  {
    Set s{1, 2, 3};
    Set a = Kernel_Set::make(s);
    ASSERT_EQ(a, s);
  }
  {
    Set s{1, 2, 3};
    Set a = Kernel_Set::make(RTValue(s));
    ASSERT_EQ(a, s);
  }
}

TEST(KernelConstructor, Unicode) {
  {
    Unicode a = Kernel_Unicode::make();
    ASSERT_TRUE(a.empty());
  }
  {
    Unicode a = Kernel_Unicode::make(10);
    ASSERT_EQ(a, Unicode(U"10"));
    a = Kernel_Unicode::make(RTValue(10));
    ASSERT_EQ(a, Unicode(U"10"));
  }
  {
    Unicode a = Kernel_Unicode::make(int64_t(10));
    ASSERT_EQ(a, Unicode(U"10"));
    a = Kernel_Unicode::make(RTValue(int64_t(10)));
    ASSERT_EQ(a, Unicode(U"10"));
  }
  {
    // TODO (mxd): fix me
    // Unicode a = Kernel_Unicode::make(10.0);
    // ASSERT_EQ(a, Unicode(U"10.0"));
    Unicode a = Kernel_Unicode::make(10.1);
    ASSERT_EQ(a, Unicode(U"10.1"));
    a = Kernel_Unicode::make(RTValue(10.1));
    ASSERT_EQ(a, Unicode(U"10.1"));
  }
  {
    Unicode a = Kernel_Unicode::make(double(10.1));
    ASSERT_EQ(a, Unicode(U"10.1"));
    a = Kernel_Unicode::make(RTValue(double(10.1)));
    ASSERT_EQ(a, Unicode(U"10.1"));
  }
  {
    Unicode a = Kernel_Unicode::make(float(10.1));
    ASSERT_EQ(a, Unicode(U"10.1"));
    // TODO (mxd) : fix me
    // a = Kernel_Unicode::make(RTValue(float(10.1)));
    // ASSERT_EQ(a, Unicode(U"10.1"));
  }
  {
    Unicode a = Kernel_Unicode::make(Unicode(U"10"));
    ASSERT_EQ(a, Unicode(U"10"));
    a = Kernel_Unicode::make(RTValue(Unicode(U"10")));
    ASSERT_EQ(a, Unicode(U"10"));
  }
}

TEST(KernelConstructor, String) {
  // TODO (liqingshuo)
}

TEST(KernelConstructor, Iterator) {
  // TODO (mxd)
}

TEST(KernelConstructor, NDArray) {
  {
    List data{1, 2, 3, 4, 5.0, 6};
    List shape{2, 3};
    Unicode dtype(U"float64");
    NDArray arr = Kernel_NDArray::make(data, shape, dtype);
    auto shape2 = arr.Shape();
    EXPECT_EQ(shape2.size(), 2);
    EXPECT_EQ(shape2[0], 2);
    EXPECT_EQ(shape2[1], 3);
    double* p = static_cast<double*>(arr->data);
    std::cout << "ndarray contents: ";
    for (int i = 0; i < 6; ++i) {
      std::cout << p[i] << " ";
    }
    std::cout << std::endl;
    ASSERT_NEAR(p[4], 4.99999, 1e-5);
  }
}

TEST(KernelConstructor, Regex) {
  Regex pat;
  pat = Kernel_Regex::make(Unicode(U"([ ]+)"), false, false, false, false, false);
  std::cout << pat.split(RTValue("this is a test")) << std::endl;
}

}  // namespace runtime
}  // namespace matxscript
