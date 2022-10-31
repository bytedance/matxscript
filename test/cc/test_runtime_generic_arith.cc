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
#include <matxscript/runtime/generic/generic_hlo_arith_funcs.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(RTValueOperators, builtin_add) {
  {
    double a = 0.1;
    double b = 0.2;
    double r = 0.3;
    ASSERT_DOUBLE_EQ(a + b, r);
    ASSERT_FLOAT_EQ(ArithOps::add(RTValue(a), b), r);
    ASSERT_FLOAT_EQ(ArithOps::add(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), RTValue(b)), RTValue(r));
  }
  {
    float a = 0.1;
    float b = 0.2;
    float r = 0.3;
    ASSERT_FLOAT_EQ(a + b, r);
    ASSERT_FLOAT_EQ(ArithOps::add(RTValue(a), b), r);
    ASSERT_FLOAT_EQ(ArithOps::add(a, RTValue(b)), r);
    ASSERT_FLOAT_EQ(ArithOps::add(RTValue(a), RTValue(b)).As<float>(), RTValue(r).As<float>());
  }
  {
    int64_t a = 1;
    int64_t b = 2;
    RTValue r = 3;
    ASSERT_EQ(ArithOps::add(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::add(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), RTValue(b)), r);
  }
  {
    int32_t a = 1;
    int32_t b = 2;
    RTValue r = 3;
    ASSERT_EQ(ArithOps::add(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::add(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), RTValue(b)), r);
  }
  {
    Unicode a(U"HELLO");
    Unicode b(U"WORLD");
    Unicode r(U"HELLOWORLD");
    ASSERT_EQ(ArithOps::add(a, b), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::add(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), RTValue(b)), RTValue(r));
  }
  {
    String a("HELLO");
    String b("WORLD");
    String r("HELLOWORLD");
    ASSERT_EQ(ArithOps::add(a, b), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::add(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), RTValue(b)), RTValue(r));
  }
  {
    List a({1});
    List b({2});
    List r({1, 2});
    ASSERT_EQ(ArithOps::add(a, b), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::add(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), RTValue(b)), RTValue(r));
  }
  {
    FTList<int64_t> a({1});
    FTList<int64_t> b({2});
    FTList<int64_t> r({1, 2});
    ASSERT_EQ(ArithOps::add(a, b), r);
    ASSERT_EQ(ArithOps::add(RTView(a), RTView(b)), r);
    ASSERT_EQ(ArithOps::add(RTValue(a), RTValue(b)), r);
    ASSERT_EQ(ArithOps::add(RTView(a), RTView(b)), RTValue(r));
  }
}

TEST(RTValueOperators, builtin_sub) {
  {
    double a = 0.3;
    double b = 0.2;
    double r = 0.1;
    ASSERT_DOUBLE_EQ(a - b, r);
    ASSERT_FLOAT_EQ(ArithOps::sub(RTValue(a), b), r);
    ASSERT_FLOAT_EQ(ArithOps::sub(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::sub(RTValue(a), RTValue(b)), RTValue(r));
  }
  {
    float a = 0.3;
    float b = 0.2;
    float r = 0.1;
    ASSERT_FLOAT_EQ(a - b, r);
    ASSERT_FLOAT_EQ(ArithOps::sub(RTValue(a), b), r);
    ASSERT_FLOAT_EQ(ArithOps::sub(a, RTValue(b)), r);
    ASSERT_FLOAT_EQ(ArithOps::sub(RTValue(a), RTValue(b)).As<float>(), RTValue(r).As<float>());
  }
  {
    int64_t a = 3;
    int64_t b = 2;
    RTValue r = 1;
    ASSERT_EQ(ArithOps::sub(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::sub(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::sub(RTValue(a), RTValue(b)), r);
  }
  {
    int32_t a = 3;
    int32_t b = 2;
    RTValue r = 1;
    ASSERT_EQ(ArithOps::sub(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::sub(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::sub(RTValue(a), RTValue(b)), r);
  }
}

TEST(RTValueOperators, builtin_mul) {
  {
    double a = 1.0;
    double b = 1.2;
    double r = 1.2;
    ASSERT_DOUBLE_EQ(a * b, r);
    ASSERT_FLOAT_EQ(ArithOps::mul(RTValue(a), b), r);
    ASSERT_FLOAT_EQ(ArithOps::mul(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), RTValue(b)), RTValue(r));
  }
  {
    float a = 1.0;
    float b = 1.2;
    float r = 1.2;
    ASSERT_FLOAT_EQ(a * b, r);
    ASSERT_FLOAT_EQ(ArithOps::mul(RTValue(a), b), r);
    ASSERT_FLOAT_EQ(ArithOps::mul(a, RTValue(b)), r);
    ASSERT_FLOAT_EQ(ArithOps::mul(RTValue(a), RTValue(b)).As<float>(), RTValue(r).As<float>());
  }
  {
    int64_t a = 1;
    int64_t b = 2;
    RTValue r = 2;
    ASSERT_EQ(ArithOps::mul(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::mul(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), RTValue(b)), r);
  }
  {
    int32_t a = 1;
    int32_t b = 2;
    RTValue r = 2;
    ASSERT_EQ(ArithOps::mul(RTValue(a), b), r);
    ASSERT_EQ(ArithOps::mul(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), RTValue(b)), r);
  }
  {
    Unicode a(U"HELLO");
    int64_t b = 2;
    Unicode r(U"HELLOHELLO");
    ASSERT_EQ(ArithOps::mul(a, b), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), b), RTValue(r));
    ASSERT_EQ(ArithOps::mul(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), RTValue(b)), RTValue(r));
    ASSERT_EQ(ArithOps::mul(RTValue(b), RTValue(a)), RTValue(r));
  }
  {
    String a("HELLO");
    int64_t b = 2;
    String r("HELLOHELLO");
    ASSERT_EQ(ArithOps::mul(a, b), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), b), RTValue(r));
    ASSERT_EQ(ArithOps::mul(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), RTValue(b)), RTValue(r));
    ASSERT_EQ(ArithOps::mul(RTValue(b), RTValue(a)), RTValue(r));
  }
  {
    List a({1});
    int64_t b = 2;
    List r({1, 1});
    ASSERT_EQ(ArithOps::mul(a, b), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), b), RTValue(r));
    ASSERT_EQ(ArithOps::mul(a, RTValue(b)), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), RTValue(b)), RTValue(r));
    ASSERT_EQ(ArithOps::mul(RTValue(b), RTValue(a)), RTValue(r));
  }
  {
    FTList<int64_t> a({1});
    int64_t b = 2;
    FTList<int64_t> r({1, 1});
    ASSERT_EQ(ArithOps::mul(a, b), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), b), RTValue(r));
    ASSERT_EQ(ArithOps::mul(RTView(a), RTValue(b)), r);
    ASSERT_EQ(ArithOps::mul(RTValue(a), RTValue(b)), RTValue(r));
    ASSERT_EQ(ArithOps::mul(RTValue(b), RTValue(a)), RTValue(r));
  }
}

TEST(RTValueOperators, builtin_eq) {
  {
    double a = 1.0;
    double b = 1.0;
    ASSERT_DOUBLE_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), b));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), a));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    float a = 1.0;
    float b = 1.0;
    ASSERT_FLOAT_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), b));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), a));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    int64_t a = 1;
    int64_t b = 1;
    ASSERT_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), b));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), a));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    int32_t a = 1;
    int32_t b = 1;
    ASSERT_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), b));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), a));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"HELLO");
    Unicode b(U"HELLO");
    ASSERT_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), b));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), a));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    String a("HELLO");
    String b("HELLO");
    ASSERT_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), b));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), a));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    List b({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), b));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), a));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    FTList<RTValue> a({1, Unicode(U"HELLO"), String("HELLO")});
    FTList<RTValue> b({1, Unicode(U"HELLO"), String("HELLO")});
    ASSERT_EQ(a, b);
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTView(b)));
    ASSERT_TRUE(ArithOps::eq(RTValue(b), RTView(a)));
    ASSERT_TRUE(ArithOps::eq(RTValue(a), RTValue(b)));
  }
  {
    ASSERT_EQ(RTValue(), RTValue());
    ASSERT_TRUE(ArithOps::eq(RTValue(), RTValue()));
  }
}

TEST(RTValueOperators, builtin_ne) {
  {
    double a = 1.0;
    double b = 1.1;
    ASSERT_NE(a, b);
    ASSERT_TRUE(ArithOps::ne(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ne(RTValue(b), a));
    ASSERT_TRUE(ArithOps::ne(RTValue(a), RTValue(b)));
  }
  {
    float a = 1.0;
    float b = 1.1;
    ASSERT_NE(a, b);
    ASSERT_TRUE(ArithOps::ne(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ne(RTValue(b), a));
    ASSERT_TRUE(ArithOps::ne(RTValue(a), RTValue(b)));
  }
  {
    int64_t a = 1;
    int64_t b = 2;
    ASSERT_NE(a, b);
    ASSERT_TRUE(ArithOps::ne(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ne(RTValue(b), a));
    ASSERT_TRUE(ArithOps::ne(RTValue(a), RTValue(b)));
  }
  {
    int32_t a = 1;
    int32_t b = 2;
    ASSERT_NE(a, b);
    ASSERT_TRUE(ArithOps::ne(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ne(RTValue(b), a));
    ASSERT_TRUE(ArithOps::ne(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"HELLO");
    Unicode b(U"WORLD");
    ASSERT_NE(a, b);
    ASSERT_TRUE(ArithOps::ne(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ne(RTValue(b), a));
    ASSERT_TRUE(ArithOps::ne(RTValue(a), RTValue(b)));
  }
  {
    String a("HELLO");
    String b("WORLD");
    ASSERT_NE(a, b);
    ASSERT_TRUE(ArithOps::ne(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ne(RTValue(b), a));
    ASSERT_TRUE(ArithOps::ne(RTValue(a), RTValue(b)));
  }
  {
    List a({1, Unicode(U"HELLO"), String("HELLO")});
    List b({1, Unicode(U"WORLD"), String("WORLD")});
    ASSERT_NE(a, b);
    ASSERT_TRUE(ArithOps::ne(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ne(RTValue(b), a));
    ASSERT_TRUE(ArithOps::ne(RTValue(a), RTValue(b)));
  }
  {
    ASSERT_TRUE(ArithOps::ne(RTValue(), 1));
    ASSERT_TRUE(ArithOps::ne(RTValue(), 1.1));
    ASSERT_TRUE(ArithOps::ne(RTValue(), List{}));
    ASSERT_TRUE(ArithOps::ne(RTValue(), RTValue(1)));
    ASSERT_TRUE(ArithOps::ne(RTValue(1), RTValue()));
  }
}

TEST(RTValueOperators, builtin_gt) {
  {
    double a = 1.1;
    double b = 1.0;
    ASSERT_GT(a, b);
    ASSERT_TRUE(ArithOps::gt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::gt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::gt(RTValue(a), RTValue(b)));
  }
  {
    float a = 1.1;
    float b = 1.0;
    ASSERT_GT(a, b);
    ASSERT_TRUE(ArithOps::gt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::gt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::gt(RTValue(a), RTValue(b)));
  }
  {
    int64_t a = 2;
    int64_t b = 1;
    ASSERT_GT(a, b);
    ASSERT_TRUE(ArithOps::gt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::gt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::gt(RTValue(a), RTValue(b)));
  }
  {
    int32_t a = 2;
    int32_t b = 1;
    ASSERT_GT(a, b);
    ASSERT_TRUE(ArithOps::gt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::gt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::gt(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"WORLD");
    Unicode b(U"HELLO");
    ASSERT_GT(a, b);
    ASSERT_TRUE(ArithOps::gt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::gt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::gt(RTValue(a), RTValue(b)));
  }
  {
    String a("WORLD");
    String b("HELLO");
    ASSERT_GT(a, b);
    ASSERT_TRUE(ArithOps::gt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::gt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::gt(RTValue(a), RTValue(b)));
  }
}

TEST(RTValueOperators, builtin_ge) {
  {
    double a = 1.1;
    double b = 1.0;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
    b = 1.1;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
  {
    float a = 1.1;
    float b = 1.0;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
    b = 1.1;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
  {
    int64_t a = 2;
    int64_t b = 1;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
    b = 2;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
  {
    int32_t a = 2;
    int32_t b = 1;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
    b = 2;
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"WORLD");
    Unicode b(U"HELLO");
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"WORLD");
    Unicode b(U"WORLD");
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
  {
    String a("WORLD");
    String b("HELLO");
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
  {
    String a("WORLD");
    String b("WORLD");
    ASSERT_GE(a, b);
    ASSERT_TRUE(ArithOps::ge(RTValue(a), b));
    ASSERT_TRUE(ArithOps::ge(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::ge(RTValue(a), RTValue(b)));
  }
}

TEST(RTValueOperators, builtin_lt) {
  {
    double a = 1.0;
    double b = 1.1;
    ASSERT_LT(a, b);
    ASSERT_TRUE(ArithOps::lt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::lt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::lt(RTValue(a), RTValue(b)));
  }
  {
    float a = 1.0;
    float b = 1.1;
    ASSERT_LT(a, b);
    ASSERT_TRUE(ArithOps::lt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::lt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::lt(RTValue(a), RTValue(b)));
  }
  {
    int64_t a = 1;
    int64_t b = 2;
    ASSERT_LT(a, b);
    ASSERT_TRUE(ArithOps::lt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::lt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::lt(RTValue(a), RTValue(b)));
  }
  {
    int32_t a = 1;
    int32_t b = 2;
    ASSERT_LT(a, b);
    ASSERT_TRUE(ArithOps::lt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::lt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::lt(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"HELLO");
    Unicode b(U"WORLD");
    ASSERT_LT(a, b);
    ASSERT_TRUE(ArithOps::lt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::lt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::lt(RTValue(a), RTValue(b)));
  }
  {
    String a("HELLO");
    String b("WORLD");
    ASSERT_LT(a, b);
    ASSERT_TRUE(ArithOps::lt(RTValue(a), b));
    ASSERT_TRUE(ArithOps::lt(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::lt(RTValue(a), RTValue(b)));
  }
}

TEST(RTValueOperators, builtin_le) {
  {
    double a = 1.0;
    double b = 1.1;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
    b = 1.0;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
  {
    float a = 1.0;
    float b = 1.1;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
    b = 1.0;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
  {
    int64_t a = 1;
    int64_t b = 2;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
    b = 1;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
  {
    int32_t a = 1;
    int32_t b = 2;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
    b = 1;
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"HELLO");
    Unicode b(U"WORLD");
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
  {
    Unicode a(U"WORLD");
    Unicode b(U"WORLD");
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
  {
    String a("HELLO");
    String b("WORLD");
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
  {
    String a("WORLD");
    String b("WORLD");
    ASSERT_LE(a, b);
    ASSERT_TRUE(ArithOps::le(RTValue(a), b));
    ASSERT_TRUE(ArithOps::le(a, RTValue(b)));
    ASSERT_TRUE(ArithOps::le(RTValue(a), RTValue(b)));
  }
}

TEST(ArithOps, PythonicAnd) {
  Unicode u_a(U"hello");
  Unicode u_b(U"world");
  String s_a("hello");
  String s_b("world");
  // Any and other
  {
    RTValue r1 = ArithOps::And(RTValue(u_a), u_b);
    EXPECT_EQ(r1, RTValue(u_b));
    RTValue r2 = ArithOps::And(u_a, RTValue(u_b));
    EXPECT_EQ(r2, RTValue(u_b));
    RTValue r3 = ArithOps::And(RTValue(u_a), RTValue(u_b));
    EXPECT_EQ(r3, RTValue(u_b));
    RTValue r4 = ArithOps::And(RTView(u_a), u_b);
    EXPECT_EQ(r4, RTValue(u_b));
    RTValue r5 = ArithOps::And(u_a, RTView(u_b));
    EXPECT_EQ(r5, RTValue(u_b));
    RTValue r6 = ArithOps::And(RTView(u_a), RTView(u_b));
    EXPECT_EQ(r6, RTValue(u_b));
  }
  {
    RTValue r1 = ArithOps::And(RTValue(u_a), u_b.view());
    EXPECT_EQ(r1, RTValue(u_b));
    RTValue r2 = ArithOps::And(u_a.view(), RTValue(u_b));
    EXPECT_EQ(r2, RTValue(u_b));
    RTValue r3 = ArithOps::And(RTView(u_a), u_b.view());
    EXPECT_EQ(r3, RTValue(u_b));
    RTValue r4 = ArithOps::And(u_a.view(), RTView(u_b));
    EXPECT_EQ(r4, RTValue(u_b));
  }
  {
    RTValue r1 = ArithOps::And(RTValue(s_a), s_b);
    EXPECT_EQ(r1, RTValue(s_b));
    RTValue r2 = ArithOps::And(s_a, RTValue(s_b));
    EXPECT_EQ(r2, RTValue(s_b));
    RTValue r3 = ArithOps::And(RTValue(s_a), RTValue(s_b));
    EXPECT_EQ(r3, RTValue(s_b));
    RTValue r4 = ArithOps::And(RTView(s_a), s_b);
    EXPECT_EQ(r4, RTValue(s_b));
    RTValue r5 = ArithOps::And(s_a, RTView(s_b));
    EXPECT_EQ(r5, RTValue(s_b));
    RTValue r6 = ArithOps::And(RTView(s_a), RTView(s_b));
    EXPECT_EQ(r6, RTValue(s_b));
  }
  {
    RTValue r1 = ArithOps::And(RTValue(s_a), s_b.view());
    EXPECT_EQ(r1, RTValue(s_b));
    RTValue r2 = ArithOps::And(s_a.view(), RTValue(s_b));
    EXPECT_EQ(r2, RTValue(s_b));
    RTValue r3 = ArithOps::And(RTView(s_a), s_b.view());
    EXPECT_EQ(r3, RTValue(s_b));
    RTValue r4 = ArithOps::And(s_a.view(), RTView(s_b));
    EXPECT_EQ(r4, RTValue(s_b));
  }
  // unicode
  {
    Unicode r1 = ArithOps::And(u_a, u_b);
    EXPECT_EQ(r1, u_b);
    Unicode r2 = ArithOps::And(u_a.view(), u_b);
    EXPECT_EQ(r2, u_b);
    Unicode r3 = ArithOps::And(u_a, u_b.view());
    EXPECT_EQ(r3, u_b);
    Unicode r4 = ArithOps::And(u_a.view(), u_b.view());
    EXPECT_EQ(r4, u_b);
  }
  {
    String r1 = ArithOps::And(s_a, s_b);
    EXPECT_EQ(r1, s_b);
    String r2 = ArithOps::And(s_a.view(), s_b);
    EXPECT_EQ(r2, s_b);
    String r3 = ArithOps::And(s_a, s_b.view());
    EXPECT_EQ(r3, s_b);
    String r4 = ArithOps::And(s_a.view(), s_b.view());
    EXPECT_EQ(r4, s_b);
  }
}

TEST(ArithOps, PythonicOr) {
  Unicode u_a(U"hello");
  Unicode u_b;
  String s_a("hello");
  String s_b;
  // Any and other
  {
    RTValue r1 = ArithOps::Or(RTValue(u_a), u_b);
    EXPECT_EQ(r1, RTValue(u_a));
    RTValue r2 = ArithOps::Or(u_a, RTValue(u_b));
    EXPECT_EQ(r2, RTValue(u_a));
    RTValue r3 = ArithOps::Or(RTValue(u_a), RTValue(u_b));
    EXPECT_EQ(r3, RTValue(u_a));
    RTValue r4 = ArithOps::Or(RTView(u_a), u_b);
    EXPECT_EQ(r4, RTValue(u_a));
    RTValue r5 = ArithOps::Or(u_a, RTView(u_b));
    EXPECT_EQ(r5, RTValue(u_a));
    RTValue r6 = ArithOps::Or(RTView(u_a), RTView(u_b));
    EXPECT_EQ(r6, RTValue(u_a));
  }
  {
    RTValue r1 = ArithOps::Or(RTValue(u_a), u_b.view());
    EXPECT_EQ(r1, RTValue(u_a));
    RTValue r2 = ArithOps::Or(u_a.view(), RTValue(u_b));
    EXPECT_EQ(r2, RTValue(u_a));
    RTValue r3 = ArithOps::Or(RTView(u_a), u_b.view());
    EXPECT_EQ(r3, RTValue(u_a));
    RTValue r4 = ArithOps::Or(u_a.view(), RTView(u_b));
    EXPECT_EQ(r4, RTValue(u_a));
  }
  {
    RTValue r1 = ArithOps::Or(RTValue(s_a), s_b);
    EXPECT_EQ(r1, RTValue(s_a));
    RTValue r2 = ArithOps::Or(s_a, RTValue(s_b));
    EXPECT_EQ(r2, RTValue(s_a));
    RTValue r3 = ArithOps::Or(RTValue(s_a), RTValue(s_b));
    EXPECT_EQ(r3, RTValue(s_a));
    RTValue r4 = ArithOps::Or(RTView(s_a), s_b);
    EXPECT_EQ(r4, RTValue(s_a));
    RTValue r5 = ArithOps::Or(s_a, RTView(s_b));
    EXPECT_EQ(r5, RTValue(s_a));
    RTValue r6 = ArithOps::Or(RTView(s_a), RTView(s_b));
    EXPECT_EQ(r6, RTValue(s_a));
  }
  {
    RTValue r1 = ArithOps::Or(RTValue(s_a), s_b.view());
    EXPECT_EQ(r1, RTValue(s_a));
    RTValue r2 = ArithOps::Or(s_a.view(), RTValue(s_b));
    EXPECT_EQ(r2, RTValue(s_a));
    RTValue r3 = ArithOps::Or(RTView(s_a), s_b.view());
    EXPECT_EQ(r3, RTValue(s_a));
    RTValue r4 = ArithOps::Or(s_a.view(), RTView(s_b));
    EXPECT_EQ(r4, RTValue(s_a));
  }
  // unicode
  {
    Unicode r1 = ArithOps::Or(u_a, u_b);
    EXPECT_EQ(r1, u_a);
    Unicode r2 = ArithOps::Or(u_a.view(), u_b);
    EXPECT_EQ(r2, u_a);
    Unicode r3 = ArithOps::Or(u_a, u_b.view());
    EXPECT_EQ(r3, u_a);
    Unicode r4 = ArithOps::Or(u_a.view(), u_b.view());
    EXPECT_EQ(r4, u_a);
  }
  {
    String r1 = ArithOps::Or(s_a, s_b);
    EXPECT_EQ(r1, s_a);
    String r2 = ArithOps::Or(s_a.view(), s_b);
    EXPECT_EQ(r2, s_a);
    String r3 = ArithOps::Or(s_a, s_b.view());
    EXPECT_EQ(r3, s_a);
    String r4 = ArithOps::Or(s_a.view(), s_b.view());
    EXPECT_EQ(r4, s_a);
  }
}

TEST(ArithOps, floormod) {
  {
    int64_t a = 9223372036854775807LL;
    int64_t b = 4611686018427387904LL;
    int64_t r = ArithOps::floormod(a, b);
    EXPECT_EQ(r, 4611686018427387903LL);
  }
  {
    double a = 5.0;
    double b = 3.0;
    double r = ArithOps::floormod(a, b);
    EXPECT_DOUBLE_EQ(r, 2.0);
  }
}

TEST(ArithOps, floordiv) {
  {
    int64_t a = 9223372036854775807LL;
    int64_t b = 2LL;
    int64_t r = ArithOps::floordiv(a, b);
    EXPECT_EQ(r, 4611686018427387903LL);
  }
  {
    double a = 5.1;
    double b = 3.2;
    double r = ArithOps::floordiv(a, b);
    EXPECT_DOUBLE_EQ(r, 1.0);
  }
}

}  // namespace runtime
}  // namespace matxscript
