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
#include <matxscript/runtime/half.h>

namespace matxscript {
namespace runtime {

TEST(Half, float_half_conversion) {
  { EXPECT_EQ(sizeof(Half), 2); }
  {
    float a = 2.2;
    Half h(a);
    float b = h;
    EXPECT_TRUE(std::abs(a - b) <= 0.01);
  }
  {
    int32_t a = 2;
    Half h((float(a)));
    int32_t b = std::floor(float(h));
    EXPECT_EQ(a, b);
  }
}

TEST(Half, Add) {
  Half h1 = 2.2;
  Half h2 = 1.1;
  Half h_result = h1 + h2;
  std::cout << "h1(" << h1 << ") + h2(" << h2 << ") = " << h_result << std::endl;
}

TEST(Half, Sub) {
  Half h1 = 2.2;
  Half h2 = 1.1;
  Half h_result = h1 - h2;
  std::cout << "h1(" << h1 << ") - h2(" << h2 << ") = " << h_result << std::endl;
}

TEST(Half, Mul) {
  Half h1 = 2.2;
  Half h2 = 1.1;
  Half h_result = h1 * h2;
  std::cout << "h1(" << h1 << ") * h2(" << h2 << ") = " << h_result << std::endl;
}

TEST(Half, Div) {
  Half h1 = 2.2;
  Half h2 = 1.1;
  Half h_result = h1 / h2;
  std::cout << "h1(" << h1 << ") / h2(" << h2 << ") = " << h_result << std::endl;
}

}  // namespace runtime
}  // namespace matxscript