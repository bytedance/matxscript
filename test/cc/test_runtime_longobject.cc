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
#include <matxscript/runtime/builtins_modules/_longobject.h>

namespace matxscript {
namespace runtime {

TEST(PythonBuiltins, longobject_fast_mod) {
  EXPECT_EQ(py_builtins::fast_mod(5, 3), 2);
  EXPECT_EQ(py_builtins::fast_mod(5, -3), -1);
  EXPECT_EQ(py_builtins::fast_mod(-5, 3), 1);
  EXPECT_EQ(py_builtins::fast_mod(-5, -3), -2);

  EXPECT_EQ(py_builtins::fast_mod(5ULL, 3), 2);
  EXPECT_EQ(py_builtins::fast_mod(5ULL, -3), -1);
  EXPECT_EQ(py_builtins::fast_mod(-5, 3ULL), 1);
  EXPECT_EQ(py_builtins::fast_mod(5, 3ULL), 2);
  EXPECT_EQ(py_builtins::fast_mod(5ULL, 3ULL), 2);

  EXPECT_EQ(py_builtins::fast_mod(INT64_MIN, 3), 1);
  EXPECT_EQ(py_builtins::fast_mod(INT64_MIN, -3), -2);
  EXPECT_EQ(py_builtins::fast_mod(INT64_MIN, INT64_MIN), 0);
  EXPECT_EQ(py_builtins::fast_mod(INT64_MIN, INT64_MAX), INT64_MAX - 1);

  EXPECT_EQ(py_builtins::fast_mod(INT64_MAX, 3), 1);
  EXPECT_EQ(py_builtins::fast_mod(INT64_MAX, -3), -2);
  EXPECT_EQ(py_builtins::fast_mod(INT64_MAX, INT64_MAX), 0);
  EXPECT_EQ(py_builtins::fast_mod(INT64_MAX, INT64_MIN), -1);

  EXPECT_EQ(py_builtins::fast_mod(0, 3), 0);
  EXPECT_EQ(py_builtins::fast_mod(0, -3), 0);
  EXPECT_EQ(py_builtins::fast_mod(0, INT64_MIN), 0);
  EXPECT_EQ(py_builtins::fast_mod(0, INT64_MAX), 0);
  EXPECT_EQ(py_builtins::fast_mod(0, UINT64_MAX), 0);

  EXPECT_EQ(py_builtins::fast_mod(UINT64_MAX, 3), 0);
  EXPECT_EQ(py_builtins::fast_mod(UINT64_MAX, -3), 0);
  EXPECT_EQ(py_builtins::fast_mod(UINT64_MAX, UINT64_MAX), 0);
  EXPECT_EQ(py_builtins::fast_mod(UINT64_MAX, INT64_MAX), 1);
  EXPECT_EQ(py_builtins::fast_mod(UINT64_MAX, INT64_MIN), -1);
  EXPECT_EQ(py_builtins::fast_mod(UINT64_MAX, -INT64_MAX), -INT64_MAX + 1);
}

TEST(PythonBuiltins, longobject_fast_floor_div) {
  EXPECT_EQ(py_builtins::fast_floor_div(5, 3), 1);
  EXPECT_EQ(py_builtins::fast_floor_div(5, -3), -2);
  EXPECT_EQ(py_builtins::fast_floor_div(-5, 3), -2);
  EXPECT_EQ(py_builtins::fast_floor_div(-5, -3), 1);

  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MIN, 3), -3074457345618258603LL);
  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MIN, -3), 3074457345618258602LL);
  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MIN, INT64_MIN), 1);
  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MIN, INT64_MAX), -2);

  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MAX, 3), 3074457345618258602LL);
  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MAX, -3), -3074457345618258603LL);
  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MAX, INT64_MAX), 1);
  EXPECT_EQ(py_builtins::fast_floor_div(INT64_MAX, INT64_MIN), -1);

  EXPECT_EQ(py_builtins::fast_floor_div(0, 3), 0);
  EXPECT_EQ(py_builtins::fast_floor_div(0, -3), 0);
  EXPECT_EQ(py_builtins::fast_floor_div(0, INT64_MIN), 0);
  EXPECT_EQ(py_builtins::fast_floor_div(0, INT64_MAX), 0);
  EXPECT_EQ(py_builtins::fast_floor_div(0, UINT64_MAX), 0);
}

}  // namespace runtime
}  // namespace matxscript
