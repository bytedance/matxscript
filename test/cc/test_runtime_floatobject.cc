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
#include <cfloat>

#include <matxscript/runtime/builtins_modules/_floatobject.h>

namespace matxscript {
namespace runtime {

TEST(PythonBuiltins, floatobject_float_rem) {
  EXPECT_EQ(py_builtins::float_rem(5.0, 3.0), 2.0);
  EXPECT_EQ(py_builtins::float_rem(5.0, -3.0), -1.0);
  EXPECT_EQ(py_builtins::float_rem(-5.0, 3.0), 1.0);
  EXPECT_EQ(py_builtins::float_rem(-5.0, -3.0), -2.0);
}

TEST(PythonBuiltins, floatobject_float_floor_div) {
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(5.1, 3.2), 1.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(5.1, -3.2), -2.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(-5.1, 3.2), -2.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(-5.1, -3.2), 1.0);

  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.1, 3.2), 0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.1, -3.2), -1.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.1, 0.1), 1.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.1, -0.1), -1.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(-0.1, 3.2), -1.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(-0.1, -3.2), -0.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(-0.1, 0.1), -1.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(-0.1, -0.1), 1.0);

  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.0, 3.2), 0.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.0, -3.2), -0.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.0, 0.1), 0.0);
  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(0.0, -0.1), -0.0);

  EXPECT_DOUBLE_EQ(py_builtins::float_floor_div(double(INT64_MAX), double(INT64_MAX)), 1.0);
}

}  // namespace runtime
}  // namespace matxscript
