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
#include <matxscript/runtime/runtime_value.h>
#include <iostream>

namespace matxscript {
namespace runtime {

TEST(Runtime, Printer) {
  {
    List abc{"abc", U"hello", 1};
    {
      std::stringstream stream;
      stream << abc;
      auto repr = stream.str();
      EXPECT_EQ(repr, "[b'abc', \'hello\', 1]");
    }
    {
      std::stringstream stream;
      stream << RTValue(abc);
      auto repr = stream.str();
      EXPECT_EQ(repr, "[b'abc', \'hello\', 1]");
    }
  }
  {
    Dict abc{{"abc", U"abc"}, {U"hello", "hi"}};
    {
      std::stringstream stream;
      stream << abc;
      auto repr = stream.str();
      EXPECT_EQ(repr, "{b'abc': \'abc\', \'hello\': b\'hi\'}");
    }
    {
      std::stringstream stream;
      stream << RTValue(abc);
      auto repr = stream.str();
      EXPECT_EQ(repr, "{b'abc': \'abc\', \'hello\': b\'hi\'}");
    }
  }
}

}  // namespace runtime
}  // namespace matxscript
