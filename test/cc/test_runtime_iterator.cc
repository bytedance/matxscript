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

namespace matxscript {
namespace runtime {

TEST(Iterator, Constructor) {
  {
    List container{"hello", "hello", 3, List{"bb"}, Dict{{"hello", "good"}}, Set{"jj"}};
    Iterator iter = container.iter();
    RTValue xx;
    List new_con;
    while (iter.HasNext()) {
      xx = iter.Next();
      new_con.append(xx);
      std::cout << "xx: " << xx << std::endl;
    }
    EXPECT_EQ(new_con, container);
  }
  {
    Set container{"hello", "hello", 3};
    Iterator iter = container.iter();
    RTValue xx;
    Set new_con;
    while (iter.HasNext()) {
      xx = iter.Next();
      new_con.add(xx);
      std::cout << "xx: " << xx << std::endl;
    }
    EXPECT_EQ(new_con, container);
  }
  {
    Set container{"hello", "hello", 3};
    Iterator iter = container.iter();
    RTValue xx;
    Set new_con;
    bool has_next = iter.HasNext();
    while (has_next) {
      xx = iter.Next(&has_next);
      new_con.add(xx);
      std::cout << "xx: " << xx << std::endl;
    }
    EXPECT_EQ(new_con, container);
  }
}

}  // namespace runtime
}  // namespace matxscript
