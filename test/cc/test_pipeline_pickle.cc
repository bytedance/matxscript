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
#include <matxscript/pipeline/pickle.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/json_util.h>

namespace matxscript {
namespace runtime {

TEST(Pipeline, pickle) {
  Dict d{{1, 2}, {Unicode(U"hello"), 3}};
  auto json_doc = pickle::ToJsonStruct(RTView(d));
  RTValue dd = pickle::FromJsonStruct(json_doc);
  ASSERT_EQ(dd, d);
  std::cout << JsonUtil::ToString(&json_doc, true) << std::endl;

  // Add Tupe
  auto tup = Tuple::dynamic(1, 2.5f, "a", List({1, 2}));
  auto tup_json = pickle::ToJsonStruct(RTView(tup));
  auto new_tup = pickle::FromJsonStruct(tup_json).AsObjectRefNoCheck<Tuple>();
  ASSERT_EQ(tup.size(), new_tup.size());
  ASSERT_EQ(tup[3], new_tup[3]);
}

TEST(Pickle, ndarray) {
  auto arr = ::matxscript::runtime::Kernel_NDArray::make({1.0, 2.0, 3.0, 4.0}, {2, 2}, U"float32");
  auto json_doc = pickle::ToJsonStruct(RTView(arr));

  auto new_arr = pickle::FromJsonStruct(json_doc).AsObjectRefNoCheck<NDArray>();
  ASSERT_EQ(arr.ShapeList(), new_arr.ShapeList());
  ASSERT_EQ(arr.DTypeUnicode(), new_arr.DTypeUnicode());
}

TEST(Pickle, serialzation) {
  auto arr = ::matxscript::runtime::Kernel_NDArray::make({1.0, 2.0, 3.0, 4.0}, {2, 2}, U"float32");
  auto json_doc = pickle::ToJsonStruct(RTView(arr));

  auto str = pickle::Serialize(RTView(arr));
  auto new_arr = pickle::DeSerialize(str);
  auto shape = new_arr.AsObjectRef<NDArray>().Shape();
  ASSERT_EQ(shape[0], 2);
  ASSERT_EQ(shape[1], 2);
}

}  // namespace runtime
}  // namespace matxscript
