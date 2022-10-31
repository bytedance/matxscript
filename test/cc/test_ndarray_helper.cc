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
#include <matxscript/runtime/container/ndarray_helper.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <iostream>

namespace matxscript {
namespace runtime {

bool vector_equal(const std::vector<int64_t>& x, const std::vector<int64_t>& y) {
  if (x.size() != y.size()) {
    return false;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] != y[i]) {
      return false;
    }
  }
  return true;
}

TEST(NDArrayHelper, GetBroadcastShape) {
  std::vector<int64_t> shape;
  bool match = NDArrayHelper::GetBroadcastShape({3, 1, 4}, {2, 1, 1, 4}, shape);
  ASSERT_TRUE(match);
  ASSERT_TRUE(vector_equal(shape, {2, 3, 1, 4}));

  match = NDArrayHelper::GetBroadcastShape({3, 1, 4}, {2, 2}, shape);
  ASSERT_FALSE(match);

  match = NDArrayHelper::GetBroadcastShape({3, 1, 4}, {2, 1}, shape);
  ASSERT_TRUE(match);
  ASSERT_TRUE(vector_equal(shape, {3, 2, 4}));

  match = NDArrayHelper::GetBroadcastShape({2, 3}, {2, 3}, shape);
  ASSERT_TRUE(match);
  ASSERT_TRUE(vector_equal(shape, {2, 3}));
}

TEST(NDArrayHelper, AddIndexes) {
  std::vector<int64_t> shape{4, 1, 2, 5};
  std::vector<int64_t> indexes(4, 0);
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {0, 0, 0, 1}));
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {0, 0, 0, 2}));
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {0, 0, 0, 3}));
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {0, 0, 0, 4}));
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {0, 0, 1, 0}));
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {0, 0, 1, 1}));
  indexes = {3, 0, 1, 4};
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {0, 0, 0, 0}));
  indexes = {2, 0, 1, 4};
  NDArrayHelper::IndexesAddOne(shape, 4, indexes);
  ASSERT_TRUE(vector_equal(indexes, {3, 0, 0, 0}));
}

TEST(NDArrayHelper, Sub) {
  auto a =
      Kernel_NDArray::make({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2}, Unicode(U"int32"));
  auto b = Kernel_NDArray::make(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 2, 1, 2}, Unicode(U"int32"));
  std::cout << NDArrayOperate::Sub(a, b) << std::endl;
  auto c = Kernel_NDArray::make({7, 8, 9, 10, 11, 12}, {3, 2}, Unicode(U"int32"));
  std::cout << NDArrayOperate::Sub(1.5, c) << std::endl;
  std::cout << NDArrayOperate::Sub((int64_t)(10), c) << std::endl;

  std::cout << NDArrayOperate::Add(a, b) << std::endl;
  std::cout << NDArrayOperate::Add(c, 1.5) << std::endl;
  std::cout << NDArrayOperate::Add(c, (int64_t)(10)) << std::endl;
}

}  // namespace runtime
}  // namespace matxscript