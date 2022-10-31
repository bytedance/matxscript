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
#include <matxscript/runtime/container/iterator_adapter.h>
#include <matxscript/runtime/registry.h>
#include <iostream>
#include <random>

namespace matxscript {
namespace runtime {

MATXSCRIPT_REGISTER_GLOBAL("test.PackedFunc_FloatTensorMaker").set_body_typed([](List shape) {
  // automatically convert arguments to desired type.
  using Converter = GenericValueConverter<int64_t>;
  using IterAdapter = IteratorAdapter<Converter, List::iterator>;
  std::vector<int64_t> stl_shape{IterAdapter(shape.begin()), IterAdapter(shape.end())};
  DataType dtype = DataType::Float(32);
  auto result = NDArray::Empty(stl_shape, dtype, {kDLCPU, 0});
  auto ptr_b = static_cast<float*>(result->data);
  for (size_t i = 0; i < 6; ++i) {
    ptr_b[i] = 1.1;
  }
  // automatically assign value return to rv
  return result;
});

TEST(PackedFunc, GenericTest) {
  const auto* test_func =
      ::matxscript::runtime::FunctionRegistry::Get("test.PackedFunc_FloatTensorMaker");
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  List shape = {2, 3};
  NDArray tsr = (*test_func)({RTView(shape)}).As<NDArray>();
  std::cout << tsr << std::endl;
}

}  // namespace runtime
}  // namespace matxscript
