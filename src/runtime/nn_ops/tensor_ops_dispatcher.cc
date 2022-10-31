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
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/registry.h>

namespace {

using namespace ::matxscript::runtime;

// only an simple example
static NDArray tensor_ops_zeros(const RTValue& shape) {
  if (!shape.IsObjectRef<List>()) {
    MXTHROW << "expect shape is list type, but get " << shape.type_name();
  }
  int64_t bytes_len = 1;
  auto ls = shape.AsObjectRefNoCheck<List>();
  std::vector<int64_t> ft_shape;
  for (const auto& di : ls) {
    ft_shape.push_back(di.As<int64_t>());
    bytes_len *= ft_shape.back();
  }
  if (ft_shape.empty()) {
    bytes_len = 0;
  }
  auto ret = NDArray::Empty(ft_shape, {kDLFloat, 64, 1}, {kDLCPU, 0});
  auto dptr = (char*)(ret->data) + ret->byte_offset;
  memset(dptr, 0, bytes_len);
  return ret;
}

MATX_REGISTER_NATIVE_NAMED_FUNC("numpy_ops_zeros", tensor_ops_zeros);
MATX_REGISTER_NATIVE_NAMED_FUNC("torch_ops_zeros", tensor_ops_zeros);

}  // namespace
