// Copyright 2023 ByteDance Ltd. and/or its affiliates.
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
#include <array>
#include "matxscript/runtime/mlir/convert_memref.h"
#include "matxscript/runtime/mlir/memref_cpp_interface.h"

namespace matxscript {
namespace runtime {
namespace mlir {

TEST(MLIR, dl_pack_conversion_1d) {
  std::array<int32_t, 10> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  DLDevice device = {kDLCPU, 0};
  int32_t ndim = 1;
  DLDataType dtype = {kDLInt, 32, 1};
  std::array<int64_t, 1> shape = {10};
  std::array<int64_t, 1> strides = {1};
  uint64_t byte_offset = 0;
  DLTensor dl_Tensor = {
      data.data(), device, ndim, dtype, shape.data(), strides.data(), byte_offset};
  DLManagedTensor dl_managed_tensor = {dl_Tensor, nullptr, nullptr};
  std::cout << 41 << std::endl;
  auto&& memref_shared_ptr = convert_from_dl_managed_tensor(&dl_managed_tensor);
  std::cout << 42 << std::endl;
  MemrefCPPInterface memref_ptr(memref_shared_ptr.get(), ndim);

  EXPECT_EQ(memref_ptr.get_allocated(), data.data());
  EXPECT_EQ(memref_ptr.get_aligned(), data.data());
  EXPECT_EQ(memref_ptr.get_offset(), 0);
  EXPECT_EQ(*(memref_ptr.get_sizes()), 10);
  EXPECT_EQ(*(memref_ptr.get_strides()), 1);

  auto&& convert_back_dl_tensor = convert_to_dl_managed_tensor(memref_shared_ptr, ndim, dtype);

  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.data, data.data());
  EXPECT_EQ(dl_Tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dl_Tensor.device.device_id, 0);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.ndim, ndim);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.bits, 32);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.lanes, 1);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.shape[0], 10);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.strides[0], 1);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.byte_offset, 0);
}

TEST(MLIR, dl_pack_conversion_2d) {
  std::array<int32_t, 12> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  DLDevice device = {kDLCPU, 0};
  const int32_t ndim = 2;
  DLDataType dtype = {kDLInt, 32, 1};
  std::array<int64_t, ndim> shape = {3, 4};
  std::array<int64_t, ndim> strides = {4, 1};
  uint64_t byte_offset = 0;
  DLTensor dl_Tensor = {
      data.data(), device, ndim, dtype, shape.data(), strides.data(), byte_offset};
  DLManagedTensor dl_managed_tensor = {dl_Tensor, nullptr, nullptr};
  auto&& memref_shared_ptr = convert_from_dl_managed_tensor(&dl_managed_tensor);
  MemrefCPPInterface memref_ptr(memref_shared_ptr.get(), ndim);

  EXPECT_EQ(memref_ptr.get_allocated(), data.data());
  EXPECT_EQ(memref_ptr.get_aligned(), data.data());
  EXPECT_EQ(memref_ptr.get_offset(), 0);
  EXPECT_EQ(*(memref_ptr.get_sizes()), 3);
  EXPECT_EQ(*(memref_ptr.get_sizes() + 1), 4);
  EXPECT_EQ(*(memref_ptr.get_strides()), 4);
  EXPECT_EQ(*(memref_ptr.get_strides() + 1), 1);

  auto&& convert_back_dl_tensor = convert_to_dl_managed_tensor(memref_shared_ptr, ndim, dtype);

  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.data, data.data());
  EXPECT_EQ(dl_Tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dl_Tensor.device.device_id, 0);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.ndim, ndim);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.bits, 32);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.lanes, 1);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.shape[0], 3);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.shape[1], 4);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.strides[0], 4);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.strides[1], 1);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.byte_offset, 0);
}

TEST(MLIR, dl_pack_conversion_2d_null_strides) {
  std::array<int32_t, 12> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  DLDevice device = {kDLCPU, 0};
  const int32_t ndim = 2;
  DLDataType dtype = {kDLInt, 32, 1};
  std::array<int64_t, ndim> shape = {3, 4};
  uint64_t byte_offset = 0;
  DLTensor dl_Tensor = {data.data(), device, ndim, dtype, shape.data(), nullptr, byte_offset};
  DLManagedTensor dl_managed_tensor = {dl_Tensor, nullptr, nullptr};
  auto&& memref_shared_ptr = convert_from_dl_managed_tensor(&dl_managed_tensor);
  MemrefCPPInterface memref_ptr(memref_shared_ptr.get(), ndim);

  EXPECT_EQ(memref_ptr.get_allocated(), data.data());
  EXPECT_EQ(memref_ptr.get_aligned(), data.data());
  EXPECT_EQ(memref_ptr.get_offset(), 0);
  EXPECT_EQ(*(memref_ptr.get_sizes()), 3);
  EXPECT_EQ(*(memref_ptr.get_sizes() + 1), 4);
  EXPECT_EQ(*(memref_ptr.get_strides()), 4);
  EXPECT_EQ(*(memref_ptr.get_strides() + 1), 1);

  auto&& convert_back_dl_tensor = convert_to_dl_managed_tensor(memref_shared_ptr, ndim, dtype);

  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.data, data.data());
  EXPECT_EQ(dl_Tensor.device.device_type, kDLCPU);
  EXPECT_EQ(dl_Tensor.device.device_id, 0);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.ndim, ndim);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.code, kDLInt);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.bits, 32);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.dtype.lanes, 1);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.shape[0], 3);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.shape[1], 4);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.strides[0], 4);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.strides[1], 1);
  EXPECT_EQ(convert_back_dl_tensor->dl_tensor.byte_offset, 0);
}

TEST(MLIR, no_free_data) {
  std::array<int32_t, 12> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto&& no_free = convert_from_raw_ptr(data.data(), false);
  EXPECT_NO_FATAL_FAILURE(no_free.reset());
}

TEST(MLIR, convert_from_ndarray) {
  std::vector<int64_t> shape = {2, 3};
  DLDataType dtype = {kDLInt, 64, 1};
  DLDevice device = {kDLCPU, 0};
  auto nd = NDArray::Empty(shape, dtype, device);
  auto memref_shared_ptr = convert_from_ndarray(nd);
  MemrefCPPInterface memref_ptr(memref_shared_ptr.get(), 2);
  EXPECT_EQ(memref_ptr.get_allocated(), nd.Data<void>());
  EXPECT_EQ(memref_ptr.get_aligned(), nd.Data<void>());
  EXPECT_EQ(memref_ptr.get_offset(), 0);
  EXPECT_EQ(*(memref_ptr.get_sizes()), 2);
  EXPECT_EQ(*(memref_ptr.get_sizes() + 1), 3);
  EXPECT_EQ(*(memref_ptr.get_strides()), 3);
  EXPECT_EQ(*(memref_ptr.get_strides() + 1), 1);
}

}  // namespace mlir
}  // namespace runtime
}  // namespace matxscript