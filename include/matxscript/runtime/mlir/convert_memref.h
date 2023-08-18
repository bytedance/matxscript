// Copyright 2023 ByteDance Ltd. and/or its affiliates.
//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include "cstdlib"
#include "matxscript/runtime/container/ndarray.h"
#include "matxscript/runtime/data_type.h"
#include "matxscript/runtime/dlpack.h"
#include "memory"
#include "string"
#include "vector"

namespace matxscript {
namespace runtime {
namespace mlir {

std::shared_ptr<void> alloc_memref_descriptor_ptr(size_t ndim,
                                                  DLManagedTensor* dl_managed_tensor = nullptr);
std::shared_ptr<void> convert_from_raw_ptr(void* raw_memref, bool ok_to_free_data = false);
std::shared_ptr<void> convert_from_dl_managed_tensor(DLManagedTensor* dl_managed_tensor);
DLManagedTensor* convert_to_dl_managed_tensor(std::shared_ptr<void>& memref_ptr,
                                              size_t ndim,
                                              DLDataType dtype);

NDArray convert_to_ndarray(std::shared_ptr<void>& memref_ptr, size_t ndim, DLDataType dtype);
NDArray convert_to_ndarray(void* memref_ptr,
                           size_t ndim,
                           DLDataType dtype,
                           bool ok_to_free_data = false);
std::shared_ptr<void> convert_from_ndarray(NDArray& nd);

DLDataType cvt_str_to_dl_dtype(const std::string& str);

bool is_overlapping(void* target, std::initializer_list<void*> others);

}  // namespace mlir
}  // namespace runtime
}  // namespace matxscript