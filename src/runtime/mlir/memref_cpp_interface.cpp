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

#include "matxscript/runtime/mlir/memref_cpp_interface.h"

namespace matxscript {
namespace runtime {
namespace mlir {

void* MemrefCPPInterface::get_memref_des_ptr() const {
  return _memref_des_ptr;
}

size_t MemrefCPPInterface::get_ndim() const {
  return _ndim;
}

data_t* MemrefCPPInterface::get_allocated() const {
  return *allocated_ptr;
}

data_t* MemrefCPPInterface::get_aligned() const {
  return *aligned_ptr;
}

memref_size_t MemrefCPPInterface::get_offset() const {
  return *offset_ptr;
}

memref_size_t* MemrefCPPInterface::get_sizes() const {
  return sizes_array;
}

memref_size_t* MemrefCPPInterface::get_strides() const {
  return strides_array;
}

void MemrefCPPInterface::set_allocated_ptr(data_t* new_allocated_ptr) {
  *allocated_ptr = new_allocated_ptr;
}

void MemrefCPPInterface::set_aligned_ptr(data_t* new_aligned_ptr) {
  *aligned_ptr = new_aligned_ptr;
}

void MemrefCPPInterface::set_offset(memref_size_t new_offset) {
  *offset_ptr = new_offset;
}

}  // namespace mlir
}  // namespace runtime
}  // namespace matxscript
