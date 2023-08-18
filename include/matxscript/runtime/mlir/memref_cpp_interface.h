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
#include <iostream>

namespace matxscript {
namespace runtime {
namespace mlir {

using memref_size_t = intptr_t;
using data_t = void;
using data_array = data_t *;

template <typename T, size_t N>
struct MemRefDescriptor {
  T* allocated;
  T* aligned;
  memref_size_t offset;
  memref_size_t sizes[N];
  memref_size_t strides[N];
};


typedef struct {
  data_t* allocated;
  data_t* aligned;
  memref_size_t offset;
  memref_size_t rest[];
} MemrefCInterface;

class MemrefCPPInterface {
 public:
  // MemRefDescriptor layout:
  // |allocated_ptr|aligned_ptr|offset|sizes|strides|
  // | data_array  | data_array|memref_size_t*(1+2*ndim)|
  explicit MemrefCPPInterface(void * memref_des_ptr, size_t ndim) : _memref_des_ptr(memref_des_ptr), _ndim(ndim) {
    // view_as_data_arrays layout:
    // |allocated_ptr|aligned_ptr| rest ...
    // | data_array  | data_array| data_array ...
    auto ci = static_cast<MemrefCInterface *>(memref_des_ptr);
    allocated_ptr = &(ci->allocated);
    aligned_ptr = &(ci->aligned);

    // rest layout:
    // |offset|sizes|strides|
    // |memref_size_t|memref_size_t*ndim|memref_size_t*ndim|
    offset_ptr = &(ci->offset);
    sizes_array = &(ci->rest[0]);
    strides_array = &(ci->rest[ndim]);
  }

  void* get_memref_des_ptr() const;
  size_t get_ndim() const;

  data_t* get_allocated() const;
  data_t* get_aligned() const;
  memref_size_t get_offset() const;
  memref_size_t* get_sizes() const;
  memref_size_t* get_strides() const;

  void set_allocated_ptr(data_t* new_allocated_ptr);
  void set_aligned_ptr(data_t* new_aligned_ptr);
  void set_offset(memref_size_t new_offset);
  template <typename T>
  void set_sizes(T* shape) {
    for (int i = 0; i < _ndim; i++) {
      sizes_array[i] = static_cast<memref_size_t>(shape[i]);
    }
  }

  template <typename T>
  void set_strides(T* strides) {
    for (int i = 0; i < _ndim; i++) {
      strides_array[i] = static_cast<memref_size_t>(strides[i]);
    }
  }

  template <typename T>
  void compute_strides(T* shape) {
    uint64_t last = 1;
    strides_array[_ndim-1] = static_cast<memref_size_t>(last);

    for (int64_t i = static_cast<int64_t>(_ndim) - 2; i >= 0 ; --i) {
      last = last * shape[i+1];
      strides_array[i] = static_cast<memref_size_t>(last);
    }
  }

  template <typename T>
  void display() {
    std::cout << "get_memref_des_ptr() = " << get_memref_des_ptr() << std::endl;
    std::cout << "get_ndim()           = " << get_ndim() << std::endl;
    std::cout << "allocated_ptr        = " << allocated_ptr << std::endl;
    std::cout << "get_allocated()      = " << get_allocated() << std::endl;
    std::cout << "aligned_ptr          = " << aligned_ptr << std::endl;
    std::cout << "get_aligned()        = " << get_aligned() << std::endl;
    std::cout << "offset_ptr           = " << offset_ptr << std::endl;
    std::cout << "get_offset()         = " << get_offset() << std::endl;
    std::cout << "get_sizes()          = " << get_sizes() << std::endl;
    std::cout << "get_strides()        = " << get_strides() << std::endl;
    size_t total = 1;
    for(int i=0; i<get_ndim(); i++){
      std::cout << "memrefCInterface.size_ptr["<<i<<"]" <<(get_sizes())[i]<< std::endl;
      total *= (get_sizes())[i];
    }
    for(int i=0; i<get_ndim(); i++){
      std::cout << "memrefCInterface.stride_ptr["<<i<<"]" <<(get_strides())[i]<< std::endl;
    }
    auto *data_ptr = static_cast<T*>(get_aligned());
    for(int i=0; i<total; i++){
      std::cout << "memrefCInterface.get_aligned()["<<i<<"]" <<data_ptr[i]<< std::endl;
    }

    std::cout << std::endl;

  }

  private:
  void* _memref_des_ptr;
  size_t _ndim;
  data_t** allocated_ptr;
  data_t** aligned_ptr;
  memref_size_t * offset_ptr;
  memref_size_t * sizes_array;
  memref_size_t * strides_array;
};


}  // namespace mlir
}  // namespace runtime
}  // namespace matxscript