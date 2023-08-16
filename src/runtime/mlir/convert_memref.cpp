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

#include "matxscript/runtime/mlir/convert_memref.h"

template <typename src_t, typename dst_t>
void copy(src_t* src, dst_t* dst, size_t n) {
  for (int i = 0; i < n; i++) {
    dst[n] = static_cast<dst_t>(src[n]);
  }
}

size_t calculate_memref_descriptor_size(size_t ndim) {
  return sizeof(void*) * 2 + sizeof(intptr_t) * (2 * ndim + 1);
}

namespace matxscript {
namespace runtime {
namespace mlir {

typedef struct {
  void* raw_ptr;
  void** allocated_dptr;
  void** aligned_dptr;
  memref_size_t* offset_ptr;
  memref_size_t* size_ptr;
  memref_size_t* stride_ptr;
} MemrefCInterface;

inline MemrefCInterface get_memref_ci(void* raw_ptr, size_t ndim) {
  void* memref = raw_ptr;
  void** allocated = (void**)(memref);
  void** aligned = allocated + 1;
  auto* offset = (memref_size_t*)(allocated + 1);
  memref_size_t* sizes = (offset + 1);
  memref_size_t* strides = sizes + ndim;
  return {raw_ptr, allocated, aligned, offset, sizes, strides};
}

std::shared_ptr<void> alloc_memref_descriptor_ptr(size_t ndim, DLManagedTensor* dl_managed_tensor) {
  void* memref = malloc(calculate_memref_descriptor_size(ndim));
  std::shared_ptr<void> result(memref, [dl_managed_tensor](void* p) {
    free(p);
    if (dl_managed_tensor != nullptr) {
      dl_managed_tensor->deleter(dl_managed_tensor);
    }
  });
  return result;
}

std::shared_ptr<void> convert_from_raw_ptr(void* raw_memref) {
  std::shared_ptr<void> result(raw_memref, [](void* p) { free(p); });
  return result;
}

std::shared_ptr<void> convert_from_dl_managed_tensor(DLManagedTensor* dl_managed_tensor) {
  DLTensor& dl_tensor = dl_managed_tensor->dl_tensor;
  int& ndim = dl_tensor.ndim;
  // allocate memref_descriptor_ptr
  auto result = alloc_memref_descriptor_ptr(ndim, dl_managed_tensor);
  // get memory address
  auto&& memref_ci = get_memref_ci(result.get(), ndim);
  // setup memref
  *(memref_ci.allocated_dptr) = dl_tensor.data;
  *(memref_ci.aligned_dptr) = dl_tensor.data;
  *(memref_ci.offset_ptr) = static_cast<memref_size_t>(dl_tensor.byte_offset);
  copy(dl_tensor.shape, memref_ci.size_ptr, ndim);
  copy(dl_tensor.strides, memref_ci.stride_ptr, ndim);
  return result;
}

DLManagedTensor* convert_to_dl_managed_tensor(std::shared_ptr<void>& memref_ptr,
                                              size_t ndim,
                                              DLDataType dtype) {
  // get memory address
  auto&& memref_ci = get_memref_ci(memref_ptr.get(), ndim);

  // in case the element type of dltensor is change
  using shape_ptr_type = decltype(DLTensor::shape);
  using shape_element_type = std::remove_pointer_t<shape_ptr_type>;
  using strides_ptr_type = decltype(DLTensor::strides);
  using strides_element_type = std::remove_pointer_t<strides_ptr_type>;

  // allocate memory space for shape array and stride array
  size_t allocate_size = (sizeof(shape_element_type) + sizeof(strides_element_type)) * ndim;
  void* dl_tensor_shape_and_strides = malloc(allocate_size);
  auto dl_tensor_shape = static_cast<shape_ptr_type>(dl_tensor_shape_and_strides);
  auto dl_tensor_strides = static_cast<strides_ptr_type>(dl_tensor_shape + ndim);

  // copy info from memref to dl tensor
  copy(memref_ci.stride_ptr, dl_tensor_shape, ndim);
  copy(memref_ci.stride_ptr, dl_tensor_strides, ndim);

  auto dl_offset = static_cast<uint64_t>(*(memref_ci.offset_ptr));

  // for now assume it to be on CPU
  DLDevice dlDevice = {kDLCPU, 0};
  DLTensor dlTensor = {*(memref_ci.aligned_dptr),
                       dlDevice,
                       (int32_t)ndim,
                       dtype,
                       dl_tensor_shape,
                       dl_tensor_strides,
                       dl_offset};
  auto* manager_ctx = new std::shared_ptr<void>[2];
  manager_ctx[0] = memref_ptr;
  manager_ctx[1] = std::shared_ptr<void>(dl_tensor_shape_and_strides, [](void* p) { free(p); });

  auto deleter = [](DLManagedTensor* self) {
    auto* manager_ctx = static_cast<std::shared_ptr<void>*>(self->manager_ctx);
    manager_ctx[0].reset();
    manager_ctx[1].reset();
    delete manager_ctx;
    delete self;
  };
  auto managed_tensor = new DLManagedTensor{dlTensor, manager_ctx, deleter};

  return managed_tensor;
}

NDArray convert_to_ndarray(std::shared_ptr<void>& memref_ptr, size_t ndim, DLDataType dtype) {
  DLManagedTensor* dl_managed_tensor = convert_to_dl_managed_tensor(memref_ptr, ndim, dtype);
  return NDArray::FromDLPack(dl_managed_tensor);
}

NDArray convert_to_ndarray(void* memref_ptr, size_t ndim, DLDataType dtype) {
  auto&& shared_memref = convert_from_raw_ptr(memref_ptr);
  return convert_to_ndarray(shared_memref, ndim, dtype);
}

std::shared_ptr<void> convert_from_ndarray(NDArray& nd) {
  auto* dl_managed_tensor = nd.ToDLPack();
  return convert_from_dl_managed_tensor(dl_managed_tensor);
}

DLDataType cvt_str_to_dl_dtype(const std::string& str) {
  static std::unordered_map<std::string, DLDataType> map = {{"bool", {kDLInt, 1, 1}},
                                                            {"int8", {kDLInt, 8, 1}},
                                                            {"int16", {kDLInt, 16, 1}},
                                                            {"int32", {kDLInt, 32, 1}},
                                                            {"int64", {kDLInt, 64, 1}},
                                                            {"uint8", {kDLUInt, 8, 1}},
                                                            {"uint16", {kDLUInt, 16, 1}},
                                                            {"uint32", {kDLUInt, 32, 1}},
                                                            {"uint64", {kDLUInt, 64, 1}},
                                                            {"float16", {kDLFloat, 16, 1}},
                                                            {"float32", {kDLFloat, 32, 1}},
                                                            {"float64", {kDLFloat, 64, 1}}};
  return map[str];
}

}  // namespace mlir
}  // namespace runtime
}  // namespace matxscript