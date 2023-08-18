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
#include "matxscript/runtime/mlir/memref_cpp_interface.h"

template <typename src_t, typename dst_t>
void copy(src_t* src, dst_t* dst, size_t n) {
  for (int i = 0; i < n; i++) {
    dst[i] = static_cast<dst_t>(src[i]);
  }
}

size_t calculate_memref_descriptor_size(size_t ndim) {
  return sizeof(void*) * 2 + sizeof(intptr_t) * (2 * ndim + 1);
}

template <typename shape_t, typename dst_t>
void compute_strides(shape_t* shape, dst_t* dst, size_t n) {
  uint64_t last = 1;
  dst[n - 1] = static_cast<dst_t>(last);

  for (int64_t i = static_cast<int64_t>(n) - 2; i >= 0; --i) {
    last = last * shape[i + 1];
    dst[i] = static_cast<dst_t>(last);
  }
}

namespace matxscript {
namespace runtime {
namespace mlir {

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

std::shared_ptr<void> convert_from_raw_ptr(void* raw_memref, bool ok_to_free_data) {
  std::shared_ptr<void> result(raw_memref, [ok_to_free_data](void* p) {
    if (ok_to_free_data) {
      free(p);
    }
  });
  return result;
}

std::shared_ptr<void> convert_from_dl_managed_tensor(DLManagedTensor* dl_managed_tensor) {
  DLTensor& dl_tensor = dl_managed_tensor->dl_tensor;
  auto& ndim = dl_tensor.ndim;
  // allocate memref_descriptor_ptr
  auto result = alloc_memref_descriptor_ptr(ndim, dl_managed_tensor);
  // get memory address
  MemrefCPPInterface memref_ci(result.get(), ndim);
  // setup memref
  memref_ci.set_allocated_ptr(dl_tensor.data);
  memref_ci.set_aligned_ptr(dl_tensor.data);
  memref_ci.set_offset(static_cast<memref_size_t>(dl_tensor.byte_offset));
  memref_ci.set_sizes(dl_tensor.shape);
  if (dl_tensor.strides == nullptr) {
    memref_ci.compute_strides(dl_tensor.shape);
  } else {
    memref_ci.set_strides(dl_tensor.strides);
  }
  memref_ci.display<int32_t>();
  return result;
}

DLManagedTensor* convert_to_dl_managed_tensor(std::shared_ptr<void>& memref_ptr,
                                              size_t ndim,
                                              DLDataType dtype) {
  // get memory address
  MemrefCPPInterface memref_ci(memref_ptr.get(), ndim);
  memref_ci.display<int32_t>();

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
  copy(memref_ci.get_sizes(), dl_tensor_shape, ndim);
  copy(memref_ci.get_strides(), dl_tensor_strides, ndim);

  auto dl_offset = static_cast<uint64_t>(memref_ci.get_offset());

  // for now assume it to be on CPU
  DLDevice dlDevice = {kDLCPU, 0};
  DLTensor dlTensor = {memref_ci.get_aligned(),
                       dlDevice,
                       static_cast<int32_t>(ndim),
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
    delete[] manager_ctx;
    delete self;
  };
  auto managed_tensor = new DLManagedTensor{dlTensor, manager_ctx, deleter};

  return managed_tensor;
}

NDArray convert_to_ndarray(std::shared_ptr<void>& memref_ptr, size_t ndim, DLDataType dtype) {
  DLManagedTensor* dl_managed_tensor = convert_to_dl_managed_tensor(memref_ptr, ndim, dtype);
  return NDArray::FromDLPack(dl_managed_tensor);
}

NDArray convert_to_ndarray(void* memref_ptr, size_t ndim, DLDataType dtype, bool ok_to_free_data) {
  auto&& shared_memref = convert_from_raw_ptr(memref_ptr, ok_to_free_data);
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

bool is_overlapping(void* target, std::initializer_list<void*> others) {
  auto* target_dptr = reinterpret_cast<void**>(target);
  return std::any_of(others.begin(), others.end(), [target_dptr](void* other_ptr) {
    auto* other_dptr = reinterpret_cast<void**>(other_ptr);
    return *target_dptr == *other_dptr;
  });
}

}  // namespace mlir
}  // namespace runtime
}  // namespace matxscript