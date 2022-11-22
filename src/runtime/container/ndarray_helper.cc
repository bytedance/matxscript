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
#include <matxscript/runtime/container/ndarray_helper.h>

#include <algorithm>
#include <functional>
#include <random>

#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/list_ref.h>
#include <matxscript/runtime/container/ndarray_private.h>
#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

namespace {
template <typename DstDtype, typename SrcDtype>
void Assign(DstDtype* dst_data,
            const SrcDtype* src_data,
            const int64_t* dst_strides,
            const int64_t* src_strides,
            const int64_t* shape,
            int64_t ndim) {
  if (ndim == 1) {
    for (int64_t i = 0; i < shape[0]; ++i) {
      dst_data[i * dst_strides[0]] = src_data[i * src_strides[0]];
    }
    return;
  }
  for (int64_t i = 0; i < shape[0]; ++i) {
    Assign(dst_data + i * dst_strides[0],
           src_data + i * src_strides[0],
           dst_strides + 1,
           src_strides + 1,
           shape + 1,
           ndim - 1);
  }
}

template <typename DstDtype, typename SrcDtype>
void Assign(DstDtype* dst_data, const SrcDtype* src_data, int64_t element_num) {
  for (int64_t i = 0; i < element_num; ++i) {
    dst_data[i] = src_data[i];
  }
}

template <typename Dtype>
void Assign(Dtype* dst_data, const Dtype* src_data, int64_t element_num) {
  memcpy(dst_data, src_data, element_num * sizeof(Dtype));
}

}  // namespace

std::vector<int64_t> NDArrayHelper::ExpandShape(const std::vector<int64_t>& shape, size_t dim) {
  if (dim <= shape.size()) {
    return shape;
  }
  std::vector<int64_t> ret(dim - shape.size(), 1);
  ret.insert(ret.end(), shape.begin(), shape.end());
  return ret;
}

std::vector<int64_t> NDArrayHelper::ExpandStrides(const std::vector<int64_t>& strides, size_t dim) {
  if (dim <= strides.size()) {
    return strides;
  }
  std::vector<int64_t> ret(dim - strides.size(), 0);
  ret.insert(ret.end(), strides.begin(), strides.end());
  return ret;
}

int64_t NDArrayHelper::Offset(const std::vector<int64_t>& indexes,
                              const std::vector<int64_t>& shape,
                              const std::vector<int64_t>& strides) {
  int64_t offset = 0;
  for (size_t i = 0; i < indexes.size(); ++i) {
    if (shape[i] != 1) {
      offset += indexes[i] * strides[i];
    }
  }
  return offset;
}

void NDArrayHelper::IndexesAddOne(const std::vector<int64_t>& shape,
                                  size_t dim,
                                  std::vector<int64_t>& indexes) {
  // assert dim > 0
  int i = dim - 1;
  for (int i = dim - 1; i >= 0; --i) {
    ++indexes[i];
    if (indexes[i] < shape[i]) {
      return;
    }
    indexes[i] = 0;
  }
}

bool NDArrayHelper::GetBroadcastShape(const std::vector<int64_t>& shape1,
                                      const std::vector<int64_t>& shape2,
                                      std::vector<int64_t>& broadcast_shape) {
  size_t dim1 = shape1.size();
  size_t dim2 = shape2.size();
  broadcast_shape.clear();
  broadcast_shape.resize(std::max(dim1, dim2));
  auto it1 = shape1.rbegin();
  auto it2 = shape2.rbegin();
  auto it = broadcast_shape.rbegin();
  while (it1 != shape1.rend() && it2 != shape2.rend()) {
    if (*it1 == *it2) {
      *it = *it1;
    } else if (*it1 == 1) {
      *it = *it2;
    } else if (*it2 == 1) {
      *it = *it1;
    } else {
      return false;
    }
    ++it1;
    ++it2;
    ++it;
  }
  for (; it1 != shape1.rend(); ++it1, ++it) {
    *it = *it1;
  }
  for (; it2 != shape2.rend(); ++it2, ++it) {
    *it = *it2;
  }
  return true;
}

bool NDArrayHelper::IsSameShape(const std::vector<int64_t>& shape1,
                                const std::vector<int64_t>& shape2) {
  if (shape1.size() != shape2.size()) {
    return false;
  }
  return std::equal(shape1.begin(), shape1.end(), shape2.begin());
}

bool NDArrayHelper::IsSameShape(const NDArray& nd1, const NDArray& nd2) {
  if (nd1.GetDim() != nd2.GetDim()) {
    return false;
  }
  int64_t ndim = nd1.GetDim();
  const int64_t* shape1 = nd1.GetShapePtr();
  const int64_t* shape2 = nd2.GetShapePtr();
  return std::equal(shape1, shape1 + ndim, shape2);
}

DataType NDArrayHelper::DTypePromotion(const DataType& dt1, const DataType& dt2) {
  if (dt1.is_float() && dt2.is_float()) {
    return DataType(dt1.code(), std::max(dt1.bits(), dt2.bits()), dt1.lanes());
  }
  if (dt1.is_int() && dt2.is_int()) {
    return DataType(dt1.code(), std::max(dt1.bits(), dt2.bits()), dt1.lanes());
  }
  if (dt1.is_float() && dt2.is_int()) {
    return dt1;
  }
  if (dt1.is_int() && dt2.is_float()) {
    return dt2;
  }
  MXTHROW << "unsupported dtype compare between " << DLDataType2String(dt1) << " and "
          << DLDataType2String(dt2);
  return {};
}

DataType NDArrayHelper::DTypeFromDouble(const DataType& dt) {
  if (dt.is_int()) {
    return DataType(String2DLDataType("float32"));
  }
  if (dt.is_float()) {
    return dt;
  }
  MXTHROW << "unsupported dtype " << DLDataType2String(dt) << " operating with double";
  return {};
}

void* NDArrayHelper::GetData(const NDArray& nd) {
  const DLTensor* dl_tensor = &((nd.get_mutable())->dl_tensor);
  return static_cast<void*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
}

int64_t NDArrayHelper::GetItemNum(const int64_t* shape, int dim) {
  int64_t ret = 1;
  for (int i = 0; i < dim; ++i) {
    ret *= shape[i];
  }
  return ret;
}

DLDevice NDArrayHelper::GetCPUDevice() {
  DLDevice device;
  device.device_type = DLDeviceType::kDLCPU;
  device.device_id = 0;
  return device;
}

NDArray NDArrayOperate::Rand(const std::vector<int64_t>& shape) {
  auto ret = NDArray::Empty(shape, String2DLDataType("float32"), NDArrayHelper::GetCPUDevice());
  float* p =
      static_cast<float*>(static_cast<void*>(static_cast<char*>(ret->data) + ret->byte_offset));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (int64_t i = 0; i < NDArrayHelper::GetItemNum(shape.data(), shape.size()); ++i) {
    p[i] = dis(gen);
  }
  return ret;
}

NDArray NDArrayOperate::Concatenate(const Any& seq, int64_t axis) {
  std::vector<NDArray> arrays;
  if (seq.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeList) {
    const auto& l = seq.AsObjectRefNoCheck<List>();
    for (const auto& array : l) {
      MXCHECK(array.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeNDArray)
          << "seq element must be a NDArray";
      arrays.push_back(array.AsNoCheck<NDArray>());
    }
  } else if (seq.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeTuple) {
    const auto& t = seq.AsObjectRefNoCheck<Tuple>();
    for (int i = 0; i < t.size(); ++i) {
      MXCHECK(t[i].type_code() == ::matxscript::runtime::TypeIndex::kRuntimeNDArray)
          << "seq element must be a NDArray";
      arrays.push_back(t[i].AsNoCheck<NDArray>());
    }
  } else {
    MXTHROW << "unsupported seq type, type_code: " << seq.type_code();
    return None.As<NDArray>();
  }

  if (arrays.empty()) {
    MXTHROW << "need at least one array to concatenate";
  }

  // all ndarray must have the same shapes
  auto ndim = arrays[0].get_mutable()->dl_tensor.ndim;
  if (ndim == 0) {
    MXTHROW << "zero-dimensional arrays cannot be concatenated";
  }
  axis = index_correction(axis, ndim);
  if (axis < 0 || axis >= ndim) {
    return None.As<NDArray>();
  }
  auto shape = arrays[0].Shape();
  for (int i = 1; i < arrays.size(); ++i) {
    MXCHECK(arrays[i].get_mutable()->dl_tensor.ndim == ndim)
        << "all the input arrays must have same "
        << "number of dimensions, but the array at "
        << "index " << 0 << " has " << ndim << " dimension(s) and the array at index " << i
        << " has " << arrays[i].get_mutable()->dl_tensor.ndim << " dimension(s)";
    const int64_t* arr_shape = arrays[i].get_mutable()->ShapeBegin();
    for (int idim = 0; idim < ndim; ++idim) {
      if (idim == axis) {
        shape[idim] += arr_shape[idim];
      } else {
        MXCHECK(shape[idim] == arr_shape[idim])
            << "all the input array dimensions for the "
            << "concatenation axis must match exactly, but "
            << "along dimension " << idim << ", the array at index " << 0 << " has "
            << "size " << shape[idim] << " and the array at index " << i << " has size "
            << arr_shape[idim];
      }
    }
  }

  DLDataType dtype = arrays[0].get_mutable()->dl_tensor.dtype;
  auto ret = NDArray::Empty(shape, dtype, arrays[0].get_mutable()->dl_tensor.device);
  auto sliding_view = ret.CreateView(shape, dtype);
  auto sliding_view_container = sliding_view.get_mutable();
  DLTensor* sliding_view_tensor = &(sliding_view_container->dl_tensor);
  sliding_view_tensor->strides = const_cast<int64_t*>(sliding_view_container->StridesBegin());
  for (int i = 0; i < arrays.size(); ++i) {
    sliding_view_tensor->shape[axis] = arrays[i].get_mutable()->Shape(axis);
    NDArray::AssignNDArray(arrays[i], sliding_view);
    MATX_NDARRAY_TYPE_SWITCH(dtype, DT, {
      sliding_view_tensor->byte_offset +=
          sliding_view_container->Shape(axis) * sliding_view_container->Strides(axis) * sizeof(DT);
    });
  }

  return ret;
}

NDArray NDArrayOperate::Stack(const Any& seq, int64_t axis) {
  std::vector<NDArray> arrays;
  if (seq.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeList) {
    const auto& l = seq.AsObjectRefNoCheck<List>();
    for (const auto& array : l) {
      MXCHECK(array.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeNDArray)
          << "seq element must be a NDArray";
      arrays.push_back(array.AsNoCheck<NDArray>());
    }
  } else if (seq.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeTuple) {
    const auto& t = seq.AsObjectRefNoCheck<Tuple>();
    for (int i = 0; i < t.size(); ++i) {
      MXCHECK(t[i].type_code() == ::matxscript::runtime::TypeIndex::kRuntimeNDArray)
          << "seq element must be a NDArray";
      arrays.push_back(t[i].AsNoCheck<NDArray>());
    }
  } else {
    MXTHROW << "unsupported seq type, type_code: " << seq.type_code();
    return None.As<NDArray>();
  }

  if (arrays.empty()) {
    MXTHROW << "need at least one array to stack";
  }

  // all arrays must have the same shapes
  for (int i = 1; i < arrays.size(); ++i) {
    if (!NDArrayHelper::IsSameShape(arrays[0], arrays[i])) {
      MXTHROW << "all array must be the same shape";
    }
  }

  int ndim = arrays[0].get_mutable()->dl_tensor.ndim + 1;
  axis = index_correction(axis, ndim);
  if (axis < 0 || axis >= ndim) {
    return None.As<NDArray>();
  }

  std::vector<int64_t> shape(ndim);
  int64_t arg_element_num = 1;
  for (int i = 0; i < ndim; ++i) {
    if (i < axis) {
      shape[i] = arrays[0].get_mutable()->Shape(i);
      arg_element_num *= shape[i];
    } else if (i == axis) {
      shape[i] = arrays.size();
    } else {
      shape[i] = arrays[0].get_mutable()->Shape(i - 1);
      arg_element_num *= shape[i];
    }
  }

  const auto& dtype = arrays[0].get_mutable()->dl_tensor.dtype;
  const auto& ctx = arrays[0].get_mutable()->dl_tensor.device;
  const int64_t* arg_shape = arrays[0].get_mutable()->dl_tensor.shape;
  NDArray ret = NDArray::Empty(shape, dtype, ctx);
  std::vector<int64_t> shrink_target_strides(ret.GetStridesPtr(),
                                             ret.GetStridesPtr() + ret.GetDim());
  int64_t axis_stride = shrink_target_strides[axis];
  shrink_target_strides.erase(shrink_target_strides.begin() + axis);

  MATX_NDARRAY_TYPE_SWITCH(dtype, DT, {
    DT* target_ptr = static_cast<DT*>(NDArrayHelper::GetData(ret));
    for (int i = 0; i < shape[axis]; ++i) {
      const int64_t* arg_strides = arrays[i].GetStridesPtr();
      MATX_NDARRAY_TYPE_SWITCH(arrays[i].DataType(), SDT, {
        SDT* source_ptr = static_cast<SDT*>(NDArrayHelper::GetData(arrays[i]));
        if (arrays[i].IsContiguous() && axis == 0) {
          Assign(target_ptr + i * axis_stride, source_ptr, arg_element_num);
          continue;
        }
        Assign(target_ptr + i * axis_stride,
               source_ptr,
               shrink_target_strides.data(),
               arg_strides,
               arg_shape,
               ndim - 1);
      });
    }
  });
  return ret;
}

static std::unordered_map<Unicode, DLDevice> InitCommonDeviceMapping() {
  constexpr int max_id = 32;
  std::unordered_map<Unicode, DLDevice> str2device;
  str2device.reserve(max_id * 3 + 2);

  constexpr int dev_buf_size = 32;
  char dev_buf[dev_buf_size];
  // cpu
  str2device.emplace(Unicode(U"cpu"), DLDevice{DLDeviceType::kDLCPU, 0});
  str2device.emplace(Unicode(U"cpu:0"), DLDevice{DLDeviceType::kDLCPU, 0});
  for (int i = 0; i < max_id; ++i) {
    auto n = snprintf(dev_buf, dev_buf_size, "gpu:%d", i);
    str2device.emplace(UTF8Decode(dev_buf, n), DLDevice{DLDeviceType::kDLCUDA, i});

    n = snprintf(dev_buf, dev_buf_size, "cuda:%d", i);
    str2device.emplace(UTF8Decode(dev_buf, n), DLDevice{DLDeviceType::kDLCUDA, i});

    n = snprintf(dev_buf, dev_buf_size, "cuda_host:%d", i);
    str2device.emplace(UTF8Decode(dev_buf, n), DLDevice{DLDeviceType::kDLCUDAHost, i});
  }
  return str2device;
}

DLDevice NDArrayHelper::GetDevice(const Unicode& device) {
  static auto str2device = InitCommonDeviceMapping();
  DLDevice ret;
  if (device == U"cpu" || device.empty()) {
    return {DLDeviceType::kDLCPU, 0};
  }
  auto device_it = str2device.find(device);
  if (device_it != str2device.end()) {
    return device_it->second;
  } else {
    auto bin_device = UTF8Encode(device);
    string_view device_view = bin_device.view();
    auto pos = device_view.find_last_of(':');
    if (pos == string_view::npos) {
      MXTHROW << "unsupported device:" << device;
    }
    auto dev_type = DeviceNameToType(device_view.substr(0, pos));
    return {DLDeviceType(dev_type), std::atoi(device_view.substr(pos + 1).data())};
  }
}

Unicode NDArrayHelper::GetDeviceStr(const DLDevice& device) {
  static string_view unk("Unknown");
  auto dev_name = DeviceTypeToName(device.device_id);
  if (dev_name == unk) {
    MXTHROW << "unknown device_type: " << device.device_type << ", device_id: " << device.device_id;
  }
  constexpr int dev_buf_size = 256;
  char device_buf[dev_buf_size];
  auto n = snprintf(device_buf, dev_buf_size, "%s:%d", dev_name, device.device_id);
  if (n < 0 || n >= dev_buf_size) {
    MXTHROW << "unknown device_type: " << device.device_type << ", device_id: " << device.device_id;
  }
  return UTF8Decode(device_buf, n);
}

}  // namespace runtime
}  // namespace matxscript
