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

/*!
 * \file ndarray.cc
 * \brief NDArray container infratructure.
 */
#include <matxscript/runtime/container/ndarray.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

#include <matxscript/runtime/c_runtime_api.h>
#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/itertor_private.h>
#include <matxscript/runtime/container/list_helper.h>
#include <matxscript/runtime/container/ndarray_helper.h>
#include <matxscript/runtime/container/ndarray_private.h>
#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/device_api.h>
#include <matxscript/runtime/dlpack.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/generic/generic_funcs.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/runtime_port.h>
#include <matxscript/runtime/runtime_value.h>
#include "../runtime_base.h"

extern "C" {
// C-mangled dlpack deleter.
static void MATXScriptNDArrayDLPackDeleter(DLManagedTensor* tensor);
// helper function to get NDArray's type index, only used by ctypes.
MATX_DLL int MATXScriptArrayGetTypeIndex(MATXScriptTensorHandle handle, unsigned* out_tindex);
MATX_DLL int MATXScriptGetDLTensor(::matxscript::runtime::NDArray::Container* handle,
                                   MATXScriptTensorHandle* out);
MATX_DLL int MATXScriptNDArrayAlloc(const matx_script_index_t* shape,
                                    int ndim,
                                    int dtype_code,
                                    int dtype_bits,
                                    int dtype_lanes,
                                    int device_type,
                                    int device_id,
                                    void** out);
}

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Generic NDArray Iterator
 *****************************************************************************/

class NDArrayIteratorNode : public IteratorNode {
 public:
  explicit NDArrayIteratorNode(NDArray container) {
    container_ = std::move(container);
    if (container_.defined() && container_->ndim > 0) {
      pos_ = 0;
      end_ = container_->shape[0];
    } else {
      pos_ = 0;
      end_ = 0;
    }
  }
  ~NDArrayIteratorNode() = default;

  bool HasNext() const override {
    return pos_ != end_;
  }
  RTValue Next() override {
    return container_.get_item(pos_++);
  }
  RTValue Next(bool* has_next) override {
    auto ret = container_.get_item(pos_++);
    *has_next = (pos_ != end_);
    return ret;
  }
  RTView NextView(bool* has_next, RTValue* holder_or_null) override {
    *holder_or_null = container_.get_item(pos_++);
    *has_next = (pos_ != end_);
    return *holder_or_null;
  }
  int64_t Distance() const override {
    return end_ - pos_;
  }

  uint64_t HashCode() const override {
    return reinterpret_cast<uint64_t>(container_.get());
  }

 public:
  NDArray container_;
  int64_t pos_ = 0;
  int64_t end_ = 0;
};

// iterators
Iterator NDArray::iter() const {
  auto data = make_object<NDArrayIteratorNode>(*this);
  return Iterator(std::move(data));
}

namespace {

template <typename T>
MATXSCRIPT_ALWAYS_INLINE RTValue ElementData2AnyValue(const T& d) {
  return RTValue(d);
}

template <>
MATXSCRIPT_ALWAYS_INLINE RTValue ElementData2AnyValue(const Half& d) {
  return RTValue(static_cast<float>(d));
}

}  // namespace

size_t GetDataSize(const DLTensor& arr) {
  size_t size = 1;
  for (matx_script_index_t i = 0; i < arr.ndim; ++i) {
    size *= static_cast<size_t>(arr.shape[i]);
  }
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

template <typename Type>
bool IsContiguous(const Type& shape, const Type& strides, int ndim) {
  int64_t expected_stride = 1;
  for (int64_t i = ndim; i != 0; --i) {
    int64_t k = i - 1;
    if (strides[k] != expected_stride) {
      return false;
    }
    expected_stride *= shape[k];
  }

  return true;
}

bool IsContiguous(const DLTensor& arr) {
  if (arr.strides == nullptr)
    return true;

  return IsContiguous(arr.shape, arr.strides, arr.ndim);
}

template <typename T>
bool flat_to_1d_imp(const List& data,
                    std::vector<T>& flat_list,
                    std::vector<int64_t>& shape,
                    int depth,
                    int& max_depth) {
  if (data.empty()) {
    return false;
  }
  if (shape.size() == depth) {
    shape.push_back(data.size());
  } else {
    if (data.size() != shape[depth]) {
      return false;
    }
  }
  for (const auto& e : data) {
    if (e.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeList) {
      if (!flat_to_1d_imp(e, flat_list, shape, depth + 1, max_depth)) {
        return false;
      }
    } else {
      if (max_depth == 0) {
        max_depth = depth;
      } else if (depth != max_depth) {
        return false;
      }
      flat_list.push_back(e);
    }
  }
  return true;
}

template <typename T>
bool flat_to_1d(const List& data, std::vector<T>& flat_list, std::vector<int64_t>& shape) {
  int max_depth = 0;
  return flat_to_1d_imp(data, flat_list, shape, 0, max_depth);
}

bool NDArray::operator==(const NDArray& other) const {
  auto* lhs = static_cast<Container*>(data_.get());
  auto* rhs = static_cast<Container*>(other.data_.get());
  if (lhs == rhs)
    return true;

  auto ldt = lhs->dl_tensor.dtype;
  auto rdt = rhs->dl_tensor.dtype;
  MXCHECK_EQ(lhs->dl_tensor.device.device_type, kDLCPU) << "can only compare CPU tensor";
  MXCHECK_EQ(rhs->dl_tensor.device.device_type, kDLCPU) << "can only compare CPU tensor";
  MXCHECK(::matxscript::runtime::IsContiguous(lhs->dl_tensor))
      << "Can only compare contiguous tensor";
  MXCHECK(::matxscript::runtime::IsContiguous(rhs->dl_tensor))
      << "Can only compare contiguous tensor";

  if (lhs->dl_tensor.ndim != rhs->dl_tensor.ndim)
    return false;
  for (int i = 0; i < lhs->dl_tensor.ndim; ++i) {
    if (lhs->dl_tensor.shape[i] != rhs->dl_tensor.shape[i])
      return false;
  }
  if (ldt.code == rdt.code && ldt.lanes == rdt.lanes && ldt.bits == rdt.bits) {
    size_t data_size = GetDataSize(lhs->dl_tensor);
    return std::memcmp(lhs->dl_tensor.data, rhs->dl_tensor.data, data_size) == 0;
  } else {
    return false;
  }
}

bool NDArray::IsContiguous() const {
  return ::matxscript::runtime::IsContiguous(get_mutable()->dl_tensor);
}

void NDArray::CopyFrom(const DLTensor* other) {
  MXCHECK(data_ != nullptr);
  CopyFromTo(other, &(get_mutable()->dl_tensor));
}

void NDArray::CopyFrom(const NDArray& other) {
  MXCHECK(data_ != nullptr);
  MXCHECK(other.data_ != nullptr);
  CopyFromTo(&(other.get_mutable()->dl_tensor), &(get_mutable()->dl_tensor));
}

void NDArray::CopyTo(DLTensor* other) const {
  MXCHECK(data_ != nullptr);
  CopyFromTo(&(get_mutable()->dl_tensor), other);
}

void NDArray::CopyTo(const NDArray& other) const {
  MXCHECK(data_ != nullptr);
  MXCHECK(other.data_ != nullptr);
  CopyFromTo(&(get_mutable()->dl_tensor), &(other.get_mutable()->dl_tensor));
}

NDArray NDArray::CopyTo(const DLDevice& device) const {
  MXCHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  NDArray ret =
      Empty(std::vector<int64_t>(dptr->shape, dptr->shape + dptr->ndim), dptr->dtype, device);
  this->CopyTo(ret);
  return ret;
}

int NDArray::use_count() const {
  return data_.use_count();
}

const DLTensor* NDArray::operator->() const {
  return &(get_mutable()->dl_tensor);
}

NDArray::Container* NDArray::get_mutable() const {
  return static_cast<NDArray::Container*>(data_.get());
}

ObjectPtr<Object> NDArray::FFIDataFromHandle(MATXScriptTensorHandle handle) {
  return GetObjectPtr<Object>(
      static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle)));
}

MATXScriptTensorHandle NDArray::FFIGetHandle(const ObjectRef& nd) {
  // NOTE: it is necessary to cast to container then to base
  //       so that the FFI handle uses the ContainerBase address.
  return reinterpret_cast<MATXScriptTensorHandle>(static_cast<NDArray::ContainerBase*>(
      static_cast<NDArray::Container*>(const_cast<Object*>(nd.get()))));
}

void NDArray::FFIDecRef(MATXScriptTensorHandle handle) {
  static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle))->DecRef();
}

Object* TVMArrayHandleToObjectHandle(MATXScriptTensorHandle handle) {
  return static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle));
}

template <>
RTValue::RTValue(NDArray val) noexcept {
  value_.code = TypeIndex::kRuntimeNDArray;
  value_.data.v_handle = val.data_.data_;
  val.data_.data_ = nullptr;
}

template <>
bool IsConvertible<NDArray>(const Object* node) {
  return node ? node->IsInstance<NDArray::ContainerType>() : NDArray::_type_is_nullable;
}

inline void VerifyDataType(DLDataType dtype) {
  MXCHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    MXCHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt)
      return;
    // allow int1/uint4/int4
    else if (dtype.bits == 1 && dtype.code == kDLInt)
      return;
    else if (dtype.bits == 4 && dtype.code == kDLUInt)
      return;
    else if (dtype.bits == 4 && dtype.code == kDLInt)
      return;
    else
      MXCHECK_EQ(dtype.bits % 8, 0);
  }
  MXCHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment)
    return kAllocAlignment;
  return align;
}

void ArrayCopyFromBytes(DLTensor* handle, const void* data, size_t nbytes) {
  size_t arr_size = GetDataSize(*handle);
  MXCHECK(IsContiguous(*handle)) << "ArrayCopyFromBytes only support contiguous array for now";
  MXCHECK_EQ(arr_size, nbytes) << "ArrayCopyFromBytes: size mismatch";
  DLDevice cpu_dev{kDLCPU, 0};
  auto* device_api = DeviceAPI::Get(handle->device);
  auto stream = device_api->GetCurrentThreadStream(handle->device);
  device_api->CopyDataFromTo(data,
                             0,
                             handle->data,
                             static_cast<size_t>(handle->byte_offset),
                             nbytes,
                             cpu_dev,
                             handle->device,
                             handle->dtype,
                             stream);
  // Synchronize in case data become unavailable later.
  device_api->CreateEventSync(stream);
}

void ArrayCopyToBytes(const DLTensor* handle, void* data, size_t nbytes) {
  DLDevice cpu_dev{kDLCPU, 0};
  size_t arr_size = GetDataSize(*handle);
  MXCHECK(IsContiguous(*handle)) << "ArrayCopyToBytes only support contiguous array for now";
  MXCHECK_EQ(arr_size, nbytes) << "ArrayCopyToBytes: size mismatch";

  auto* device_api = DeviceAPI::Get(handle->device);
  auto stream = device_api->GetCurrentThreadStream(handle->device);

  device_api->CopyDataFromTo(handle->data,
                             static_cast<size_t>(handle->byte_offset),
                             data,
                             0,
                             nbytes,
                             handle->device,
                             cpu_dev,
                             handle->dtype,
                             stream);
  // Synchronize in case data become unavailable later.
  device_api->CreateEventSync(stream);
}

namespace {
template <typename DType>
List ToListImpl(int64_t ndim, DType* data, const int64_t* shape, const int64_t* strides) {
  List ret;
  if (ndim <= 0) {
    return ret;
  }
  ret.reserve(shape[0]);
  if (ndim == 1) {
    for (int64_t i = 0; i < shape[0]; ++i) {
      ret.push_back(ElementData2AnyValue(data[i * strides[0]]));
    }
  } else {
    for (int64_t i = 0; i < shape[0]; ++i) {
      ret.push_back(ToListImpl(ndim - 1, data + i * strides[0], shape + 1, strides + 1));
    }
  }
  return ret;
}

}  // namespace

List NDArray::ToList() const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  MXCHECK(dl_tensor->device.device_type == kDLCPU) << "Only CPU NDArray supports ToList method.";
  int64_t ndim = dl_tensor->ndim;
  const int64_t* shape = dl_tensor->shape;
  const int64_t* strides = get_mutable()->StridesBegin();
  void* p = static_cast<void*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
  List ret;
  MATX_NDARRAY_TYPE_SWITCH(
      dl_tensor->dtype, DT, { return ToListImpl(ndim, static_cast<DT*>(p), shape, strides); });
  return ret;
}

struct NDArray::Internal {
  // Default deleter for the container
  static void DefaultDeleter(Object* ptr_obj) {
    auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
    if (ptr->manager_ctx != nullptr) {
      static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
    } else if (ptr->dl_tensor.data != nullptr) {
      ::matxscript::runtime::DeviceAPI::Get(ptr->dl_tensor.device)
          ->Free(ptr->dl_tensor.device, ptr->dl_tensor.data);
    }
    delete ptr;
  }
  // Deleter for NDArray converted from DLPack
  // This is used from data which is passed from external DLPack(DLManagedTensor)
  // that are not allocated inside of TVM.
  // This enables us to create NDArray from memory allocated by other
  // frameworks that are DLPack compatible
  static void DLPackDeleter(Object* ptr_obj) {
    auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete ptr;
  }
  // Local create function which allocates tensor metadata
  // but does not allocate space for the data.
  static NDArray Create(std::vector<int64_t> shape,
                        std::vector<int64_t> strides,
                        DLDataType dtype,
                        DLDevice device,
                        bool contiguous = true) {
    VerifyDataType(dtype);

    // critical zone: construct header
    NDArray::Container* data = new NDArray::Container();
    data->ref_counter_ = 0;
    data->SetDeleter(DefaultDeleter);

    // RAII now in effect
    NDArray ret(GetObjectPtr<Object>(data));
    // setup shape
    data->shape_ = std::move(shape);
    data->dl_tensor.shape = ::matxscript::runtime::BeginPtr(data->shape_);
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup strides
    data->strides_ = std::move(strides);
    // setup dtype
    data->dl_tensor.dtype = dtype;
    // setup device
    data->dl_tensor.device = device;
    if (!contiguous) {
      data->dl_tensor.strides = ::matxscript::runtime::BeginPtr(data->strides_);
    }
    return ret;
  }

  static NDArray Create(
      const int64_t* shape, int ndim, const int64_t* strides, DLDataType dtype, DLDevice device) {
    VerifyDataType(dtype);

    // critical zone: construct header
    NDArray::Container* data = new NDArray::Container();
    data->ref_counter_ = 0;
    data->SetDeleter(DefaultDeleter);

    // RAII now in effect
    NDArray ret(GetObjectPtr<Object>(data));
    // setup shape
    data->shape_.resize(ndim);
    data->shape_.assign(shape, shape + ndim);
    data->dl_tensor.shape = ::matxscript::runtime::BeginPtr(data->shape_);
    data->dl_tensor.ndim = ndim;
    // setup strides
    if (strides == nullptr) {
      data->strides_ = GenStridesFromShape(data->shape_);
    } else {
      data->strides_.resize(ndim);
      data->strides_.assign(strides, strides + ndim);
      if (!::matxscript::runtime::IsContiguous(shape, strides, ndim)) {
        data->dl_tensor.strides = ::matxscript::runtime::BeginPtr(data->strides_);
      }
    }
    // setup dtype
    data->dl_tensor.dtype = dtype;
    // setup device
    data->dl_tensor.device = device;
    return ret;
  }

  static NDArray FromDLPack(DLManagedTensor* tensor) {
    NDArray::Container* data = new NDArray::Container();
    data->ref_counter_ = 0;
    // construct header
    data->SetDeleter(Internal::DLPackDeleter);
    // fill up content.
    data->manager_ctx = tensor;
    data->dl_tensor = tensor->dl_tensor;
    // update shape_
    data->shape_.resize(data->dl_tensor.ndim);
    data->shape_.assign(data->dl_tensor.shape, data->dl_tensor.shape + data->dl_tensor.ndim);
    data->dl_tensor.shape = ::matxscript::runtime::BeginPtr(data->shape_);
    // update strides_
    if (data->dl_tensor.strides == nullptr) {
      data->strides_ = GenStridesFromShape(data->shape_);
    } else {
      data->strides_.resize(data->dl_tensor.ndim);
      data->strides_.assign(data->dl_tensor.strides,
                            data->dl_tensor.strides + data->dl_tensor.ndim);
      data->dl_tensor.strides = ::matxscript::runtime::BeginPtr(data->strides_);
    }
    return NDArray(GetObjectPtr<Object>(data));
  }

  // Implementation of API function
  static DLTensor* MoveToFFIHandle(NDArray arr) {
    DLTensor* handle = NDArray::FFIGetHandle(arr);
    ObjectRef::FFIClearAfterMove(&arr);
    return handle;
  }
  static void FFIDecRef(MATXScriptTensorHandle tensor) {
    NDArray::FFIDecRef(tensor);
  }
  // Container to DLManagedTensor
  static DLManagedTensor* ToDLPack(MATXScriptTensorHandle handle) {
    auto* from =
        static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle));
    return ToDLPack(from);
  }

  static DLManagedTensor* ToDLPack(NDArray::Container* from) {
    MXCHECK(from != nullptr);
    DLManagedTensor* ret = new DLManagedTensor();
    ret->dl_tensor = from->dl_tensor;
    ret->manager_ctx = from;
    from->IncRef();
    ret->deleter = MATXScriptNDArrayDLPackDeleter;
    return ret;
  }
  // Delete dlpack object.
  static void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
    static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
    delete tensor;
  }
};

NDArray NDArray::Reshape(std::vector<int64_t> newshape) const {
  MXCHECK(IsContiguous()) << "only support contiguous ndarray";
  auto curr_shape = Shape();
  size_t curr_size = 1;
  for (size_t i = 0; i < curr_shape.size(); i++) {
    curr_size *= curr_shape[i];
  }
  size_t given_size = 1;
  int64_t newaxis = -1;
  bool has_zero = false;
  for (size_t i = 0; i < newshape.size(); i++) {
    if (newshape[i] < 0) {
      MXCHECK(newaxis == -1) << "ValueError: can only specify one unknown dimension";
      newaxis = i;
      continue;
    }
    given_size *= newshape[i];
    has_zero = has_zero || (newshape[i] == 0);
  }

  MXCHECK(!(newaxis == -1 && given_size != curr_size))
      << "cannot reshape array of size " << curr_size << " into the given shape";
  MXCHECK(!(has_zero && newaxis != -1))
      << "cannot reshape array of size " << curr_size << " into the given shape";

  if (newaxis != -1) {
    newshape[newaxis] = curr_size / given_size;
  }
  return CreateView(std::move(newshape), (*this)->dtype);
}

NDArray NDArray::Reshape(const FTList<int64_t>& newshape) const {
  std::vector<int64_t> shape;
  for (auto& e : newshape) {
    shape.push_back(e);
  }
  return Reshape(std::move(shape));
}

NDArray NDArray::Reshape(const List& newshape) const {
  std::vector<int64_t> shape;
  for (auto& e : newshape) {
    shape.push_back(e.As<int64_t>());
  }
  return Reshape(std::move(shape));
}

NDArray NDArray::Reshape(const Tuple& newshape) const {
  std::vector<int64_t> shape;
  for (auto& e : newshape) {
    shape.push_back(e.As<int64_t>());
  }
  return Reshape(std::move(shape));
}

NDArray NDArray::Reshape(const Any& newshape) const {
  switch (newshape.type_code()) {
    case TypeIndex::kRuntimeList: {
      auto it = newshape.AsObjectRefNoCheck<List>();
      return Reshape(it);
    } break;
    case TypeIndex::kRuntimeFTList: {
      auto it = newshape.As<FTList<int64_t>>();
      return Reshape(it);
    } break;
    case TypeIndex::kRuntimeTuple: {
      auto it = newshape.AsObjectRefNoCheck<Tuple>();
      return Reshape(it);
    } break;
    default: {
      MXTHROW << "expect 'list' but get '" << TypeIndex2Str(newshape.type_code());
    } break;
  }
  return NDArray();
}

NDArray NDArray::CreateView(std::vector<int64_t> shape, DLDataType dtype) const {
  MXCHECK(data_ != nullptr);
  MXCHECK(get_mutable()->dl_tensor.strides == nullptr) << "Can only create view for compact tensor";
  auto strides = GenStridesFromShape(shape);
  NDArray ret = Internal::Create(
      std::move(shape), std::move(strides), dtype, get_mutable()->dl_tensor.device);
  ret.get_mutable()->dl_tensor.byte_offset = this->get_mutable()->dl_tensor.byte_offset;
  size_t curr_size = GetDataSize(this->get_mutable()->dl_tensor);
  size_t view_size = GetDataSize(ret.get_mutable()->dl_tensor);
  MXCHECK_LE(view_size, curr_size)
      << "Tries to create a view that has bigger memory than current one";
  // increase ref count
  get_mutable()->IncRef();
  ret.get_mutable()->manager_ctx = get_mutable();
  ret.get_mutable()->dl_tensor.data = get_mutable()->dl_tensor.data;
  return ret;
}

NDArray NDArray::CreateViewWithStrides(std::vector<int64_t> shape,
                                       std::vector<int64_t> strides,
                                       DLDataType dtype) const {
  MXCHECK(data_ != nullptr);
  bool contiguous = ::matxscript::runtime::IsContiguous(shape, strides, shape.size());
  NDArray ret = Internal::Create(
      std::move(shape), std::move(strides), dtype, get_mutable()->dl_tensor.device, contiguous);
  Container* ret_container = ret.get_mutable();
  Container* this_container = this->get_mutable();
  ret_container->dl_tensor.byte_offset = this_container->dl_tensor.byte_offset;
  size_t curr_size = GetDataSize(this_container->dl_tensor);
  size_t view_size = GetDataSize(ret_container->dl_tensor);
  MXCHECK_LE(view_size, curr_size)
      << "Tries to create a view that has bigger memory than current one";
  // TODO: check dot(view.shape, view.strides) <= dot(self.shape, self.strides)
  // increase ref count
  this_container->IncRef();
  ret_container->manager_ctx = this_container;
  ret_container->dl_tensor.data = this_container->dl_tensor.data;
  return ret;
}

DLManagedTensor* NDArray::ToDLPack() const {
  return Internal::ToDLPack(get_mutable());
}

std::vector<int64_t> NDArray::GenStridesFromShape(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size());
  if (shape.empty()) {
    return strides;
  }
  strides.back() = 1;
  int64_t ndim = shape.size();
  for (auto i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

NDArray NDArray::Empty(std::vector<int64_t> shape, DLDataType dtype, DLDevice device) {
  auto strides = GenStridesFromShape(shape);
  NDArray ret = Internal::Create(std::move(shape), std::move(strides), dtype, device);
  // setup memory content
  size_t size = GetDataSize(ret.get_mutable()->dl_tensor);
  size_t alignment = GetDataAlignment(ret.get_mutable()->dl_tensor);
  ret.get_mutable()->dl_tensor.data =
      DeviceAPI::Get(ret->device)->Alloc(ret->device, size, alignment, ret->dtype);
  return ret;
}

NDArray NDArray::Empty(const int64_t* shape, int64_t dim, DLDataType dtype, DLDevice device) {
  NDArray ret = Internal::Create(shape, dim, nullptr, dtype, device);
  // setup memory content
  size_t size = GetDataSize(ret.get_mutable()->dl_tensor);
  size_t alignment = GetDataAlignment(ret.get_mutable()->dl_tensor);
  ret.get_mutable()->dl_tensor.data =
      DeviceAPI::Get(ret->device)->Alloc(ret->device, size, alignment, ret->dtype);
  return ret;
}

NDArray NDArray::FromDLPack(DLManagedTensor* tensor) {
  return Internal::FromDLPack(tensor);
}

void NDArray::CopyToBytes(void* data, size_t nbytes) const {
  MXCHECK(data != nullptr);
  MXCHECK(data_ != nullptr);
  ArrayCopyToBytes(&get_mutable()->dl_tensor, data, nbytes);
}

void NDArray::CopyFromBytes(const void* data, size_t nbytes) {
  MXCHECK(data != nullptr);
  MXCHECK(data_ != nullptr);
  ArrayCopyFromBytes(&get_mutable()->dl_tensor, data, nbytes);
}

void NDArray::CopyFromTo(const DLTensor* from, DLTensor* to, MATXScriptStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  MXCHECK_EQ(from_size, to_size) << "MATXScriptArrayCopyFromTo: The size must exactly match";

  MXCHECK(from->device.device_type == to->device.device_type ||
          from->device.device_type == kDLCPU || to->device.device_type == kDLCPU ||
          from->device.device_type == kDLCUDAHost || to->device.device_type == kDLCUDAHost)
      << "Can not copy across different device types directly";

  // Use the context that is *not* a cpu device to get the correct device
  // api manager.
  DLDevice device = from->device.device_type != kDLCPU ? from->device : to->device;

  DeviceAPI::Get(device)->CopyDataFromTo(from->data,
                                         static_cast<size_t>(from->byte_offset),
                                         to->data,
                                         static_cast<size_t>(to->byte_offset),
                                         from_size,
                                         from->device,
                                         to->device,
                                         from->dtype,
                                         stream);
}

void NDArray::CopyFromTo(const DLTensor* from, DLTensor* to) {
  DLDevice device = from->device.device_type != kDLCPU ? from->device : to->device;
  auto* device_api = DeviceAPI::Get(device);
  auto stream = device_api->GetCurrentThreadStream(device);
  return CopyFromTo(from, to, stream);
}

std::vector<int64_t> NDArray::Shape() const {
  return get_mutable()->ShapeVec();
}
::matxscript::runtime::DataType NDArray::DataType() const {
  return ::matxscript::runtime::DataType(get_mutable()->dl_tensor.dtype);
}

List NDArray::ShapeList() const {
  return List(get_mutable()->ShapeBegin(), get_mutable()->ShapeEnd());
}

Unicode NDArray::DTypeUnicode() const {
  return DLDataType2String(get_mutable()->dl_tensor.dtype).decode();
}

Unicode NDArray::Device() const {
  return NDArrayHelper::GetDeviceStr(get_mutable()->dl_tensor.device);
}

size_t NDArray::DataSize() const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  return GetDataSize(*dl_tensor);
}

int64_t NDArray::ElementSize() const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  size_t ndim = dl_tensor->ndim;
  int64_t elem_size = 1;
  for (size_t i = 0; i < ndim; ++i) {
    elem_size *= dl_tensor->shape[i];
  }
  return elem_size;
}

const void* NDArray::RawData() const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  void* p = static_cast<void*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
  return p;
}

const int64_t* NDArray::GetStridesPtr() const {
  return get_mutable()->StridesBegin();
}

const int64_t* NDArray::GetShapePtr() const {
  return get_mutable()->ShapeBegin();
}

int NDArray::GetDim() const {
  return get_mutable()->dl_tensor.ndim;
}

RTValue NDArray::get_item(int64_t index) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  int64_t idx = index_correction(index, dl_tensor->shape[0]);
  MXCHECK(0 <= idx && idx < dl_tensor->shape[0])
      << "[NDArray.get_item] index " << index << " is out of bounds for axis 0 with size "
      << dl_tensor->shape[0];
  void* p = static_cast<void*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
  if (dl_tensor->ndim == 1) {
    MXCHECK(dl_tensor->device.device_type == kDLCPU)
        << "[NDArray]: get item from gpu is not supported";
    MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
      return ElementData2AnyValue(static_cast<DT*>(p)[idx * get_mutable()->Strides(0)]);
    });
  } else {
    NDArray ret = Internal::Create(dl_tensor->shape + 1,
                                   dl_tensor->ndim - 1,
                                   get_mutable()->StridesBegin() + 1,
                                   dl_tensor->dtype,
                                   dl_tensor->device);
    get_mutable()->IncRef();
    ret.get_mutable()->dl_tensor.byte_offset = dl_tensor->byte_offset;
    ret.get_mutable()->manager_ctx = get_mutable();
    MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
      ret.get_mutable()->dl_tensor.data =
          static_cast<DT*>(dl_tensor->data) + get_mutable()->Strides(0) * idx;
    });
    return ret;
  }
  return None;
}

RTValue NDArray::get_item(const Any& index) const {
  switch (index.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return NDArray::get_item(index.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeTuple: {
      // TODO: support tuple
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return None;
    } break;
    default: {
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return None;
    } break;
  }
}

int64_t NDArray::get_item_as_int64(int64_t index) const {
  auto* d_ptr = get_mutable();
  const DLTensor* dl_tensor = &(d_ptr->dl_tensor);
  int64_t idx = index_correction(index, dl_tensor->shape[0]);
  MXCHECK(0 <= idx && idx < dl_tensor->shape[0])
      << "[NDArray.get_item] index " << index << " is out of bounds for axis 0 with size "
      << dl_tensor->shape[0];
  MXCHECK(dl_tensor->ndim == 1) << "can not convert ndarray as int type";
  MXCHECK(dl_tensor->device.device_type == kDLCPU)
      << "[NDArray]: get item from gpu is not supported";
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    auto* p = reinterpret_cast<DT*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
    return (int64_t)(p[idx * d_ptr->Strides(0)]);
  });
  // unreachable code
  return 0;
}

double NDArray::get_item_as_double(int64_t index) const {
  auto* d_ptr = get_mutable();
  const DLTensor* dl_tensor = &(d_ptr->dl_tensor);
  int64_t idx = index_correction(index, dl_tensor->shape[0]);
  MXCHECK(0 <= idx && idx < dl_tensor->shape[0])
      << "[NDArray.get_item] index " << index << " is out of bounds for axis 0 with size "
      << dl_tensor->shape[0];
  MXCHECK(dl_tensor->ndim == 1) << "can not convert ndarray as int type";
  MXCHECK(dl_tensor->device.device_type == kDLCPU)
      << "[NDArray]: get item from gpu is not supported";
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    auto* p = reinterpret_cast<DT*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
    return (double)(p[idx * d_ptr->Strides(0)]);
  });
  // unreachable code
  return 0;
}

int64_t NDArray::get_item_as_int64(const Any& index) const {
  switch (index.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return NDArray::get_item_as_int64(index.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeTuple: {
      // TODO: support tuple
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return 0;
    } break;
    default: {
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return 0;
    } break;
  }
}

double NDArray::get_item_as_double(const Any& index) const {
  switch (index.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return NDArray::get_item_as_double(index.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeTuple: {
      // TODO: support tuple
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return 0;
    } break;
    default: {
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return 0;
    } break;
  }
}

RTValue NDArray::fused_get_item(const int64_t* indexes, size_t num_indexes) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  const int64_t* strides = get_mutable()->StridesBegin();
  MXCHECK(num_indexes <= dl_tensor->ndim);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    auto* p = reinterpret_cast<DT*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
    int i = 0;
    for (size_t pos = 0; pos < num_indexes; ++pos) {
      int64_t index = index_correction(indexes[pos], dl_tensor->shape[i]);
      MXCHECK(0 <= index && index < dl_tensor->shape[i])
          << "[NDArray.get_item] index " << index << " is out of bounds for axis " << i
          << " with size " << dl_tensor->shape[i];
      p += strides[i] * index;
      ++i;
    }
    if (dl_tensor->ndim == num_indexes) {
      MXCHECK(dl_tensor->device.device_type == kDLCPU)
          << "[NDArray]: get scalar value is only supported for cpu array, but get "
          << dl_tensor->device.device_type;
      return ElementData2AnyValue(*p);
    } else {
      NDArray ret = Internal::Create(dl_tensor->shape + i,
                                     dl_tensor->ndim - i,
                                     get_mutable()->StridesBegin() + i,
                                     dl_tensor->dtype,
                                     dl_tensor->device);
      get_mutable()->IncRef();
      ret.get_mutable()->dl_tensor.byte_offset = 0;
      ret.get_mutable()->manager_ctx = get_mutable();
      ret.get_mutable()->dl_tensor.data = p;
      return ret;
    }
  });
  return None;
}

RTValue NDArray::fused_get_item(const std::initializer_list<int64_t>& indexes) const {
  return fused_get_item(indexes.begin(), indexes.size());
}

int64_t NDArray::fused_get_item_as_int64(const int64_t* indexes, size_t num_indexes) const {
  auto* d_ptr = get_mutable();
  const DLTensor* dl_tensor = &(d_ptr->dl_tensor);
  const int64_t* strides = d_ptr->StridesBegin();
  MXCHECK(num_indexes == dl_tensor->ndim);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    auto* p = reinterpret_cast<DT*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
    int i = 0;
    for (size_t pos = 0; pos < num_indexes; ++pos) {
      auto index = index_correction(indexes[pos], dl_tensor->shape[i]);
      MXCHECK(0 <= index && index < dl_tensor->shape[i])
          << "[NDArray.get_item] index " << index << " is out of bounds for axis " << i
          << " with size " << dl_tensor->shape[i];
      p += strides[i] * index;
      ++i;
    }
    MXCHECK(dl_tensor->device.device_type == kDLCPU)
        << "[NDArray]: get scalar value is only supported for cpu array, but get "
        << dl_tensor->device.device_type;
    return int64_t(*p);
  });
  return 0;
}

int64_t NDArray::fused_get_item_as_int64(const std::initializer_list<int64_t>& indexes) const {
  return fused_get_item_as_int64(indexes.begin(), indexes.size());
}

double NDArray::fused_get_item_as_double(const int64_t* indexes, size_t num_indexes) const {
  auto* d_ptr = get_mutable();
  const DLTensor* dl_tensor = &(d_ptr->dl_tensor);
  const int64_t* strides = d_ptr->StridesBegin();
  MXCHECK(num_indexes == dl_tensor->ndim);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    auto* p = reinterpret_cast<DT*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
    int i = 0;
    for (size_t pos = 0; pos < num_indexes; ++pos) {
      auto index = index_correction(indexes[pos], dl_tensor->shape[i]);
      MXCHECK(0 <= index && index < dl_tensor->shape[i])
          << "[NDArray.get_item] index " << index << " is out of bounds for axis " << i
          << " with size " << dl_tensor->shape[i];
      p += strides[i] * index;
      ++i;
    }
    MXCHECK(dl_tensor->device.device_type == kDLCPU)
        << "[NDArray]: get scalar value is only supported for cpu array, but get "
        << dl_tensor->device.device_type;
    return double(*p);
  });
  return 0;
}

double NDArray::fused_get_item_as_double(const std::initializer_list<int64_t>& indexes) const {
  return fused_get_item_as_double(indexes.begin(), indexes.size());
}

// general assign
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

// for compact tensors
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

void NDArray::set_item_helper(void* dst_data,
                              const int64_t* dst_shape,
                              const int64_t* dst_strides,
                              int dst_ndim,
                              const Any& item) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  if (item.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeNDArray) {
    NDArray src = item.AsObjectRefNoCheck<NDArray>();
    const DLTensor* src_tensor = &(src.get_mutable()->dl_tensor);
    MXCHECK(dst_ndim == src_tensor->ndim) << "[NDArray::set_item_helper] dimensional mismatch";
    MXCHECK(std::equal(dst_shape, dst_shape + dst_ndim, src_tensor->shape))
        << "[NDArray::set_item_helper] shape mismatch";
    const auto& src_strides = src.get_mutable()->StridesBegin();
    void* src_data =
        static_cast<void*>(static_cast<char*>(src_tensor->data) + src_tensor->byte_offset);
    MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
      MATX_NDARRAY_TYPE_SWITCH(src_tensor->dtype, SRC_DT, {
        if (::matxscript::runtime::IsContiguous(dst_shape, dst_strides, dst_ndim) &&
            src.IsContiguous()) {
          Assign(static_cast<DT*>(dst_data),
                 static_cast<SRC_DT*>(src_data),
                 dst_shape[0] * dst_strides[0]);
        } else {
          Assign(static_cast<DT*>(dst_data),
                 static_cast<SRC_DT*>(src_data),
                 dst_strides,
                 src_strides,
                 dst_shape,
                 dst_ndim);
        }
      });
    });
  } else if (item.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeList) {
    List src_list = item.AsObjectRefNoCheck<List>();
    MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
      std::vector<int64_t> shape;
      auto data = ListHelper::IsNDArray<DT>(src_list, shape);
      MXCHECK(data != nullptr) << "[NDArray::set_item] item shape is invalid";
      int ndim = shape.size();
      MXCHECK(dst_ndim == ndim) << "[NDArray::set_item] dimensional mismatch";
      MXCHECK(std::equal(dst_shape, dst_shape + dst_ndim, shape.begin()))
          << "[NDArray::set_item] shape mismatch";
      std::vector<int64_t> strides = GenStridesFromShape(shape);
      if (::matxscript::runtime::IsContiguous(dst_shape, dst_strides, dst_ndim)) {
        Assign(static_cast<DT*>(dst_data), data->data(), dst_shape[0] * dst_strides[0]);
      } else {
        Assign(static_cast<DT*>(dst_data),
               data->data(),
               dst_strides,
               ::matxscript::runtime::BeginPtr(strides),
               dst_shape,
               dst_ndim);
      }
    });
  } else {
    MXTHROW << "unsupported item type, type_code" << item.type_code();
  }
}

void NDArray::set_item(int64_t index, int64_t value) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  const int64_t* strides = get_mutable()->StridesBegin();
  // TODO: fix broadcast
  MXCHECK(dl_tensor->ndim == 1);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    DT* p = reinterpret_cast<DT*>(static_cast<unsigned char*>(dl_tensor->data) +
                                  dl_tensor->byte_offset);
    index = index_correction(index, dl_tensor->shape[0]);
    MXCHECK(0 <= index && index < dl_tensor->shape[0])
        << "[NDArray.set_item] index " << index << " is out of bounds for axis " << 0
        << " with size " << dl_tensor->shape[0];
    p += strides[0] * index;
    // TODO: fix broadcast
    *p = (DT)(value);
  });
}

void NDArray::set_item(int64_t index, double value) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  const int64_t* strides = get_mutable()->StridesBegin();
  // TODO: fix broadcast
  MXCHECK(dl_tensor->ndim == 1);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    DT* p = reinterpret_cast<DT*>(static_cast<unsigned char*>(dl_tensor->data) +
                                  dl_tensor->byte_offset);
    index = index_correction(index, dl_tensor->shape[0]);
    MXCHECK(0 <= index && index < dl_tensor->shape[0])
        << "[NDArray.set_item] index " << index << " is out of bounds for axis " << 0
        << " with size " << dl_tensor->shape[0];
    p += strides[0] * index;
    // TODO: fix broadcast
    *p = (DT)(value);
  });
}

void NDArray::set_item(int64_t index, const Any& item) const {
  switch (item.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      set_item(index, item.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      set_item(index, item.AsNoCheck<double>());
    } break;
    default: {
      // TODO: optimize set_item
      const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
      const int64_t* strides = get_mutable()->StridesBegin();
      void* dst_data = nullptr;
      MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
        DT* p = reinterpret_cast<DT*>(static_cast<unsigned char*>(dl_tensor->data) +
                                      dl_tensor->byte_offset);
        index = index_correction(index, dl_tensor->shape[0]);
        MXCHECK(0 <= index && index < dl_tensor->shape[0])
            << "[NDArray.set_item] index " << index << " is out of bounds for axis " << 0
            << " with size " << dl_tensor->shape[0];
        p += strides[0] * index;
        dst_data = p;
      });
      // TODO: fix broadcast
      set_item_helper(dst_data, dl_tensor->shape + 1, strides + 1, dl_tensor->ndim - 1, item);
    } break;
  }
}

void NDArray::set_item(const Any& index, int64_t value) const {
  switch (index.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return NDArray::set_item(index.AsNoCheck<int64_t>(), value);
    } break;
    case TypeIndex::kRuntimeTuple: {
      // TODO: support tuple
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return;
    } break;
    default: {
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return;
    } break;
  }
}

void NDArray::set_item(const Any& index, double value) const {
  switch (index.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return NDArray::set_item(index.AsNoCheck<int64_t>(), value);
    } break;
    case TypeIndex::kRuntimeTuple: {
      // TODO: support tuple
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return;
    } break;
    default: {
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return;
    } break;
  }
}

void NDArray::set_item(const Any& index, const Any& item) const {
  switch (index.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      return NDArray::set_item(index.AsNoCheck<int64_t>(), item);
    } break;
    case TypeIndex::kRuntimeTuple: {
      // TODO: support tuple
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return;
    } break;
    default: {
      MXTHROW << "unsupported index type, type_code" << index.type_code();
      return;
    } break;
  }
}

void NDArray::fused_set_item(const int64_t* indexes, size_t num_indexes, int64_t value) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  const int64_t* strides = get_mutable()->StridesBegin();
  // TODO: fix broadcast
  MXCHECK(num_indexes == dl_tensor->ndim);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    DT* p = reinterpret_cast<DT*>(static_cast<unsigned char*>(dl_tensor->data) +
                                  dl_tensor->byte_offset);
    int i = 0;
    for (size_t pos = 0; pos < num_indexes; ++pos) {
      auto index = index_correction(indexes[pos], dl_tensor->shape[i]);
      MXCHECK(0 <= index && index < dl_tensor->shape[i])
          << "[NDArray.set_item] index " << index << " is out of bounds for axis " << i
          << " with size " << dl_tensor->shape[i];
      p += strides[i++] * index;
    }
    // TODO: fix broadcast
    *p = (DT)value;
  });
}

void NDArray::fused_set_item(const std::initializer_list<int64_t>& indexes, int64_t value) const {
  return fused_set_item(indexes.begin(), indexes.size(), value);
}

void NDArray::fused_set_item(const int64_t* indexes, size_t num_indexes, double value) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  const int64_t* strides = get_mutable()->StridesBegin();
  // TODO: fix broadcast
  MXCHECK(num_indexes == dl_tensor->ndim);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    DT* p = reinterpret_cast<DT*>(static_cast<unsigned char*>(dl_tensor->data) +
                                  dl_tensor->byte_offset);
    int i = 0;
    for (size_t pos = 0; pos < num_indexes; ++pos) {
      auto index = index_correction(indexes[pos], dl_tensor->shape[i]);
      MXCHECK(0 <= index && index < dl_tensor->shape[i])
          << "[NDArray.set_item] index " << index << " is out of bounds for axis " << i
          << " with size " << dl_tensor->shape[i];
      p += strides[i++] * index;
    }
    // TODO: fix broadcast
    *p = (DT)value;
  });
}

void NDArray::fused_set_item(const std::initializer_list<int64_t>& indexes, double value) const {
  return fused_set_item(indexes.begin(), indexes.size(), value);
}

void NDArray::fused_set_item(const int64_t* indexes, size_t num_indexes, const Any& item) const {
  switch (item.type_code()) {
    case TypeIndex::kRuntimeInteger: {
      fused_set_item(indexes, num_indexes, item.AsNoCheck<int64_t>());
    } break;
    case TypeIndex::kRuntimeFloat: {
      fused_set_item(indexes, num_indexes, item.AsNoCheck<double>());
    } break;
    default: {
      // TODO: optimize set_item
      const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
      const int64_t* strides = get_mutable()->StridesBegin();
      void* dst_data = nullptr;
      MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
        DT* p = reinterpret_cast<DT*>(static_cast<unsigned char*>(dl_tensor->data) +
                                      dl_tensor->byte_offset);
        int i = 0;
        for (size_t pos = 0; pos < num_indexes; ++pos) {
          auto index = index_correction(indexes[pos], dl_tensor->shape[i]);
          MXCHECK(0 <= index && index < dl_tensor->shape[i])
              << "[NDArray.set_item] index " << index << " is out of bounds for axis " << i
              << " with size " << dl_tensor->shape[i];
          p += strides[i++] * index;
        }
        dst_data = p;
      });
      // TODO: fix broadcast
      set_item_helper(dst_data,
                      dl_tensor->shape + num_indexes,
                      strides + num_indexes,
                      dl_tensor->ndim - num_indexes,
                      item);
      // THROW_PY_ValueError("setting an array element with a ", item.type_name(), '.');
    } break;
  }
}

void NDArray::fused_set_item(const std::initializer_list<int64_t>& indexes, const Any& item) const {
  return fused_set_item(indexes.begin(), indexes.size(), item);
}

NDArray NDArray::get_slice(int64_t begin, int64_t end, int64_t step) const {
  MXCHECK_GT(step, 0) << "[NDArray::get_slice step must greater than 0";
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  begin = slice_index_correction(begin, dl_tensor->shape[0]);
  end = slice_index_correction(end, dl_tensor->shape[0]);
  if (begin >= end) {
    std::vector<int64_t> shape = get_mutable()->ShapeVec();
    shape[0] = 0;
    NDArray ret =
        Internal::Create(shape, get_mutable()->StridesVec(), dl_tensor->dtype, dl_tensor->device);
    ret.get_mutable()->dl_tensor.data = nullptr;
    return ret;
  }

  std::vector<int64_t> shape = get_mutable()->ShapeVec();
  std::vector<int64_t> strides = get_mutable()->StridesVec();
  shape[0] = (end - begin + step - 1) / step;
  strides[0] *= step;
  bool contiguous = ::matxscript::runtime::IsContiguous(shape, strides, shape.size());
  NDArray ret = Internal::Create(shape, strides, dl_tensor->dtype, dl_tensor->device, contiguous);
  get_mutable()->IncRef();
  ret.get_mutable()->dl_tensor.byte_offset = dl_tensor->byte_offset;
  ret.get_mutable()->manager_ctx = get_mutable();
  void* p = static_cast<void*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
  MATX_NDARRAY_TYPE_SWITCH(dl_tensor->dtype, DT, {
    ret.get_mutable()->dl_tensor.data =
        static_cast<DT*>(dl_tensor->data) + get_mutable()->Strides(0) * begin;
  });
  return ret;
}

void NDArray::set_slice(int64_t begin, int64_t end, const Any& item) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  begin = slice_index_correction(begin, dl_tensor->shape[0]);
  end = slice_index_correction(end, dl_tensor->shape[0]);
  if (begin >= end) {
    return;
  }

  void* p = static_cast<void*>(static_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset);
  void* dst_data = nullptr;
  const int64_t* dst_strides = get_mutable()->StridesBegin();
  MATX_NDARRAY_TYPE_SWITCH(
      dl_tensor->dtype, DT, { dst_data = static_cast<DT*>(p) + dst_strides[0] * begin; });
  int64_t dst_shape[dl_tensor->ndim];
  std::copy(dl_tensor->shape, dl_tensor->shape + dl_tensor->ndim, dst_shape);
  dst_shape[0] = end - begin;
  set_item_helper(dst_data, dst_shape, dst_strides, dl_tensor->ndim, item);
}

int64_t NDArray::size() const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  if (dl_tensor->ndim <= 0) {
    return 0;
  }

  return dl_tensor->shape[0];
}

NDArray NDArray::transpose(const Any& axes) const {
  const DLTensor* dl_tensor = &(get_mutable()->dl_tensor);
  int ndim = dl_tensor->ndim;
  std::vector<int64_t> permutation;
  permutation.reserve(ndim);
  if (axes.is_nullptr()) {
    for (int64_t i = 0; i < ndim; ++i) {
      permutation.push_back(ndim - 1 - i);
    }
  } else if (axes.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeList) {
    const auto& l = axes.AsObjectRefNoCheck<List>();
    MXCHECK(l.size() == ndim) << "axes don't match array";
    for (const auto& e : l) {
      MXCHECK(e.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeInteger)
          << "[NDArray::transpose] axes element must be an integer";
      int64_t axis = index_correction(e.As<int64_t>(), ndim);
      MXCHECK(0 <= axis && axis < ndim) << "[NDArray::transpose] axis  " << axis
                                        << " is out of bounds for array of dimension " << ndim;
      permutation.push_back(axis);
    }
  } else if (axes.type_code() == ::matxscript::runtime::TypeIndex::kRuntimeTuple) {
    const auto& t = axes.AsObjectRefNoCheck<Tuple>();
    MXCHECK(t.size() == ndim) << "axes don't match array";
    for (int i = 0; i < t.size(); ++i) {
      MXCHECK(t[i].type_code() == ::matxscript::runtime::TypeIndex::kRuntimeInteger)
          << "[NDArray::transpose] axes element must be an integer";
      int64_t axis = index_correction(t[i].As<int64_t>(), ndim);
      MXCHECK(0 <= axis && axis < ndim) << "[NDArray::transpose] axis  " << axis
                                        << " is out of bounds for array of dimension " << ndim;
      permutation.push_back(axis);
    }
  } else {
    MXTHROW << "unsupported axes type, type_code: " << axes.type_code();
    return None.As<NDArray>();
  }

  // check unique
  MXCHECK(std::unordered_set<int64_t>(permutation.begin(), permutation.end()).size() == ndim)
      << "repeated axis in axes";

  std::vector<int64_t> shape(ndim);
  std::vector<int64_t> strides(ndim);

  for (int i = 0; i < ndim; ++i) {
    shape[i] = get_mutable()->Shape(permutation[i]);
    strides[i] = get_mutable()->Strides(permutation[i]);
  }

  bool contiguous = ::matxscript::runtime::IsContiguous(shape, strides, shape.size());
  NDArray ret = Internal::Create(shape, strides, dl_tensor->dtype, dl_tensor->device, contiguous);
  get_mutable()->IncRef();
  ret.get_mutable()->dl_tensor.byte_offset = dl_tensor->byte_offset;
  ret.get_mutable()->dl_tensor.data = dl_tensor->data;
  ret.get_mutable()->manager_ctx = get_mutable();
  return ret;
}

NDArray NDArray::as_type(const unicode_view& dtype_str) const {
  check_dtype_valid(dtype_str);

  auto src_container = get_mutable();
  const DLTensor* src_tensor = &(src_container->dl_tensor);
  ::matxscript::runtime::DataType dst_dtype(String2DLDataType(UTF8Encode(dtype_str)));
  auto ret = Empty(src_container->ShapeVec(), dst_dtype, src_tensor->device);
  auto dst_container = ret.get_mutable();
  const DLTensor* dst_tensor = &(dst_container->dl_tensor);
  auto src_data =
      static_cast<void*>(static_cast<char*>(src_tensor->data) + src_tensor->byte_offset);
  auto dst_data =
      static_cast<void*>(static_cast<char*>(dst_tensor->data) + dst_tensor->byte_offset);
  MATX_NDARRAY_TYPE_SWITCH(dst_dtype, DST_DT, {
    MATX_NDARRAY_TYPE_SWITCH(src_tensor->dtype, SRC_DT, {
      if (IsContiguous()) {
        Assign(static_cast<DST_DT*>(dst_data),
               static_cast<SRC_DT*>(src_data),
               src_container->Shape(0) * src_container->Strides(0));
      } else {
        Assign(static_cast<DST_DT*>(dst_data),
               static_cast<SRC_DT*>(src_data),
               dst_container->StridesBegin(),
               src_container->StridesBegin(),
               dst_tensor->shape,
               src_tensor->ndim);
      }
    });
  });

  return ret;
}

void NDArray::AssignNDArray(const NDArray& src, NDArray& dst) {
  auto src_container = src.get_mutable();
  auto dst_container = dst.get_mutable();
  const DLTensor* src_tensor = &(src_container->dl_tensor);
  DLTensor* dst_tensor = &(dst_container->dl_tensor);

  MATX_NDARRAY_TYPE_SWITCH(src_tensor->dtype, SRC_DT, {
    MATX_NDARRAY_TYPE_SWITCH(dst_tensor->dtype, DST_DT, {
      void* src_data = static_cast<char*>(src_tensor->data) + src_tensor->byte_offset;
      void* dst_data = static_cast<char*>(dst_tensor->data) + dst_tensor->byte_offset;
      if (src.IsContiguous() && dst.IsContiguous()) {
        Assign(static_cast<DST_DT*>(dst_data),
               static_cast<SRC_DT*>(src_data),
               src_container->Shape(0) * src_container->Strides(0));
      } else {
        Assign(static_cast<DST_DT*>(dst_data),
               static_cast<SRC_DT*>(src_data),
               dst_container->StridesBegin(),
               src_container->StridesBegin(),
               src_tensor->shape,
               src_tensor->ndim);
      }
    });
  });
}

NDArray NDArray::Contiguous() const {
  if (IsContiguous()) {
    return *this;
  }
  MXCHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  const DLDevice src_dev = dptr->device;
  MXCHECK(src_dev.device_type == kDLCPU);
  auto container = get_mutable();
  std::vector<int64_t> src_shape(std::move(container->ShapeVec()));
  DLDataType src_dtype = dptr->dtype;
  NDArray dst_arr = NDArray::Empty(src_shape, src_dtype, src_dev);
  AssignNDArray(*this, dst_arr);
  return dst_arr;
}

void NDArray::check_dtype_valid(const unicode_view& dtype_str) {
  if (dtype_str != U"int32" && dtype_str != U"int64" && dtype_str != U"float16" &&
      dtype_str != U"float32" && dtype_str != U"float64" && dtype_str != U"uint8" &&
      dtype_str != U"bool" && dtype_str != U"int8" && dtype_str != U"int16" &&
      dtype_str != U"uint16") {
    THROW_PY_ValueError(
        "unsupported ndarray type ",
        dtype_str,
        ". expect ndarray type is Union[int8, int32, int64, uint8, uint16, float16, half, float32, float64]");
  }
}

MATXSCRIPT_REGISTER_OBJECT_TYPE(NDArray::Container);

template <typename DType>
static inline void PrintNDArray(int64_t ndim,
                                DType* data,
                                const int64_t* sizes,
                                const int64_t* strides,
                                std::ostream& ss,
                                int depth = 0,
                                bool need_space = false) {
  char space_small[1024] = {0};
  char* space = space_small;
  size_t space_len = sizeof(space_small);
  if (space_len <= 2 * depth) {
    space_len = 2 * depth + 1;
    space = new char[space_len];
  }
  snprintf(space, space_len, "%*s", 6 + depth, " ");
  if (ndim == 1) {
    for (size_t i = 0; i < sizes[0]; ++i) {
      // treat unit8_t as int(not char)
      if (i > 0) {
        ss << ", ";
      }
      ss << (data[i * strides[0]]);
    }
  } else {
    for (size_t i = 0; i < sizes[0]; ++i) {
      if (i > 0) {
        ss << ",\n";
        if (ndim == 3) {
          ss << "\n";
        }
        need_space = true;
      }
      if (need_space) {
        ss << space;
        need_space = false;
      }
      ss << "[";
      PrintNDArray<DType>(
          ndim - 1, data + strides[0] * i, sizes + 1, strides + 1, ss, depth + 1, need_space);
      ss << "]";
    }
  }
  if (space_len != sizeof(space_small)) {
    delete[] space;
  }
}

static inline void PrintNDArray(const NDArray& tensor, std::ostream& ss, int depth = 0) {
  int64_t strides_buf[8];
  auto dtype = tensor.DataType();
  ss << "<matx.NDArray shape=(";
  auto* shape_ptr = tensor.GetShapePtr();
  for (auto dim_pos = 0; dim_pos < tensor->ndim; ++dim_pos) {
    if (dim_pos > 0) {
      ss << ", ";
    }
    ss << shape_ptr[dim_pos];
  }
  ss << "), ";
  ss << DeviceName(tensor->device.device_type);
  ss << "(" << tensor->device.device_id << ")>\n";
  ss << "array([";
  const int64_t* strides = tensor.GetStridesPtr();
  if (tensor->device.device_type != DLDeviceType::kDLCPU &&
      tensor->device.device_type != DLDeviceType::kDLCUDAHost) {
    int64_t max_bytes = 0;
    for (int i = 0; i < tensor->ndim; ++i) {
      max_bytes += (tensor->shape[i] - 1) * strides[i];
    }
    max_bytes = (max_bytes + 1) * tensor.DataType().bytes();
    DeviceAPI* cpu_device = DeviceAPI::Get(DLDevice{kDLCPU, 0});
    void* to = cpu_device->Alloc(DLDevice{kDLCPU, 0}, max_bytes);
    DeviceAPI* gpu_device = DeviceAPI::Get(tensor->device);
    auto stream = gpu_device->GetCurrentThreadStream(tensor->device);

    gpu_device->CopyDataFromTo(tensor->data,
                               tensor->byte_offset,
                               to,
                               0,
                               max_bytes,
                               tensor->device,
                               DLDevice{kDLCPU, 0},
                               tensor->dtype,
                               stream);
    gpu_device->CreateEventSync(stream);
    MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
      PrintNDArray(tensor->ndim, static_cast<DT*>(to), tensor->shape, strides, ss, depth);
    });
    cpu_device->Free(DLDevice{kDLCPU, 0}, to);
  } else {
    MATX_NDARRAY_TYPE_SWITCH_WITH_BOOL(dtype, DT, {
      PrintNDArray(tensor->ndim, static_cast<DT*>(tensor->data), tensor->shape, strides, ss, depth);
    });
  }
  ss << "], dtype=" << DLDataType2String(tensor->dtype) << ")";
}

std::ostream& operator<<(std::ostream& os, NDArray const& n) {
  PrintNDArray(n, os);
  return os;
}

}  // namespace runtime
}  // namespace matxscript

using namespace ::matxscript::runtime;

void MATXScriptNDArrayDLPackDeleter(DLManagedTensor* tensor) {
  NDArray::Internal::NDArrayDLPackDeleter(tensor);
}

int MATXScriptArrayGetTypeIndex(MATXScriptTensorHandle handle, unsigned* out_tindex) {
  API_BEGIN();
  *out_tindex = TVMArrayHandleToObjectHandle(handle)->type_index();
  API_END();
}

int MATXScriptArrayAlloc(const matx_script_index_t* shape,
                         int ndim,
                         int dtype_code,
                         int dtype_bits,
                         int dtype_lanes,
                         int device_type,
                         int device_id,
                         MATXScriptTensorHandle* out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  DLDevice device;
  device.device_type = static_cast<DLDeviceType>(device_type);
  device.device_id = device_id;
  *out = NDArray::Internal::MoveToFFIHandle(
      NDArray::Empty(std::vector<int64_t>(shape, shape + ndim), dtype, device));
  API_END();
}

int MATXScriptNDArrayAlloc(const matx_script_index_t* shape,
                           int ndim,
                           int dtype_code,
                           int dtype_bits,
                           int dtype_lanes,
                           int device_type,
                           int device_id,
                           void** out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  DLDevice device;
  device.device_type = static_cast<DLDeviceType>(device_type);
  device.device_id = device_id;
  auto nd = NDArray::Empty(std::vector<int64_t>(shape, shape + ndim), dtype, device);
  *out = const_cast<Object*>(nd.get());
  NDArray::Internal::MoveToFFIHandle(nd);
  API_END();
}

int MATXScriptArrayFree(MATXScriptTensorHandle handle) {
  API_BEGIN();
  NDArray::Internal::FFIDecRef(handle);
  API_END();
}

int MATXScriptArrayCopyFromTo(MATXScriptTensorHandle from,
                              MATXScriptTensorHandle to,
                              MATXScriptStreamHandle stream) {
  API_BEGIN();
  NDArray::CopyFromTo(from, to, stream);
  API_END();
}

int MATXScriptArrayFromDLPack(DLManagedTensor* from, MATXScriptTensorHandle* out) {
  API_BEGIN();
  *out = NDArray::Internal::MoveToFFIHandle(NDArray::FromDLPack(from));
  API_END();
}

int MATXScriptGetDLTensor(::matxscript::runtime::NDArray::Container* handle,
                          MATXScriptTensorHandle* out) {
  API_BEGIN();
  *out = reinterpret_cast<MATXScriptTensorHandle>(static_cast<NDArray::ContainerBase*>(handle));
  API_END();
}

int MATXScriptArrayToDLPack(MATXScriptTensorHandle from, DLManagedTensor** out) {
  API_BEGIN();
  *out = NDArray::Internal::ToDLPack(from);
  API_END();
}

void MATXScriptDLManagedTensorCallDeleter(DLManagedTensor* dltensor) {
  (*(dltensor->deleter))(dltensor);
}

int MATXScriptArrayCopyFromBytes(MATXScriptTensorHandle handle, void* data, size_t nbytes) {
  API_BEGIN();
  ArrayCopyFromBytes(handle, data, nbytes);
  API_END();
}

int MATXScriptArrayCopyToBytes(MATXScriptTensorHandle handle, void* data, size_t nbytes) {
  API_BEGIN();
  ArrayCopyToBytes(handle, data, nbytes);
  API_END();
}
