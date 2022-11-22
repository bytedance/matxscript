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
#pragma once

#include "ndarray.h"

namespace matxscript {
namespace runtime {
/*!
 * \brief The container base structure
 *        contains all the fields except for the Object header.
 *
 * \note We explicitly declare this structure in order to pass
 *       PackedFunc argument using ContainerBase*.
 */
class NDArray::ContainerBase {
 public:
  /*!
   * \brief The corresponding dl_tensor field.
   * \note it is important that the first field is DLTensor
   *  So that this data structure is DLTensor compatible.
   *  The head ptr of this struct can be viewed as DLTensor*.
   */
  DLTensor dl_tensor;

  /*!
   * \brief additional context, reserved for recycling
   * \note We can attach additional content here
   *  which the current container depend on
   *  (e.g. reference to original memory when creating views).
   */
  void* manager_ctx{nullptr};

 protected:
  /*!
   * \brief The shape container,
   *  can be used used for shape data.
   */
  std::vector<int64_t> shape_;

  /*!
   * \brief The strides container.
   */
  std::vector<int64_t> strides_;
};

/*!
 * \brief Object container class that backs NDArray.
 * \note do not use this function directly, use NDArray.
 */
class NDArray::Container : public Object, public NDArray::ContainerBase {
 public:
  /*! \brief default constructor */
  Container() {
    // Initialize the type index.
    type_index_ = Container::RuntimeTypeIndex();
    dl_tensor.data = nullptr;
    dl_tensor.ndim = 0;
    dl_tensor.shape = nullptr;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
  }

  Container(void* data, std::vector<int64_t> shape, DLDataType dtype, DLDevice device) {
    // Initialize the type index.
    type_index_ = Container::RuntimeTypeIndex();
    dl_tensor.data = data;
    shape_ = std::move(shape);
    dl_tensor.ndim = static_cast<int>(shape_.size());
    dl_tensor.shape = ::matxscript::runtime::BeginPtr(shape_);
    dl_tensor.dtype = dtype;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
    dl_tensor.device = device;
  }
  /*!
   * \brief Set the deleter field.
   * \param deleter The deleter.
   */
  void SetDeleter(FDeleter deleter) {
    deleter_ = deleter;
  }

  MATXSCRIPT_ALWAYS_INLINE const int64_t* StridesBegin() {
    return strides_.data();
  }

  MATXSCRIPT_ALWAYS_INLINE const int64_t* StridesEnd() {
    return strides_.data() + dl_tensor.ndim;
  }

  MATXSCRIPT_ALWAYS_INLINE const int64_t* ShapeBegin() {
    return shape_.data();
  }

  MATXSCRIPT_ALWAYS_INLINE const int64_t* ShapeEnd() {
    return shape_.data() + dl_tensor.ndim;
  }

  MATXSCRIPT_ALWAYS_INLINE std::vector<int64_t> StridesVec() {
    return strides_;
  }

  MATXSCRIPT_ALWAYS_INLINE std::vector<int64_t> ShapeVec() {
    return shape_;
  }

  MATXSCRIPT_ALWAYS_INLINE int64_t Strides(int i) {
    return strides_[i];
  }

  MATXSCRIPT_ALWAYS_INLINE int64_t Shape(int i) {
    return shape_[i];
  }

  // Expose DecRef and IncRef as public function
  // NOTE: they are only for developer purposes only.
  using Object::DecRef;
  using Object::IncRef;

  // Information for object protocol.
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeNDArray;
  static constexpr const uint32_t _type_child_slots = 0;
  static constexpr const uint32_t _type_child_slots_can_overflow = true;
  static constexpr const char* _type_key = "runtime.NDArray";
  MATXSCRIPT_DECLARE_BASE_OBJECT_INFO(NDArray::Container, Object);

 protected:
  friend class NDArray::Internal;
};

}  // namespace runtime
}  // namespace matxscript
