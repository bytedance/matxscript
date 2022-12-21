// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 *
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

#include <vector>

#include <matxscript/runtime/data_type.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class List;
class Unicode;

/*!
 * \brief Managed NDArray.
 *  The array is backed by reference counted blocks.
 */
class NDArray : public ObjectRef {
 public:
  /*! \brief ContainerBase used to back the MATXScriptArrayHandle */
  class ContainerBase;
  /*! \brief NDArray internal container type */
  class Container;
  /*! \brief Container type for Object system. */
  using ContainerType = Container;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance
  /*! \brief default constructor */
  NDArray() noexcept = default;
  /*!
   * \brief constructor.
   * \param data ObjectPtr to the data container.
   */
  explicit NDArray(ObjectPtr<Object> data) noexcept : ObjectRef(std::move(data)) {
  }

  bool operator==(const NDArray& other) const;

  /*! \brief reset the content of NDArray to be nullptr */
  void reset();
  /*!
   * \return the reference counter
   * \note this number is approximate in multi-threaded setting.
   */
  int use_count() const;
  /*! \return Pointer to content of DLTensor */
  const DLTensor* operator->() const;
  /*! \return Whether the tensor is contiguous */
  bool IsContiguous() const;
  /*!
   * \brief Copy data content from another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU device.
   *       MATXScriptSynchronize is necessary.
   */
  void CopyFrom(const DLTensor* other);
  void CopyFrom(const NDArray& other);
  /*!
   * \brief Copy data content from a byte buffer.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the buffer in bytes
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a MATXScriptSynchronize.
   */
  MATX_DLL void CopyFromBytes(const void* data, size_t nbytes);
  /*!
   * \brief Copy data content into another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU device.
   *       MATXScriptSynchronize is necessary.
   */
  void CopyTo(DLTensor* other) const;
  void CopyTo(const NDArray& other) const;
  /*!
   * \brief Copy data content into another array.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the data buffer.
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a MATXScriptSynchronize.
   */
  MATX_DLL void CopyToBytes(void* data, size_t nbytes) const;
  /*!
   * \brief Copy the data to another device.
   * \param device The target device.
   * \return The array under another device.
   */
  NDArray CopyTo(const DLDevice& device) const;
  /*!
   * \brief get a contiguous copy of current NDArray.
   * \return a contiguous copy of current NDArray.
   */
  NDArray Contiguous() const;

  NDArray Reshape(std::vector<int64_t> newshape) const;
  NDArray Reshape(const FTList<int64_t>& newshape) const;
  NDArray Reshape(const List& newshape) const;
  NDArray Reshape(const Tuple& newshape) const;
  NDArray Reshape(const Any& newshape) const;

  NDArray Squeeze(const std::vector<int64_t>& axis = {}) const;
  NDArray Squeeze(const Tuple& axis) const;
  NDArray Squeeze(const Any& axis) const;

  NDArray Unsqueeze(int64_t dim) const;
  NDArray Unsqueeze(const Any& dim) const;
  /*!
   * \brief Create a NDArray that shares the data memory with the current one.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \note The memory size of new array must be smaller than the current one.
   */
  MATX_DLL NDArray CreateView(std::vector<int64_t> shape, DLDataType dtype) const;
  MATX_DLL NDArray CreateViewWithStrides(std::vector<int64_t> shape,
                                         std::vector<int64_t> strides,
                                         DLDataType dtype) const;
  /*!
   * \brief Create a reference view of NDArray that
   *  represents as DLManagedTensor.
   * \return A DLManagedTensor
   */
  MATX_DLL DLManagedTensor* ToDLPack() const;
  /*!
   * \brief From shape to strides, only work from compact tensor
   * \param shape The shape of the Array.
   * \return The strides of Array
   */
  static std::vector<int64_t> GenStridesFromShape(const std::vector<int64_t>& shape);
  /*!
   * \brief Create an empty NDArray.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param device The device of the Array.
   * \return The created Array
   */
  MATX_DLL static NDArray Empty(std::vector<int64_t> shape, DLDataType dtype, DLDevice ctx);
  MATX_DLL static NDArray Empty(const int64_t* shape, int64_t dim, DLDataType dtype, DLDevice ctx);
  /*!
   * \brief Create a NDArray backed by a dlpack tensor.
   *
   * This allows us to create a NDArray using the memory
   * allocated by an external deep learning framework
   * that is DLPack compatible.
   *
   * The memory is retained until the NDArray went out of scope.
   * \param tensor The DLPack tensor to copy from.
   * \return The created NDArray view.
   */
  MATX_DLL static NDArray FromDLPack(DLManagedTensor* tensor);
  /*!
   * \brief Function to copy data from one array to another.
   * \param from The source array.
   * \param to The target array.
   * \param stream The stream used in copy.
   */
  MATX_DLL static void CopyFromTo(const DLTensor* from,
                                  DLTensor* to,
                                  MATXScriptStreamHandle stream);

  /*!
   * \brief Function to copy data from one array to another use current stream.
   * \param from The source array.
   * \param to The target array.
   */
  MATX_DLL static void CopyFromTo(const DLTensor* from, DLTensor* to);

  MATX_DLL std::vector<int64_t> Shape() const;
  MATX_DLL ::matxscript::runtime::DataType DataType() const;

 public:
  // iterators
  Iterator iter() const;

 public:
  const int64_t* GetStridesPtr() const;
  const int64_t* GetShapePtr() const;
  int GetDim() const;
  RTValue get_item(int64_t index) const;
  RTValue get_item(const Any& index) const;
  int64_t get_item_as_int64(int64_t index) const;
  int64_t get_item_as_int64(const Any& index) const;
  double get_item_as_double(int64_t index) const;
  double get_item_as_double(const Any& index) const;

  RTValue fused_get_item(const int64_t* indexes, size_t num_indexes) const;
  int64_t fused_get_item_as_int64(const int64_t* indexes, size_t num_indexes) const;
  double fused_get_item_as_double(const int64_t* indexes, size_t num_indexes) const;
  RTValue fused_get_item(const std::initializer_list<int64_t>& indexes) const;
  int64_t fused_get_item_as_int64(const std::initializer_list<int64_t>& indexes) const;
  double fused_get_item_as_double(const std::initializer_list<int64_t>& indexes) const;

  void set_item(int64_t index, int64_t value) const;
  void set_item(int64_t index, double value) const;
  void set_item(int64_t index, const Any& item) const;
  void set_item(const Any& index, int64_t value) const;
  void set_item(const Any& index, double value) const;
  void set_item(const Any& index, const Any& item) const;
  void fused_set_item(const int64_t* indexes, size_t num_indexes, int64_t value) const;
  void fused_set_item(const int64_t* indexes, size_t num_indexes, double value) const;
  void fused_set_item(const int64_t* indexes, size_t num_indexes, const Any& item) const;
  void fused_set_item(const std::initializer_list<int64_t>& indexes, int64_t value) const;
  void fused_set_item(const std::initializer_list<int64_t>& indexes, double value) const;
  void fused_set_item(const std::initializer_list<int64_t>& indexes, const Any& item) const;

  NDArray get_slice(int64_t begin, int64_t end, int64_t step) const;
  void set_slice(int64_t begin, int64_t end, const Any& item) const;
  int64_t size() const;
  NDArray transpose(const Any& axes = None) const;
  NDArray as_type(const unicode_view& dtype_str) const;

 public:
  static void AssignNDArray(const NDArray& src, NDArray& dst);

 public:
  static void check_dtype_valid(const unicode_view& dtype_str);

 private:
  void set_item_helper(void* dst_data,
                       const int64_t* dst_shape,
                       const int64_t* dst_strides,
                       int dst_ndim,
                       const Any& item) const;

 public:
  MATX_DLL List ToList() const;
  MATX_DLL List ShapeList() const;
  MATX_DLL Unicode DTypeUnicode() const;
  MATX_DLL Unicode Device() const;

  MATX_DLL size_t DataSize() const;
  MATX_DLL int64_t ElementSize() const;

  MATX_DLL const void* RawData() const;

  template <typename T>
  MATX_DLL const T* Data() const;

  // internal namespace
  struct Internal;

 protected:
  friend class RTValue;
  friend class RuntimeValueConverter;
  friend class NDArrayBinOpHelper;
  friend class NDArrayHelper;
  friend class NDArrayOperate;
  /*!
   * \brief Get mutable internal container pointer.
   * \return a mutable container pointer.
   */
  Container* get_mutable() const;
  // Helper functions for FFI handling.
  /*!
   * \brief Construct NDArray's Data field from array handle in FFI.
   * \param handle The array handle.
   * \return The corresponding ObjectPtr to the constructed container object.
   *
   * \note We keep a special calling convention for NDArray by passing
   *       ContainerBase pointer in FFI.
   *       As a result, the argument is compatible to DLTensor*.
   */
  static ObjectPtr<Object> FFIDataFromHandle(MATXScriptTensorHandle handle);
  /*!
   * \brief DecRef resource managed by an FFI array handle.
   * \param handle The array handle.
   */
  static void FFIDecRef(MATXScriptTensorHandle handle);
  /*!
   * \brief Get FFI Array handle from ndarray.
   * \param nd The object with ndarray type.
   * \return The result array handle.
   */
  static MATXScriptTensorHandle FFIGetHandle(const ObjectRef& nd);
};

// implementations of inline functions
/*!
 * \brief return the size of data the DLTensor hold, in term of number of bytes
 *
 *  \param arr the input DLTensor
 *  \return number of  bytes of data in the DLTensor.
 */
size_t GetDataSize(const DLTensor& arr);

/*!
 * \brief check if a DLTensor is contiguous.
 * \param arr The input DLTensor.
 * \return The check result.
 */
bool IsContiguous(const DLTensor& arr);

Object* MATXScriptArrayHandleToObjectHandle(MATXScriptTensorHandle handle);

template <>
RTValue::RTValue(NDArray val) noexcept;

template <>
MATXSCRIPT_ALWAYS_INLINE NDArray Any::As<NDArray>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeNDArray);
  return NDArray(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE NDArray Any::AsNoCheck<NDArray>() const {
  return NDArray(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
bool IsConvertible<NDArray>(const Object* node);

namespace TypeIndex {
template <>
struct type_index_traits<NDArray> {
  static constexpr int32_t value = kRuntimeNDArray;
};
}  // namespace TypeIndex

template <typename T>
const T* NDArray::Data() const {
  return static_cast<const T*>(this->RawData());
}

std::ostream& operator<<(std::ostream& os, NDArray const& n);

}  // namespace runtime
}  // namespace matxscript
