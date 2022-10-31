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

#include <matxscript/runtime/container/list_ref.h>
#include <matxscript/runtime/container/user_data_ref.h>
#include <matxscript/runtime/half.h>

namespace matxscript {
namespace runtime {

namespace list_helper_details {
template <typename T>
MATXSCRIPT_ALWAYS_INLINE void AnyValue2ElementData(T* ret, const Any& from) {
  *ret = from.template As<T>();
}

template <>
MATXSCRIPT_ALWAYS_INLINE void AnyValue2ElementData(Half* ret, const Any& from) {
  *ret = from.template As<float>();
}
}  // namespace list_helper_details

struct ListHelper {
 public:
  template <typename DT>
  class SimpleVec {
   public:
    SimpleVec(int64_t size) : size_(size), malloc_(true) {
      data_ = new DT[size];
    }
    SimpleVec(DT* data) : data_(data), malloc_(false) {
    }
    ~SimpleVec() {
      if (data_ != nullptr && malloc_) {
        delete[] data_;
      }
      data_ = nullptr;
    }
    void push_back(DT val) {
      data_[index_++] = val;
    }
    DT* data() {
      return data_;
    }

   private:
    DT* data_ = nullptr;
    int64_t index_ = 0;
    int64_t size_ = 0;
    bool malloc_;
  };

 private:
  template <typename DT>
  static bool IsNDArrayImpl(const RTValue& rt_value,
                            const std::vector<int64_t>& shape,
                            int ndim,
                            int depth,
                            SimpleVec<DT>& flat_list) {
    if (rt_value.type_code() == TypeIndex::kRuntimeInteger ||
        rt_value.type_code() == TypeIndex::kRuntimeFloat) {
      if (depth != ndim)
        return false;
      DT flat_value;
      list_helper_details::AnyValue2ElementData(&flat_value, rt_value);
      flat_list.push_back(flat_value);
      return true;
    }
    if (rt_value.type_code() == TypeIndex::kRuntimeList) {
      auto view = rt_value.AsObjectView<List>();
      const List& cur_list = view.data();
      if (cur_list.empty())
        return false;
      if (depth >= ndim)
        return false;
      if (cur_list.size() != shape[depth])
        return false;
      for (const RTValue& element : cur_list) {
        bool flag = IsNDArrayImpl(element, shape, ndim, depth + 1, flat_list);
        if (!flag)
          return false;
      }
      return true;
    }
    return false;
  }

 public:
  static bool FirstShape(const List& list, std::vector<int64_t>& shape);
  // static method
  /*!
   * \brief check whether list can be converted to ndarray.
   * \param list
   * \param shape: return shape
   * \param flat_list: return flat list
   *
   * \example
   * list = [[1,2,3], [4,5,6]]
   * shape = [2,3]
   * flat_list = [1, 2, 3, 4, 5, 6]
   */
  template <typename DT>
  static std::shared_ptr<SimpleVec<DT>> IsNDArray(const List& list, std::vector<int64_t>& shape) {
    shape.clear();
    // get shape from every first element
    if (!FirstShape(list, shape)) {
      return nullptr;
    }
    int64_t element_num = 1;
    for (auto i : shape) {
      element_num *= i;
    }
    auto ptr = std::make_shared<SimpleVec<DT>>(element_num);
    // check is matrix
    if (IsNDArrayImpl(list, shape, shape.size(), 0, *ptr)) {
      return ptr;
    } else {
      return nullptr;
    }
  }
  template <typename DT>
  static std::shared_ptr<SimpleVec<DT>> FlatList(const List& list,
                                                 const std::vector<int64_t>& shape) {
    int64_t element_num = 1;
    for (auto i : shape) {
      element_num *= i;
    }
    auto ptr = std::make_shared<SimpleVec<DT>>(element_num);
    // check is matrix
    if (IsNDArrayImpl(list, shape, shape.size(), 0, *ptr)) {
      return ptr;
    } else {
      return nullptr;
    }
  }
  template <typename DT>
  static std::shared_ptr<SimpleVec<DT>> FlatList(const List& list,
                                                 const std::vector<int64_t>& shape,
                                                 DT* data) {
    auto ptr = std::make_shared<SimpleVec<DT>>(data);
    if (IsNDArrayImpl(list, shape, shape.size(), 0, *ptr)) {
      return ptr;
    } else {
      return nullptr;
    }
  }

  static void Sort(const List& list);
  static void Sort(const List& list, const UserDataRef& comp);

  static void NthElement(const List& list, int64_t n);
  static void NthElement(const List& list, int64_t n, const UserDataRef& comp);

  static void Heapify(const List& list);
  static void Heapify(const List& list, const UserDataRef& comp);
  static void HeapReplace(const List& list, const Any& item);
  static void HeapReplace(const List& list, const Any& item, const UserDataRef& comp);
  static RTValue HeapPushPop(const List& list, const Any& item);
  static RTValue HeapPushPop(const List& list, const Any& item, const UserDataRef& comp);
};

}  // namespace runtime
}  // namespace matxscript