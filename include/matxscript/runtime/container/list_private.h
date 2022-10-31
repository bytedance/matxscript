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

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

/*! \brief list node content in array */
class ListNode : public Object {
 public:
  // data holder
  using _type = RTValue;
  using container_type = std::vector<_type>;

  // types
  using value_type = container_type::value_type;
  using reference = container_type::reference;
  using const_reference = container_type::const_reference;
  using iterator = container_type::iterator;
  using const_iterator = container_type::const_iterator;
  using size_type = container_type::size_type;
  using difference_type = container_type::difference_type;
  using allocator_type = container_type::allocator_type;
  using pointer = container_type::pointer;
  using const_pointer = container_type::const_pointer;
  using reverse_iterator = container_type::reverse_iterator;
  using const_reverse_iterator = container_type::const_reverse_iterator;

 public:
  ListNode() : data_container() {
  }

  ListNode(std::initializer_list<value_type> init) : data_container(init) {
  }

  ListNode(const size_t n, const value_type& val) : data_container(n, val) {
  }
  template <class B, class E>
  ListNode(B begin, E end) : data_container(begin, end) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeList;
  static constexpr const char* _type_key = "List";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(ListNode, Object);

 public:
  // iterators
  MATXSCRIPT_ALWAYS_INLINE iterator begin() {
    return data_container.begin();
  }

  MATXSCRIPT_ALWAYS_INLINE const_iterator begin() const {
    return data_container.begin();
  }

  MATXSCRIPT_ALWAYS_INLINE iterator end() {
    return data_container.end();
  }

  MATXSCRIPT_ALWAYS_INLINE const_iterator end() const {
    return data_container.end();
  }

  MATXSCRIPT_ALWAYS_INLINE reverse_iterator rbegin() {
    return data_container.rbegin();
  }

  MATXSCRIPT_ALWAYS_INLINE const_reverse_iterator rbegin() const {
    return data_container.rbegin();
  }

  MATXSCRIPT_ALWAYS_INLINE reverse_iterator rend() {
    return data_container.rend();
  }

  MATXSCRIPT_ALWAYS_INLINE const_reverse_iterator rend() const {
    return data_container.rend();
  }

 public:
  // const methods in std::vector
  MATXSCRIPT_ALWAYS_INLINE const value_type& operator[](int64_t i) const {
    MXCHECK(i < data_container.size()) << "ValueError: index overflow";
    return data_container[i];
  }

  MATXSCRIPT_ALWAYS_INLINE value_type& operator[](int64_t i) {
    MXCHECK(i >= 0 && i < data_container.size()) << "ValueError: index overflow";
    return data_container[i];
  }

  MATXSCRIPT_ALWAYS_INLINE size_t size() const {
    return data_container.size();
  }

  /*! \return The capacity of the array */
  MATXSCRIPT_ALWAYS_INLINE int64_t capacity() const {
    return data_container.capacity();
  }

  /*! \return Whether array is empty */
  MATXSCRIPT_ALWAYS_INLINE bool empty() const {
    return data_container.empty();
  }

  template <typename U>
  bool contains(U&& item) const {
    for (auto& val : data_container) {
      if (std::equal_to<value_type>()(val, std::forward<U>(item))) {
        return true;
      }
    }
    return false;
  }

  template <typename U>
  int64_t counts(U&& item) const {
    int64_t count = 0;
    for (auto& val : data_container) {
      if (std::equal_to<value_type>()(val, std::forward<U>(item))) {
        ++count;
      }
    }
    return count;
  }

  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  MATXSCRIPT_ALWAYS_INLINE void push_back(value_type item) {
    data_container.push_back(std::move(item));
  }

  template <class... Args>
  MATXSCRIPT_ALWAYS_INLINE void emplace_back(Args&&... args) {
    data_container.emplace_back(std::forward<Args>(args)...);
  }

  MATXSCRIPT_ALWAYS_INLINE void pop_back() {
    data_container.pop_back();
  }

  MATXSCRIPT_ALWAYS_INLINE void reserve(int64_t new_size) {
    data_container.reserve(new_size > 0 ? new_size : 0);
  }

  MATXSCRIPT_ALWAYS_INLINE void resize(int64_t new_size) {
    if (new_size >= 0) {
      data_container.resize(new_size);
    }
  }

  template <typename U>
  MATXSCRIPT_ALWAYS_INLINE void append(U&& item) {
    push_back(GenericValueConverter<value_type>{}(std::forward<U>(item)));
  }

  MATXSCRIPT_ALWAYS_INLINE void clear() {
    data_container.clear();
  }

  template <typename U>
  MATXSCRIPT_ALWAYS_INLINE void insert(int64_t index, U&& item) {
    index = index_correction(index, data_container.size());
    if (index < 0) {  // To sync with Python insert result
      index = 0;
    }
    if (index >= data_container.size()) {
      append(item);
      return;
    }
    auto itr = data_container.begin() + index;
    data_container.insert(itr, GenericValueConverter<value_type>{}(std::forward<U>(item)));
  }

  value_type pop(int64_t index = -1) {
    value_type ret;
    index = index_correction(index, data_container.size());
    if (index >= 0 && index < data_container.size()) {
      auto itr = data_container.begin() + index;
      ret = std::move(*itr);
      data_container.erase(itr);
    } else {
      if (data_container.empty()) {
        MXTHROW << "[List.pop] IndexError: pop from empty list";
      } else {
        MXTHROW << "[List.pop] IndexError: pop index out of range";
      }
    }
    return ret;
  }

  MATXSCRIPT_ALWAYS_INLINE void remove(const Any& item) {
    auto first = data_container.begin();
    auto last = data_container.end();
    for (; first != last; ++first) {
      if (Any::Equal(*first, item)) {
        data_container.erase(first);
        return;
      }
    }
    MXTHROW << "[list.remove] " << item << " not in list";
  }

  MATXSCRIPT_ALWAYS_INLINE void reverse() {
    std::reverse(data_container.begin(), data_container.end());
  }

 private:
  container_type data_container;

  // Reference class
  friend class List;
  friend class ListNodeTrait;
  friend class ListHelper;
};

}  // namespace runtime
}  // namespace matxscript
