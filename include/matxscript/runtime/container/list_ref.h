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

#include <cstdint>
#include <initializer_list>
#include <vector>

#include <matxscript/runtime/_pdqsort.h>
#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/itertor_ref.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/unicode_view.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class ListNode;

class List : public ObjectRef {
 public:
  // data holder
  using container_type = std::vector<RTValue>;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

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
  using FuncGetNextItemRandom = std::function<value_type()>;
  using FuncGetNextItemForward = std::function<value_type(bool& has_next)>;
  using FuncEqualToValue = std::function<bool(const value_type&)>;

 public:
  // constructors
  /*!
   * \brief default constructor
   */
  List();

  /*!
   * \brief move constructor
   * \param other source
   */
  List(List&& other) noexcept = default;

  /*!
   * \brief copy constructor
   * \param other source
   */
  List(const List& other) noexcept = default;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit List(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  List(IterType first, IterType last) {
    if (std::is_same<typename std::iterator_traits<IterType>::iterator_category,
                     std::random_access_iterator_tag>::value) {
      size_t iter_num = std::distance(first, last);
      FuncGetNextItemRandom func = [&first]() -> value_type { return value_type(*(first++)); };
      this->Init(func, iter_num);
    } else {
      FuncGetNextItemForward func = [&first, &last](bool& has_next) -> value_type {
        value_type next_item(*(first++));
        has_next = first != last;
        return next_item;
      };
      this->Init(func, first != last);
    }
  }

  List(const Any* begin, const Any* end);

  /*!
   * \brief constructor from initializer list
   * \param init The initializer list
   */
  List(std::initializer_list<value_type> init);

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  List(const std::vector<value_type>& init);

  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   */
  explicit List(size_t n, const value_type& val);

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  List& operator=(List&& other) noexcept = default;

  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  List& operator=(const List& other) noexcept = default;

  bool operator==(const List& other) const;

  bool operator!=(const List& other) const {
    return !operator==(other);
  }

  bool operator>(const List& other) const;
  bool operator>=(const List& other) const;

  bool operator<(const List& other) const {
    return other.operator>(*this);
  }
  bool operator<=(const List& other) const {
    return other.operator>=(*this);
  }

 public:
  // iterators
  Iterator iter() const;
  iterator begin() const;
  iterator nocheck_begin() const;
  iterator end() const;
  iterator nocheck_end() const;
  reverse_iterator rbegin() const;
  reverse_iterator nocheck_rbegin() const;
  reverse_iterator rend() const;
  reverse_iterator nocheck_rend() const;

  value_type* data() const;

 public:
  // const methods in std::vector
  /*!
   * \brief Immutably read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  value_type& operator[](int64_t i) const;

  /*! \return The size of the array */
  int64_t size() const;

  /*! \return The capacity of the array */
  int64_t capacity() const;

  /*! \return Whether array is empty */
  bool empty() const {
    return size() == 0;
  }

  template <typename U>
  int64_t index(const U&& item, int64_t start, int64_t end) const {
    int64_t len = size();
    start = slice_index_correction(start, len);
    end = slice_index_correction(end, len);

    if (start >= end) {
      THROW_PY_ValueError("ValueError: '", item, "' is not in list");
      return -1;
    }

    FuncEqualToValue fn_equal_to = [&item](const value_type& val) -> bool {
      return SmartEqualTo()(val, item);
    };

    int idx = find_match_idx_fn(fn_equal_to, start, end);
    if (idx == end) {
      THROW_PY_ValueError("ValueError: '", item, "' is not in list");
      return -1;
    }
    return idx;
  }

  template <typename U>
  int64_t index(const U&& item, int64_t start = 0) const {
    int64_t end = size();
    return index(std::move(item), start, end);
  }

  template <typename U>
  bool contains(U&& item) const {
    FuncEqualToValue fn_equal_to = [&item](const value_type& val) -> bool {
      return SmartEqualTo()(val, item);
    };
    return find_match_fn(fn_equal_to);
  }

  bool find_match_fn(const FuncEqualToValue& fn) const;

  int64_t find_match_idx_fn(const FuncEqualToValue& fn, int64_t start, int64_t end) const;

  template <typename U>
  int64_t count(U&& item) const {
    FuncEqualToValue fn_equal_to = [&item](const value_type& val) -> bool {
      return SmartEqualTo()(val, item);
    };
    return count_match_fn(fn_equal_to);
  }

  int64_t count_match_fn(const FuncEqualToValue& fn) const;

  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  void push_back(const value_type& item) const;
  void push_back(value_type&& item) const;

  void pop_back() const;

 public:
  // method for python
  value_type& get_item(int64_t i) const;

  void set_item(int64_t i, const value_type& item) const;
  void set_item(int64_t i, value_type&& item) const;

  List get_slice(int64_t b, int64_t e, int64_t step = 1) const;

  void set_slice(int64_t start, int64_t end, List&& rhs) const;
  void set_slice(int64_t start, int64_t end, const List& rhs) const;

  void reserve(int64_t new_size) const;

  void resize(int64_t new_size) const;

  template <typename U>
  void append(U&& item) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    if (std::is_same<U_TYPE, value_type>::value) {
      return this->push_back(std::forward<U>(item));
    } else {
      return this->push_back(GenericValueConverter<value_type>{}(std::forward<U>(item)));
    }
  }

  void extend(List&& items) const;
  void extend(const List& items) const;
  void extend(const Iterator& items) const;
  void extend(const Any& items) const;

  List repeat(int64_t times) const;

  // for fuse list_make and repeat
  static List repeat_one(const Any& value, int64_t times);
  static List repeat_one(value_type&& value, int64_t times);
  static List repeat_many(const std::initializer_list<value_type>& values, int64_t times);

  void clear() const;

  value_type pop(int64_t index = -1) const;

  void insert(int64_t index, const Any& item) const;
  void insert(int64_t index, value_type&& item) const;

  void remove(const Any& item) const;

  void reverse() const;

  void sort(bool reverse = false) const;
  void sort(const Any& key, bool reverse = false) const;

  template <bool try_move = false>
  static List Concat(std::initializer_list<List> data);

  template <typename... ARGS>
  static inline List Concat(ARGS&&... args) {
    std::initializer_list<List> list_args{std::forward<ARGS>(args)...};
    return List::Concat<true>(list_args);
  }

 public:
  // List's own methods

  /*! \return The underlying ListNode */
  ListNode* CreateOrGetListNode();

  /*! \return The underlying ListNode */
  ListNode* GetListNode() const;

  /*! \brief specify container node */
  using ContainerType = ListNode;

 private:
  void Init(const FuncGetNextItemForward& func, bool has_next);
  void Init(const FuncGetNextItemRandom& func, size_t len);
};

template <>
List List::Concat<true>(std::initializer_list<List> data);
template <>
List List::Concat<false>(std::initializer_list<List> data);

namespace TypeIndex {
template <>
struct type_index_traits<List> {
  static constexpr int32_t value = kRuntimeList;
};
}  // namespace TypeIndex

template <>
MATXSCRIPT_ALWAYS_INLINE List Any::As<List>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeList);
  return List(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE List Any::AsNoCheck<List>() const {
  return List(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
bool IsConvertible<List>(const Object* node);

std::ostream& operator<<(std::ostream& os, List const& n);

}  // namespace runtime
}  // namespace matxscript
