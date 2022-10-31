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

#include "_flat_hash_map.h"
#include "itertor_ref.h"

#include <initializer_list>
#include <vector>

#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

class SetNode;

class Set : public ObjectRef {
 public:
  // data holder
  using container_type = ska::flat_hash_set<RTValue>;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 public:
  // types
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using size_type = typename container_type::size_type;
  using difference_type = typename container_type::difference_type;
  using value_type = typename container_type::value_type;
  using allocator_type = typename container_type::allocator_type;
  using pointer = typename container_type::pointer;
  using const_pointer = typename container_type::const_pointer;
  using FuncGetNextItemRandom = std::function<value_type()>;
  using FuncGetNextItemForward = std::function<value_type(bool& has_next)>;

  // iterators
  Iterator iter() const;
  const_iterator begin() const;
  const_iterator end() const;

  // constructors
  /*!
   * \brief default constructor
   */
  Set();

  /*!
   * \brief move constructor
   * \param other source
   */
  Set(Set&& other) noexcept;

  /*!
   * \brief copy constructor
   * \param other source
   */
  Set(const Set& other) noexcept;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Set(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Set(IterType first, IterType last) {
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

  Set(const Any* begin, const Any* end);

  /*!
   * \brief constructor from initializer Set
   * \param init The initializer Set
   */
  Set(std::initializer_list<value_type> init);

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  Set(const std::vector<value_type>& init);

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Set& operator=(Set&& other) noexcept;

  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Set& operator=(const Set& other) noexcept;

  bool operator==(const Set& other) const;
  bool operator!=(const Set& other) const;

 public:
  // mutation in std::unordered_set
  void emplace(value_type&& item) const;
  void emplace(const value_type& item) const {
    return emplace(value_type(item));
  }

  template <class U>
  void add(U&& item) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    if (std::is_same<U_TYPE, value_type>::value) {
      return this->emplace(std::forward<U>(item));
    } else {
      return this->emplace(GenericValueConverter<value_type>{}(std::forward<U>(item)));
    }
  }

  void clear() const;

  void reserve(int64_t new_size) const;

 public:
  // const methods in std::unordered_map
  int64_t size() const;

  int64_t bucket_count() const;

  bool empty() const;

  bool contains(const Any& key) const;

  bool contains(string_view key) const;
  bool contains(unicode_view key) const;
  MATXSCRIPT_ALWAYS_INLINE bool contains(const String& key) const {
    return contains(key.view());
  }
  MATXSCRIPT_ALWAYS_INLINE bool contains(const Unicode& key) const {
    return contains(key.view());
  }
  MATXSCRIPT_ALWAYS_INLINE bool contains(const char* key) const {
    return contains(string_view(key));
  }
  MATXSCRIPT_ALWAYS_INLINE bool contains(const char32_t* const key) const {
    return contains(unicode_view(key));
  }
  bool contains(int64_t key) const;
  template <class U, typename = typename std::enable_if<!is_runtime_value<U>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE bool contains(const U& key) const {
    return this->contains(static_cast<const Any&>(GenericValueConverter<RTView>{}(key)));
  }

 public:
  void difference_update(PyArgs args) const;
  Set difference(PyArgs args) const;
  void update(PyArgs args) const;
  Set set_union(PyArgs args) const;
  void discard(const Any& rt_value) const;

  // Set's own methods
  /*! \return The underlying SetNode */
  SetNode* CreateOrGetSetNode();

  /*! \return The underlying SetNode */
  SetNode* GetSetNode() const;

  /*! \brief specify container node */
  using ContainerType = SetNode;

 private:
  void difference_update_iter(const Iterator& iter) const;
  void update_iter(const Iterator& iter) const;
  void Init(const FuncGetNextItemForward& func, bool has_next);
  void Init(const FuncGetNextItemRandom& func, size_t len);
};

namespace TypeIndex {
template <>
struct type_index_traits<Set> {
  static constexpr int32_t value = kRuntimeSet;
};
}  // namespace TypeIndex

template <>
MATXSCRIPT_ALWAYS_INLINE Set Any::As<Set>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeSet);
  return Set(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE Set Any::AsNoCheck<Set>() const {
  return Set(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

std::ostream& operator<<(std::ostream& os, Set const& n);

template <>
bool IsConvertible<Set>(const Object* node);

}  // namespace runtime
}  // namespace matxscript
