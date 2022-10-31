// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of iterator_adaptator is inspired by pythran.
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

#include "string_view.h"
#include "unicode_view.h"

#include <initializer_list>
#include <vector>

#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/itertor_ref.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

template <class I>
struct item_iterator_adaptator : public I {
  using value_type = typename I::value_type;
  using pointer = value_type*;
  using reference = value_type&;
  item_iterator_adaptator() = default;
  item_iterator_adaptator(I const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  item_iterator_adaptator(U const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  item_iterator_adaptator(item_iterator_adaptator<U> const& other)
      : item_iterator_adaptator(*(static_cast<const U*>(&other))) {
  }
  value_type operator*() const {
    return I::operator*();
  }
};

template <class I>
struct key_iterator_adaptator : public I {
  using value_type = typename I::value_type::first_type;
  using pointer = typename I::value_type::first_type*;
  using reference = typename I::value_type::first_type&;
  key_iterator_adaptator() = default;
  key_iterator_adaptator(I const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  key_iterator_adaptator(U const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  key_iterator_adaptator(key_iterator_adaptator<U> const& other)
      : key_iterator_adaptator(*(static_cast<const U*>(&other))) {
  }
  value_type operator*() const {
    return (*this)->first;
  }
};

template <class I>
struct value_iterator_adaptator : public I {
  using value_type = typename I::value_type::second_type;
  using pointer = typename I::value_type::second_type*;
  using reference = typename I::value_type::second_type&;
  value_iterator_adaptator() = default;
  value_iterator_adaptator(I const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  value_iterator_adaptator(U const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  value_iterator_adaptator(key_iterator_adaptator<U> const& other)
      : value_iterator_adaptator(*(static_cast<const U*>(&other))) {
  }
  value_type operator*() const {
    return (*this)->second;
  }
};

template <class D>
struct DictItems {
  using iterator = typename D::item_const_iterator;
  using value_type = typename iterator::value_type;
  D data;
  DictItems() = default;
  DictItems(D const& d) : data(d) {
  }
  iterator begin() const {
    return data.item_begin();
  }
  iterator end() const {
    return data.item_end();
  }
  int64_t size() const {
    return data.size();
  }
};

template <class D>
struct DictKeys {
  using iterator = typename D::key_const_iterator;
  using value_type = typename iterator::value_type;
  D data;
  DictKeys() = default;
  DictKeys(D const& d) : data(d) {
  }
  iterator begin() const {
    return data.key_begin();
  }
  iterator end() const {
    return data.key_end();
  }
  int64_t size() const {
    return data.size();
  }
};

template <class D>
struct DictValues {
  using iterator = typename D::value_const_iterator;
  using value_type = typename iterator::value_type;
  D data;
  DictValues() = default;
  DictValues(D const& d) : data(d) {
  }
  iterator begin() const {
    return data.value_begin();
  }
  iterator end() const {
    return data.value_end();
  }
  int64_t size() const {
    return data.size();
  }
};

class DictNode;

class Dict : public ObjectRef {
 public:
  // data holder
  using container_type = ska::flat_hash_map<RTValue, RTValue>;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 public:
  // types
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using iterator = key_iterator_adaptator<typename container_type::iterator>;
  using const_iterator = key_iterator_adaptator<typename container_type::const_iterator>;
  using item_iterator = item_iterator_adaptator<typename container_type::iterator>;
  using item_const_iterator = item_iterator_adaptator<typename container_type::const_iterator>;
  using key_iterator = key_iterator_adaptator<typename container_type::iterator>;
  using key_const_iterator = key_iterator_adaptator<typename container_type::const_iterator>;
  using value_iterator = value_iterator_adaptator<typename container_type::iterator>;
  using value_const_iterator = value_iterator_adaptator<typename container_type::const_iterator>;
  using size_type = typename container_type::size_type;
  using difference_type = typename container_type::difference_type;
  using value_type = typename container_type::value_type;
  using allocator_type = typename container_type::allocator_type;
  using pointer = typename container_type::pointer;
  using const_pointer = typename container_type::const_pointer;
  using key_type = typename container_type::key_type;
  using mapped_type = typename container_type::mapped_type;
  using FuncGetNextItemRandom = std::function<value_type()>;
  using FuncGetNextItemForward = std::function<value_type(bool& has_next)>;

  // iterators
  iterator begin() const;
  iterator end() const;
  item_iterator item_begin() const;
  item_iterator item_end() const;
  key_const_iterator key_begin() const;
  key_const_iterator key_end() const;
  value_iterator value_begin() const;
  value_iterator value_end() const;

  // constructors
  /*!
   * \brief default constructor
   */
  Dict();

  /*!
   * \brief move constructor
   * \param other source
   */
  Dict(Dict&& other) noexcept;

  /*!
   * \brief copy constructor
   * \param other source
   */
  Dict(const Dict& other) noexcept;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Dict(ObjectPtr<Object> n) noexcept;

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Dict(IterType first, IterType last) {
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

  /*!
   * \brief constructor from initializer Dict
   * \param init The initializer Dict
   */
  Dict(std::initializer_list<value_type> init);

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  Dict(const std::vector<value_type>& init);

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Dict& operator=(Dict&& other) noexcept = default;

  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Dict& operator=(const Dict& other) noexcept = default;

  bool operator==(const Dict& other) const;

  bool operator!=(const Dict& other) const;

 public:
  // method for python
  template <class KEY_T, typename = typename std::enable_if<!is_runtime_value<KEY_T>::value>::type>
  mapped_type& get_item(KEY_T const& key) const {
    return get_item(static_cast<const Any&>(RTView(key)));
  }

  mapped_type& get_item(const Any& key) const;
  mapped_type& get_item(int64_t key) const;

  mapped_type& get_item(const string_view& key) const;
  MATXSCRIPT_ALWAYS_INLINE mapped_type& get_item(const String& key) const {
    return get_item(key.view());
  }
  MATXSCRIPT_ALWAYS_INLINE mapped_type& get_item(char const* key) const {
    return get_item(string_view(key));
  }

  mapped_type& get_item(const unicode_view& key) const;
  MATXSCRIPT_ALWAYS_INLINE mapped_type& get_item(const Unicode& key) const {
    return get_item(key.view());
  }
  MATXSCRIPT_ALWAYS_INLINE mapped_type& get_item(char32_t const* key) const {
    return get_item(unicode_view(key));
  }

  template <class KEY_T>
  mapped_type const& get_default(KEY_T const& key, mapped_type const& default_val = None) const {
    return get_default(static_cast<const Any&>(RTView(key)), default_val);
  }

  mapped_type const& get_default(const Any& key, mapped_type const& default_val = None) const;

  mapped_type const& get_default(const string_view& key,
                                 mapped_type const& default_val = None) const;

  mapped_type const& get_default(const unicode_view& key,
                                 mapped_type const& default_val = None) const;

  MATXSCRIPT_ALWAYS_INLINE mapped_type const& get_default(
      const String& key, mapped_type const& default_val = None) const {
    return this->get_default(key.view(), default_val);
  }

  MATXSCRIPT_ALWAYS_INLINE mapped_type const& get_default(
      const Unicode& key, mapped_type const& default_val = None) const {
    return this->get_default(key.view(), default_val);
  }

  MATXSCRIPT_ALWAYS_INLINE mapped_type const& get_default(
      char const* key, mapped_type const& default_val = None) const {
    return this->get_default(string_view(key), default_val);
  }

  MATXSCRIPT_ALWAYS_INLINE mapped_type const& get_default(
      char32_t const* key, mapped_type const& default_val = None) const {
    return this->get_default(unicode_view(key), default_val);
  }

  mapped_type pop(PyArgs args) const;

  void set_item(key_type&& key, mapped_type&& value) const;
  MATXSCRIPT_ALWAYS_INLINE void set_item(const key_type& key, mapped_type&& value) const {
    return set_item(key_type(key), std::move(value));
  }
  MATXSCRIPT_ALWAYS_INLINE void set_item(key_type&& key, const mapped_type& value) const {
    return set_item(std::move(key), mapped_type(value));
  }
  MATXSCRIPT_ALWAYS_INLINE void set_item(const key_type& key, const mapped_type& value) const {
    return set_item(key_type(key), mapped_type(value));
  }

  // mutation in std::unordered_map
  mapped_type& operator[](key_type key) const;

  mapped_type& operator[](const char* key) const;

  mapped_type& operator[](const char32_t* key) const;

  void emplace(key_type&& key, mapped_type&& value) const;
  MATXSCRIPT_ALWAYS_INLINE void emplace(const key_type& key, mapped_type&& value) const {
    return emplace(key_type(key), std::move(value));
  }
  MATXSCRIPT_ALWAYS_INLINE void emplace(key_type&& key, const mapped_type& value) const {
    return emplace(std::move(key), mapped_type(value));
  }
  MATXSCRIPT_ALWAYS_INLINE void emplace(const key_type& key, const mapped_type& value) const {
    return emplace(key_type(key), mapped_type(value));
  }

  void emplace(value_type&& value) const;
  MATXSCRIPT_ALWAYS_INLINE void emplace(const value_type& value) const {
    return emplace(value_type(value));
  }

  void clear() const;

  void reserve(int64_t new_size) const;

 public:
  // const methods in std::unordered_map
  int64_t size() const;

  int64_t bucket_count() const;

  bool empty() const;

  bool contains(const Any& key) const;
  bool contains(int64_t key) const;

  bool contains(const string_view& key) const;
  MATXSCRIPT_ALWAYS_INLINE bool contains(const String& key) const {
    return this->contains(key.view());
  }
  MATXSCRIPT_ALWAYS_INLINE bool contains(const char* key) const {
    return this->contains(string_view(key));
  }

  bool contains(const unicode_view& key) const;
  MATXSCRIPT_ALWAYS_INLINE bool contains(const Unicode& key) const {
    return this->contains(key.view());
  }
  MATXSCRIPT_ALWAYS_INLINE bool contains(const char32_t* key) const {
    return this->contains(unicode_view(key));
  }

  template <class U, typename = typename std::enable_if<!is_runtime_value<U>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE bool contains(const U& key) const {
    return this->contains(static_cast<const Any&>(GenericValueConverter<RTView>{}(key)));
  }

  DictItems<Dict> items() const;

  DictKeys<Dict> keys() const;

  DictValues<Dict> values() const;

  Iterator item_iter() const;

  Iterator key_iter() const;

  Iterator value_iter() const;

 public:
  /*! \return The underlying DictNode */
  DictNode* GetDictNode() const;

  /*! \brief specify container node */
  using ContainerType = DictNode;

 private:
  void Init(const FuncGetNextItemForward& func, bool has_next);
  void Init(const FuncGetNextItemRandom& func, size_t len);
};

namespace TypeIndex {
template <>
struct type_index_traits<Dict> {
  static constexpr int32_t value = kRuntimeDict;
};
}  // namespace TypeIndex

template <>
MATXSCRIPT_ALWAYS_INLINE Dict Any::As<Dict>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeDict);
  return Dict(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
MATXSCRIPT_ALWAYS_INLINE Dict Any::AsNoCheck<Dict>() const {
  return Dict(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

std::ostream& operator<<(std::ostream& os, Dict const& n);  // namespace runtime

template <>
bool IsConvertible<Dict>(const Object* node);

}  // namespace runtime
}  // namespace matxscript
