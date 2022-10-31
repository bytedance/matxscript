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

#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <matxscript/runtime/_is_comparable.h>
#include <matxscript/runtime/container/_flat_hash_map.h>
#include <matxscript/runtime/container/iterator_adapter.h>
#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/memory.h>
#include <matxscript/runtime/py_args.h>

#include "_ft_object_base.h"

namespace matxscript {
namespace runtime {

template <class I>
struct ft_item_iterator_adaptator : public I {
  using value_type = typename I::value_type;
  using pointer = value_type*;
  using reference = value_type&;
  ft_item_iterator_adaptator() = default;
  ft_item_iterator_adaptator(I const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  ft_item_iterator_adaptator(U const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  ft_item_iterator_adaptator(ft_item_iterator_adaptator<U> const& other)
      : ft_item_iterator_adaptator(*(static_cast<const U*>(&other))) {
  }
  value_type operator*() const {
    return I::operator*();
  }
};

template <class I>
struct ft_key_iterator_adaptator : public I {
  using value_type = typename I::value_type::first_type;
  using pointer = typename I::value_type::first_type*;
  using reference = typename I::value_type::first_type&;
  ft_key_iterator_adaptator() = default;
  ft_key_iterator_adaptator(I const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  ft_key_iterator_adaptator(U const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  ft_key_iterator_adaptator(ft_key_iterator_adaptator<U> const& other)
      : ft_key_iterator_adaptator(*(static_cast<const U*>(&other))) {
  }
  value_type operator*() const {
    return (*this)->first;
  }
};

template <class I>
struct ft_value_iterator_adaptator : public I {
  using value_type = typename I::value_type::second_type;
  using pointer = typename I::value_type::second_type*;
  using reference = typename I::value_type::second_type&;
  ft_value_iterator_adaptator() = default;
  ft_value_iterator_adaptator(I const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  ft_value_iterator_adaptator(U const& i) : I(i) {
  }
  template <class U, typename = typename std::enable_if<std::is_convertible<U, I>::value>::type>
  ft_value_iterator_adaptator(ft_value_iterator_adaptator<U> const& other)
      : ft_value_iterator_adaptator(*(static_cast<const U*>(&other))) {
  }
  value_type operator*() const {
    return (*this)->second;
  }
};

template <class D>
struct FTDictItems {
  using iterator = typename D::item_const_iterator;
  using value_type = typename iterator::value_type;
  D data;
  FTDictItems() = default;
  FTDictItems(D const& d) : data(d) {
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
struct FTDictKeys {
  using iterator = typename D::key_const_iterator;
  using value_type = typename iterator::value_type;
  D data;
  FTDictKeys() = default;
  FTDictKeys(D const& d) : data(d) {
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
struct FTDictValues {
  using iterator = typename D::value_const_iterator;
  using value_type = typename iterator::value_type;
  D data;
  FTDictValues() = default;
  FTDictValues(D const& d) : data(d) {
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

template <typename K, typename V>
class MATX_DLL FTDict;

template <typename K, typename V>
struct MATX_DLL FTDictNode : public FTObjectBaseNode {
 public:
  // data holder
  using key_type = typename std::remove_cv<typename std::remove_reference<K>::type>::type;
  using mapped_type = typename std::remove_cv<typename std::remove_reference<V>::type>::type;
  using container_type = ska::flat_hash_map<key_type, mapped_type, SmartHash, SmartEqualTo>;
  using value_type = typename container_type::value_type;

 public:
  MATXSCRIPT_INLINE_VISIBILITY ~FTDictNode() = default;
  // constructors
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode()
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode(FTDictNode&& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode(const FTDictNode& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode& operator=(FTDictNode&& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode& operator=(const FTDictNode& other) = default;

  template <typename IterType>
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode(IterType first, IterType last)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(first, last) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode(std::initializer_list<value_type> init)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(init) {
  }
  MATXSCRIPT_INLINE_VISIBILITY explicit FTDictNode(container_type init)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(std::move(init)) {
  }

 public:
  static const uint64_t type_tag_;
  static const std::type_index std_type_index_;
  static const FTObjectBaseNode::FunctionTable function_table_;
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeFTDict;
  static constexpr const char* _type_key = "FTDict";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FTDictNode, FTObjectBaseNode);

 public:
  container_type data_;
};

template <typename K, typename V>
class MATX_DLL FTDict : public FTObjectBase {
 public:
  // data holder
  using key_type = typename std::remove_cv<typename std::remove_reference<K>::type>::type;
  using mapped_type = typename std::remove_cv<typename std::remove_reference<V>::type>::type;

 public:
  using ContainerType = FTDictNode<key_type, mapped_type>;
  using container_type = typename ContainerType::container_type;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 private:
  // TODO: support custom object eq
  template <class U>
  struct is_comparable_with_key {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    using type = typename std::conditional<
        std::is_same<key_type, U_TYPE>::value ||
            (std::is_arithmetic<key_type>::value && std::is_arithmetic<U_TYPE>::value) ||
            std::is_base_of<Any, key_type>::value || std::is_base_of<Any, U_TYPE>::value ||
            root_type_is_convertible<key_type, U_TYPE>::value,
        std::true_type,
        std::false_type>::type;
    static constexpr bool value = type::value;
  };

 public:
  // types
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using iterator = ft_key_iterator_adaptator<typename container_type::iterator>;
  using const_iterator = ft_key_iterator_adaptator<typename container_type::const_iterator>;
  using item_iterator = ft_item_iterator_adaptator<typename container_type::iterator>;
  using item_const_iterator = ft_item_iterator_adaptator<typename container_type::const_iterator>;
  using key_iterator = ft_key_iterator_adaptator<typename container_type::iterator>;
  using key_const_iterator = ft_key_iterator_adaptator<typename container_type::const_iterator>;
  using value_iterator = ft_value_iterator_adaptator<typename container_type::iterator>;
  using value_const_iterator = ft_value_iterator_adaptator<typename container_type::const_iterator>;
  using size_type = typename container_type::size_type;
  using difference_type = typename container_type::difference_type;
  using value_type = typename container_type::value_type;
  using allocator_type = typename container_type::allocator_type;
  using pointer = typename container_type::pointer;
  using const_pointer = typename container_type::const_pointer;

  // iterators
  MATXSCRIPT_INLINE_VISIBILITY iterator begin() const {
    return typename FTDict::iterator(MutableImpl().begin());
  }
  MATXSCRIPT_INLINE_VISIBILITY iterator end() const {
    return typename FTDict::iterator(MutableImpl().end());
  }
  MATXSCRIPT_INLINE_VISIBILITY item_iterator item_begin() const {
    return ft_item_iterator_adaptator<typename FTDict::container_type::iterator>(
        MutableImpl().begin());
  }
  MATXSCRIPT_INLINE_VISIBILITY item_iterator item_end() const {
    return ft_item_iterator_adaptator<typename FTDict::container_type::iterator>(
        MutableImpl().end());
  }
  MATXSCRIPT_INLINE_VISIBILITY key_iterator key_begin() const {
    return ft_key_iterator_adaptator<typename FTDict::container_type::iterator>(
        MutableImpl().begin());
  }
  MATXSCRIPT_INLINE_VISIBILITY key_iterator key_end() const {
    return ft_key_iterator_adaptator<typename FTDict::container_type::iterator>(
        MutableImpl().end());
  }
  MATXSCRIPT_INLINE_VISIBILITY value_iterator value_begin() const {
    return ft_value_iterator_adaptator<typename FTDict::container_type::iterator>(
        MutableImpl().begin());
  }
  MATXSCRIPT_INLINE_VISIBILITY value_iterator value_end() const {
    return ft_value_iterator_adaptator<typename FTDict::container_type::iterator>(
        MutableImpl().end());
  }

  MATXSCRIPT_INLINE_VISIBILITY ~FTDict() noexcept = default;

  // constructors
  MATXSCRIPT_INLINE_VISIBILITY FTDict()
      : FTObjectBase(make_object<FTDictNode<key_type, mapped_type>>()) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTDict(FTDict&& other) noexcept = default;
  MATXSCRIPT_INLINE_VISIBILITY FTDict(const FTDict& other) noexcept = default;
  MATXSCRIPT_INLINE_VISIBILITY FTDict& operator=(FTDict&& other) noexcept = default;
  MATXSCRIPT_INLINE_VISIBILITY FTDict& operator=(const FTDict& other) noexcept = default;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  MATXSCRIPT_INLINE_VISIBILITY explicit FTDict(ObjectPtr<Object> n) noexcept
      : FTObjectBase(std::move(n)) {
  }

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  MATXSCRIPT_INLINE_VISIBILITY FTDict(IterType first, IterType last)
      : FTObjectBase(make_object<FTDictNode<key_type, mapped_type>>(first, last)) {
  }

  /*!
   * \brief constructor from initializer FTDict
   * \param init The initializer FTDict
   */
  MATXSCRIPT_INLINE_VISIBILITY FTDict(std::initializer_list<value_type> init)
      : FTObjectBase(make_object<FTDictNode<key_type, mapped_type>>(init)) {
  }

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  MATXSCRIPT_INLINE_VISIBILITY FTDict(const std::vector<value_type>& init)
      : FTObjectBase(make_object<FTDictNode<key_type, mapped_type>>(init.begin(), init.end())) {
  }

  template <class UK, class UV>
  MATXSCRIPT_INLINE_VISIBILITY bool operator==(const FTDict<UK, UV>& other) const {
    return this->__eq__(other);
  }

  template <class UK, class UV>
  MATXSCRIPT_INLINE_VISIBILITY bool operator!=(const FTDict<UK, UV>& other) const {
    return !operator==(other);
  }

 public:
  // method for python
  MATXSCRIPT_INLINE_VISIBILITY Iterator key_iter() const {
    auto iterator_ptr = std::make_shared<key_iterator>(this->key_begin());
    auto* iter_c = iterator_ptr.get();
    auto iter_end = this->key_end();
    auto has_next = [iter_c, iterator_ptr, iter_end]() -> bool { return *iter_c != iter_end; };
    auto next = [iter_c, iter_end]() -> RTValue {
      RTValue r = key_type(*(*iter_c));
      ++(*iter_c);
      return r;
    };
    auto next_and_checker = [iter_c, iter_end](bool* has_next) -> RTValue {
      RTValue r = key_type(*(*iter_c));
      ++(*iter_c);
      *has_next = (*iter_c != iter_end);
      return r;
    };
    return Iterator::MakeGenericIterator(
        *this, std::move(has_next), std::move(next), std::move(next_and_checker));
  }

  MATXSCRIPT_INLINE_VISIBILITY Iterator value_iter() const {
    auto iterator_ptr = std::make_shared<value_iterator>(this->value_begin());
    auto* iter_c = iterator_ptr.get();
    auto iter_end = this->value_end();
    auto has_next = [iter_c, iterator_ptr, iter_end]() -> bool { return *iter_c != iter_end; };
    auto next = [iter_c, iter_end]() -> RTValue {
      RTValue r = mapped_type(*(*iter_c));
      ++(*iter_c);
      return r;
    };
    auto next_and_checker = [iter_c, iter_end](bool* has_next) -> RTValue {
      RTValue r = mapped_type(*(*iter_c));
      ++(*iter_c);
      *has_next = (*iter_c != iter_end);
      return r;
    };
    return Iterator::MakeGenericIterator(
        *this, std::move(has_next), std::move(next), std::move(next_and_checker));
  }

  MATXSCRIPT_INLINE_VISIBILITY Iterator item_iter() const {
    auto iterator_ptr = std::make_shared<item_iterator>(this->item_begin());
    auto* iter_c = iterator_ptr.get();
    auto iter_end = this->item_end();
    auto has_next = [iter_c, iterator_ptr, iter_end]() -> bool { return *iter_c != iter_end; };
    auto next = [iter_c, iter_end]() -> RTValue {
      RTValue r = Tuple(*(*iter_c));
      ++(*iter_c);
      return r;
    };
    auto next_and_checker = [iter_c, iter_end](bool* has_next) -> RTValue {
      RTValue r = Tuple(*(*iter_c));
      ++(*iter_c);
      *has_next = (*iter_c != iter_end);
      return r;
    };
    return Iterator::MakeGenericIterator(
        *this, std::move(has_next), std::move(next), std::move(next_and_checker));
  }

  MATXSCRIPT_INLINE_VISIBILITY Iterator __iter__() const {
    return this->key_iter();
  }

  template <class UK, class UV>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTDict<UK, UV>& o, std::true_type) const {
    return MutableImpl() == o.MutableImpl();
  }

  template <class UK, class UV>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTDict<UK, UV>& o, std::false_type) const {
    return Iterator::all_items_equal(item_iter(), o.item_iter());
  }

  template <class UK, class UV>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTDict<UK, UV>& o) const {
    using UK_TYPE = typename std::remove_cv<typename std::remove_reference<UK>::type>::type;
    using UV_TYPE = typename std::remove_cv<typename std::remove_reference<UK>::type>::type;
    return __eq__(o,
                  typename std::conditional < std::is_same<UK_TYPE, key_type>::value &&
                      std::is_same<UV_TYPE, mapped_type>::value,
                  std::true_type,
                  std::false_type > ::type{});
  }

  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const Any& o) const {
    if (o.type_code() == TypeIndex::kRuntimeFTDict || o.type_code() == TypeIndex::kRuntimeDict) {
      return Iterator::all_items_equal(item_iter(), Iterator::MakeItemsIterator(o));
    }
    return false;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type& __getitem__(KEY_U const& key, std::true_type) const {
    auto& data_impl = MutableImpl();
    auto iter = data_impl.find(key);
    MXCHECK(iter != data_impl.end()) << "Dict[" << key << "] not found";
    return iter->second;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type& __getitem__(KEY_U const& key, std::false_type) const {
    MXTHROW << "Dict[" << key << "] not found";
    return MutableImpl().begin()->second;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type& __getitem__(KEY_U const& key) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return __getitem__(key, typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type& get_item(KEY_U const& key) const {
    return __getitem__(key);
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type& get_item(const char* const key) const {
    return this->__getitem__(string_view(key));
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type& get_item(const char32_t* const key) const {
    return this->__getitem__(unicode_view(key));
  }

  template <class KEY_U,
            class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(KEY_U const& key,
                                                   DefaultType&& default_val,
                                                   std::true_type) const {
    auto& data_impl = MutableImpl();
    auto it = data_impl.find(key);
    if (it == data_impl.end()) {
      return std::forward<DefaultType>(default_val);
    }
    return it->second;
  }

  template <class KEY_U,
            class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(KEY_U const& key,
                                                   DefaultType&& default_val,
                                                   std::false_type) const {
    return std::forward<DefaultType>(default_val);
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type const& get_default(KEY_U const& key,
                                                              mapped_type const& default_val,
                                                              std::true_type) const {
    auto& data_impl = MutableImpl();
    auto it = data_impl.find(key);
    return it == data_impl.end() ? default_val : it->second;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type const& get_default(KEY_U const& key,
                                                              mapped_type const& default_val,
                                                              std::false_type) const {
    return default_val;
  }

  template <class KEY_U,
            class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(KEY_U const& key,
                                                   DefaultType&& default_val) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return this->get_default(key,
                             std::forward<DefaultType>(default_val),
                             typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  template <class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(const char* const key,
                                                   DefaultType&& default_val) const {
    return this->get_default(string_view(key), std::forward<DefaultType>(default_val));
  }

  template <class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(const char32_t* const key,
                                                   DefaultType&& default_val) const {
    return this->get_default(unicode_view(key), std::forward<DefaultType>(default_val));
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type const& get_default(
      KEY_U const& key, mapped_type const& default_val) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return this->get_default(key, default_val, typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type const& get_default(
      const char* const key, mapped_type const& default_val) const {
    return this->get_default(string_view(key), default_val);
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type const& get_default(
      const char32_t* const key, mapped_type const& default_val) const {
    return this->get_default(unicode_view(key), default_val);
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(KEY_U const& key) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return this->get_default(key, None, typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(const char* const key) const {
    return this->get_default(string_view(key), None);
  }

  MATXSCRIPT_INLINE_VISIBILITY RTValue get_default(const char32_t* const key) const {
    return this->get_default(unicode_view(key), None);
  }

  template <class KEY_U,
            class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue pop(KEY_U const& key,
                                           DefaultType&& default_val,
                                           std::true_type) const {
    auto& data_impl = MutableImpl();
    auto it = data_impl.find(key);
    if (it == data_impl.end()) {
      return std::forward<DefaultType>(default_val);
    }
    auto ret = std::move(it->second);
    data_impl.erase(it);
    return ret;
  }

  template <class KEY_U,
            class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue pop(KEY_U const& key,
                                           DefaultType&& default_val,
                                           std::false_type) const {
    return std::forward<DefaultType>(default_val);
  }

  template <class KEY_U,
            class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue pop(KEY_U const& key, DefaultType&& default_val) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return pop(key,
               std::forward<DefaultType>(default_val),
               typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  template <class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue pop(const char* const key, DefaultType&& default_val) const {
    return pop(string_view(key), std::forward<DefaultType>(default_val));
  }

  template <class DefaultType,
            typename = typename std::enable_if<!std::is_same<
                mapped_type,
                typename std::remove_cv<typename std::remove_reference<DefaultType>::type>::type>::
                                                   value>::type>
  MATXSCRIPT_INLINE_VISIBILITY RTValue pop(const char32_t* const key,
                                           DefaultType&& default_val) const {
    return pop(unicode_view(key), std::forward<DefaultType>(default_val));
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(KEY_U const& key,
                                               mapped_type const& default_val,
                                               std::true_type) const {
    auto& data_impl = MutableImpl();
    auto it = data_impl.find(key);
    if (it == data_impl.end()) {
      return default_val;
    }
    auto ret = std::move(it->second);
    data_impl.erase(it);
    return ret;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(KEY_U const& key,
                                               mapped_type const& default_val,
                                               std::false_type) const {
    return default_val;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(KEY_U const& key,
                                               mapped_type const& default_val) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return pop(key, default_val, typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(const char* const key,
                                               mapped_type const& default_val) const {
    return pop(string_view(key), default_val);
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(const char32_t* const key,
                                               mapped_type const& default_val) const {
    return pop(unicode_view(key), default_val);
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(KEY_U const& key, std::true_type) const {
    auto& data_impl = MutableImpl();
    auto it = data_impl.find(key);
    if (it == data_impl.end()) {
      MXTHROW << "dict.pop KeyError";
    }
    auto ret = std::move(it->second);
    data_impl.erase(it);
    return ret;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(KEY_U const& key, std::false_type) const {
    MXTHROW << "dict.pop KeyError";
    return MutableImpl().begin()->second;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(KEY_U const& key) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return pop(key, typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(const char* const key) const {
    return pop(string_view(key));
  }

  MATXSCRIPT_INLINE_VISIBILITY mapped_type pop(const char32_t* const key) const {
    return pop(unicode_view(key));
  }

  template <class KEY_U, class MAPPED_U>
  MATXSCRIPT_INLINE_VISIBILITY void set_item(KEY_U&& key, MAPPED_U&& value) const {
    GenericValueConverter<key_type> KeyConv;
    GenericValueConverter<mapped_type> MappedConv;
    auto& data_impl = MutableImpl();
    data_impl[KeyConv(std::forward<KEY_U>(key))] = MappedConv(std::forward<MAPPED_U>(value));
  }

  // mutation in std::unordered_map
  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY mapped_type& operator[](KEY_U&& key) const {
    using Converter = GenericValueConverter<key_type>;
    return MutableImpl()[Converter()(std::forward<KEY_U>(key))];
  }

  template <typename Key, typename... Args>
  MATXSCRIPT_INLINE_VISIBILITY std::pair<iterator, bool> emplace(Key&& key, Args&&... args) const {
    return MutableImpl().emplace(std::forward<Key>(key), std::forward<Args>(args)...);
  }

  MATXSCRIPT_INLINE_VISIBILITY std::pair<iterator, bool> insert(const value_type& value) const {
    return MutableImpl().insert(value);
  }
  MATXSCRIPT_INLINE_VISIBILITY std::pair<iterator, bool> insert(value_type&& value) const {
    return MutableImpl().insert(std::move(value));
  }

  MATXSCRIPT_INLINE_VISIBILITY void clear() const {
    MutableImpl().clear();
  }

  MATXSCRIPT_INLINE_VISIBILITY void reserve(int64_t new_size) const {
    if (new_size > 0) {
      MutableImpl().reserve(new_size);
    }
  }

 public:
  // const methods in std::unordered_map
  MATXSCRIPT_INLINE_VISIBILITY int64_t size() const {
    return MutableImpl().size();
  }

  MATXSCRIPT_INLINE_VISIBILITY int64_t bucket_count() const {
    return MutableImpl().bucket_count();
  }

  MATXSCRIPT_INLINE_VISIBILITY bool empty() const {
    return MutableImpl().empty();
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(KEY_U const& key, std::true_type) const {
    auto& data_impl = MutableImpl();
    return data_impl.find(key) != data_impl.end();
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(KEY_U const& key, std::false_type) const {
    return false;
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(KEY_U const& key) const {
    using KEY_U_TYPE = typename std::remove_cv<typename std::remove_reference<KEY_U>::type>::type;
    return __contains__(key, typename is_comparable_with_key<KEY_U_TYPE>::type{});
  }

  template <class KEY_U>
  MATXSCRIPT_INLINE_VISIBILITY bool contains(KEY_U const& key) const {
    return this->__contains__(key);
  }

  MATXSCRIPT_INLINE_VISIBILITY bool contains(const char* const key) const {
    return this->__contains__(string_view(key));
  }

  MATXSCRIPT_INLINE_VISIBILITY bool contains(const char32_t* const key) const {
    return this->__contains__(unicode_view(key));
  }

  MATXSCRIPT_INLINE_VISIBILITY FTDictItems<FTDict> items() const {
    return FTDictItems<FTDict>(*this);
  }

  MATXSCRIPT_INLINE_VISIBILITY FTDictKeys<FTDict> keys() const {
    return FTDictKeys<FTDict>(*this);
  }

  MATXSCRIPT_INLINE_VISIBILITY FTDictValues<FTDict> values() const {
    return FTDictValues<FTDict>(*this);
  }

 private:
  MATXSCRIPT_INLINE_VISIBILITY container_type& MutableImpl() const {
    return static_cast<FTDictNode<key_type, mapped_type>*>(data_.get())->data_;
  }
  MATXSCRIPT_INLINE_VISIBILITY FTDictNode<key_type, mapped_type>* MutableNode() const {
    return static_cast<FTDictNode<key_type, mapped_type>*>(data_.get());
  }
};

namespace TypeIndex {
template <typename K, typename V>
struct type_index_traits<FTDict<K, V>> {
  static constexpr int32_t value = kRuntimeFTDict;
};
}  // namespace TypeIndex

// python methods
#define MATXSCRIPT_CHECK_FT_DICT_ARGS(FuncName, NumArgs)                                  \
  MXCHECK(NumArgs == args.size()) << "[" << DemangleType(typeid(FTDictNode<K, V>).name()) \
                                  << "::" << #FuncName << "] Expect " << NumArgs          \
                                  << " arguments but get " << args.size()

template <typename K, typename V>
const uint64_t FTDictNode<K, V>::type_tag_ = std::hash<string_view>()(typeid(FTDict<K, V>).name());
template <typename K, typename V>
const std::type_index FTDictNode<K, V>::std_type_index_ = typeid(FTDict<K, V>);
template <typename K, typename V>
const FTObjectBaseNode::FunctionTable FTDictNode<K, V>::function_table_ = {
    {"__len__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(__len__, 0);
       return self.AsObjectView<FTDict<K, V>>().data().size();
     }},
    {"__contains__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(__contains__, 1);
       return self.AsObjectView<FTDict<K, V>>().data().contains(args[0].template As<RTValue>());
     }},
    {"__getitem__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(__getitem__, 1);
       return mapped_type(
           self.AsObjectView<FTDict<K, V>>().data().get_item(args[0].template As<RTValue>()));
     }},
    {"__setitem__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(__setitem__, 2);
       self.AsObjectView<FTDict<K, V>>().data().set_item(args[0].template As<RTValue>(),
                                                         args[1].template As<RTValue>());
       return None;
     }},
    {"__iter__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(__iter__, 0);
       return self.AsObjectView<FTDict<K, V>>().data().key_iter();
     }},
    {"__eq__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(__eq__, 1);
       return self.AsObjectView<FTDict<K, V>>().data().__eq__(args[0].template As<RTValue>());
     }},
    {"clear",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(clear, 0);
       self.AsObjectView<FTDict<K, V>>().data().clear();
       return None;
     }},
    {"items",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(items, 0);
       return self.AsObjectView<FTDict<K, V>>().data().item_iter();
     }},
    {"keys",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(keys, 0);
       return self.AsObjectView<FTDict<K, V>>().data().key_iter();
     }},
    {"values",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_DICT_ARGS(values, 0);
       return self.AsObjectView<FTDict<K, V>>().data().value_iter();
     }},
    {"pop",
     [](RTView self, PyArgs args) -> RTValue {
       static const String cls_name = DemangleType(typeid(FTDict<K, V>).name());
       const auto& view = self.AsObjectView<FTDict<K, V>>();
       switch (args.size()) {
         case 1: {
           return view.data().pop(args[0].template As<RTValue>());
         } break;
         case 2: {
           return view.data().pop(args[0].template As<RTValue>(), args[1].template As<RTValue>());
         } break;
         default: {
           MXTHROW << "[" << cls_name << "::pop] Expect 1 or 2 arguments but get " << args.size();
         } break;
       }
       return None;
     }},
    {"setdefault",
     [](RTView self, PyArgs args) -> RTValue {
       MXTHROW << "dict.setdefault is not supported";
       return None;
     }},
    {"get",
     [](RTView self, PyArgs args) -> RTValue {
       static const String cls_name = DemangleType(typeid(FTDict<K, V>).name());
       const auto& view = self.AsObjectView<FTDict<K, V>>();
       switch (args.size()) {
         case 1: {
           return view.data().get_default(args[0].template As<RTValue>());
         } break;
         case 2: {
           return view.data().get_default(args[0].template As<RTValue>(),
                                          args[1].template As<RTValue>());
         } break;
         default: {
           MXTHROW << "[" << cls_name << "::get] Expect 1 or 2 arguments but get " << args.size();
         } break;
       }
       return None;
     }},
    {"popitem",
     [](RTView self, PyArgs args) -> RTValue {
       MXTHROW << "dict.popitem is not supported";
       return None;
     }},
    {"update",
     [](RTView self, PyArgs args) -> RTValue {
       MXTHROW << "dict.update is not supported";
       return None;
     }},
};

#undef MATXSCRIPT_CHECK_FT_DICT_ARGS

template <typename K, typename V>
static inline std::ostream& operator<<(std::ostream& os, FTDict<K, V> const& obj) {
  os << '{';
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    if (it != obj.begin()) {
      os << ", ";
    }
    if (std::is_same<K, String>::value) {
      os << "b'" << it->first << "': ";
    } else if (std::is_same<K, Unicode>::value) {
      os << "\'" << it->first << "\': ";
    } else {
      os << it->first << ": ";
    }

    if (std::is_same<V, String>::value) {
      os << "b'" << it->second << "'";
    } else if (std::is_same<V, Unicode>::value) {
      os << "\'" << it->second << "\'";
    } else {
      os << it->first;
    }
  }
  os << '}';
  return os;
}

}  // namespace runtime
}  // namespace matxscript
