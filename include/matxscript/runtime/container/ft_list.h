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
#include <matxscript/runtime/container/_list_helper.h>
#include <matxscript/runtime/container/container_slice_helper.h>
#include <matxscript/runtime/container/string.h>
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/container/user_data_ref.h>
#include <matxscript/runtime/logging.h>
#include <matxscript/runtime/memory.h>

#include "_ft_object_base.h"
#include "iterator_adapter.h"

namespace matxscript {
namespace runtime {

template <typename T>
class MATX_DLL FTList;

template <typename T>
struct MATX_DLL FTListNode : public FTObjectBaseNode {
 public:
  // data holder
  using value_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  using container_type = std::vector<value_type>;

  // types
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using size_type = typename container_type::size_type;
  using difference_type = typename container_type::difference_type;
  using allocator_type = typename container_type::allocator_type;
  using pointer = typename container_type::pointer;
  using const_pointer = typename container_type::const_pointer;
  using reverse_iterator = typename container_type::reverse_iterator;
  using const_reverse_iterator = typename container_type::const_reverse_iterator;

 public:
  MATXSCRIPT_INLINE_VISIBILITY ~FTListNode() = default;
  // constructors
  MATXSCRIPT_INLINE_VISIBILITY FTListNode()
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTListNode(FTListNode&& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTListNode(const FTListNode& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTListNode& operator=(FTListNode&& other) = default;
  MATXSCRIPT_INLINE_VISIBILITY FTListNode& operator=(const FTListNode& other) = default;

  template <typename IterType>
  MATXSCRIPT_INLINE_VISIBILITY FTListNode(IterType first, IterType last)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(first, last) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTListNode(std::initializer_list<value_type> init)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(init) {
  }
  MATXSCRIPT_INLINE_VISIBILITY explicit FTListNode(container_type init)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(std::move(init)) {
  }
  MATXSCRIPT_INLINE_VISIBILITY FTListNode(size_t n, const value_type& val)
      : FTObjectBaseNode(&function_table_, &std_type_index_, type_tag_), data_(n, val) {
  }

 public:
  static const uint64_t type_tag_;
  static const std::type_index std_type_index_;
  static const FTObjectBaseNode::FunctionTable function_table_;
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeFTList;
  static constexpr const char* _type_key = "FTList";
  MATXSCRIPT_DECLARE_FINAL_OBJECT_INFO(FTListNode, FTObjectBaseNode);

 public:
  container_type data_;
};

template <typename T>
class MATX_DLL FTList : public FTObjectBase {
 public:
  // data holder
  using value_type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  using container_type = std::vector<value_type>;

 public:
  using ContainerType = FTListNode<value_type>;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

  // types
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using size_type = typename container_type::size_type;
  using difference_type = typename container_type::difference_type;
  using allocator_type = typename container_type::allocator_type;
  using pointer = typename container_type::pointer;
  using const_pointer = typename container_type::const_pointer;
  using reverse_iterator = typename container_type::reverse_iterator;
  using const_reverse_iterator = typename container_type::const_reverse_iterator;

 private:
  // TODO: support custom object eq
  template <class U>
  struct is_comparable_with_value {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    using type = typename std::conditional<
        std::is_same<value_type, U_TYPE>::value ||
            (std::is_arithmetic<value_type>::value && std::is_arithmetic<U_TYPE>::value) ||
            std::is_base_of<Any, value_type>::value || std::is_base_of<Any, U_TYPE>::value ||
            root_type_is_convertible<value_type, U_TYPE>::value,
        std::true_type,
        std::false_type>::type;
    static constexpr bool value = type::value;
  };

 public:
  MATXSCRIPT_INLINE_VISIBILITY ~FTList() noexcept = default;
  // constructors
  /*!
   * \brief default constructor
   */
  MATXSCRIPT_INLINE_VISIBILITY FTList() : FTObjectBase(make_object<FTListNode<value_type>>()) {
  }

  /*!
   * \brief move constructor
   * \param other source
   */
  MATXSCRIPT_INLINE_VISIBILITY FTList(FTList&& other) noexcept = default;

  /*!
   * \brief copy constructor
   * \param other source
   */
  MATXSCRIPT_INLINE_VISIBILITY FTList(const FTList& other) noexcept = default;

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  MATXSCRIPT_INLINE_VISIBILITY explicit FTList(ObjectPtr<Object> n) noexcept
      : FTObjectBase(std::move(n)) {
  }

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  MATXSCRIPT_INLINE_VISIBILITY FTList& operator=(FTList&& other) noexcept = default;

  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  MATXSCRIPT_INLINE_VISIBILITY FTList& operator=(const FTList& other) noexcept = default;

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  MATXSCRIPT_INLINE_VISIBILITY FTList(IterType first, IterType last)
      : FTObjectBase(make_object<FTListNode<value_type>>(first, last)) {
  }

  /*!
   * \brief constructor from initializer list
   * \param init The initializer list
   */
  MATXSCRIPT_INLINE_VISIBILITY FTList(std::initializer_list<value_type> init)
      : FTObjectBase(make_object<FTListNode<value_type>>(init)) {
  }

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  MATXSCRIPT_INLINE_VISIBILITY explicit FTList(container_type init)
      : FTObjectBase(make_object<FTListNode<value_type>>(init)) {
  }

  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   */
  MATXSCRIPT_INLINE_VISIBILITY FTList(size_t n, const value_type& v)
      : FTObjectBase(make_object<FTListNode<value_type>>(n, v)) {
  }

  MATXSCRIPT_INLINE_VISIBILITY value_type& operator[](int64_t i) const {
    return this->MutableImpl()[i];
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator==(const FTList<U>& other) const {
    return this->__eq__(other);
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator!=(const FTList<U>& other) const {
    return !operator==(other);
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator>(const FTList<U>& other) const {
    auto& cons = MutableImpl();
    auto r_cons = other.MutableImpl();
    auto l_it = cons.begin();
    auto r_it = r_cons.begin();
    for (; l_it != cons.end() && r_it != r_cons.end(); ++l_it, ++r_it) {
      if (*l_it > *r_it) {
        return true;
      } else if (*r_it > *l_it) {
        return false;
      }
    }
    if (l_it != cons.end()) {
      return true;
    } else {
      return false;
    }
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator>=(const FTList<U>& other) const {
    auto& cons = MutableImpl();
    auto r_cons = other.MutableImpl();
    auto l_it = cons.begin();
    auto r_it = r_cons.begin();
    for (; l_it != cons.end() && r_it != r_cons.end(); ++l_it, ++r_it) {
      if (*l_it > *r_it) {
        return true;
      } else if (*r_it > *l_it) {
        return false;
      }
    }
    if (r_it == r_cons.end()) {
      return true;
    } else {
      return false;
    }
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator<(const FTList<U>& other) const {
    return other.operator>(*this);
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool operator<=(const FTList<U>& other) const {
    return other.operator>=(*this);
  }

 public:
  // iterators
  MATXSCRIPT_INLINE_VISIBILITY iterator begin() const {
    return MutableImpl().begin();
  }
  MATXSCRIPT_INLINE_VISIBILITY iterator end() const {
    return MutableImpl().end();
  }
  MATXSCRIPT_INLINE_VISIBILITY reverse_iterator rbegin() const {
    return MutableImpl().rbegin();
  }
  MATXSCRIPT_INLINE_VISIBILITY reverse_iterator rend() const {
    return MutableImpl().rend();
  }

  // stl methods
  MATXSCRIPT_INLINE_VISIBILITY int64_t size() const {
    return MutableImpl().size();
  }

  MATXSCRIPT_INLINE_VISIBILITY int64_t capacity() const {
    return MutableImpl().capacity();
  }

  MATXSCRIPT_INLINE_VISIBILITY bool empty() const {
    return MutableImpl().empty();
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY void push_back(U&& item) const {
    MutableImpl().push_back(std::forward<U>(item));
  }

  MATXSCRIPT_INLINE_VISIBILITY void pop_back() const {
    MutableImpl().pop_back();
  }

  template <class... Args>
  MATXSCRIPT_INLINE_VISIBILITY auto emplace_back(Args&&... args) const {
    return MutableImpl().template emplace_back(std::forward<Args>(args)...);
  }

  MATXSCRIPT_INLINE_VISIBILITY void reserve(int64_t new_size) const {
    if (new_size > 0) {
      MutableImpl().reserve(new_size);
    }
  }

  MATXSCRIPT_INLINE_VISIBILITY void resize(int64_t new_size) const {
    if (new_size >= 0) {
      MutableImpl().resize(new_size);
    }
  }

 public:
  // method for python
  MATXSCRIPT_INLINE_VISIBILITY Iterator __iter__() const {
    auto iterator_ptr = std::make_shared<iterator>(this->begin());
    auto* iter_c = iterator_ptr.get();
    auto iter_end = this->end();
    auto has_next = [iter_c, iterator_ptr, iter_end]() -> bool { return *iter_c != iter_end; };
    auto next = [iter_c, iter_end]() -> RTValue {
      RTValue r = value_type(*(*iter_c));
      ++(*iter_c);
      return r;
    };
    auto next_and_checker = [iter_c, iter_end](bool* has_next) -> RTValue {
      RTValue r = value_type(*(*iter_c));
      ++(*iter_c);
      *has_next = (*iter_c != iter_end);
      return r;
    };
    return Iterator::MakeGenericIterator(
        *this, std::move(has_next), std::move(next), std::move(next_and_checker));
  }

  MATXSCRIPT_INLINE_VISIBILITY Iterator iter() const {
    return __iter__();
  }

  MATXSCRIPT_INLINE_VISIBILITY int64_t _check_index_error(int64_t i) const {
    int64_t len = MutableImpl().size();
    MXCHECK((i >= 0 && i < len) || (i < 0 && i >= -len)) << "ValueError: index overflow";
    return slice_index_correction(i, len);
  }

  // python methods
  MATXSCRIPT_INLINE_VISIBILITY int64_t __len__() const {
    return MutableImpl().size();
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(const U& item, std::true_type) const {
    auto& cons = MutableImpl();
    return std::find(cons.begin(), cons.end(), item) != cons.end();
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY bool __contains__(const U& item, std::false_type) const {
    if ((!is_comparable_with_value<U>::value)) {
      return false;
    }
    RTView item_view(item);
    SmartEqualTo func_equal;
    auto pred = [&item_view, &func_equal](reference val) { return func_equal(item_view, val); };
    auto& cons = MutableImpl();
    return std::find_if(cons.begin(), cons.end(), pred) != cons.end();
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY uint64_t
  IndexImpl(const U& item, int64_t start, int64_t end, std::true_type) const {
    auto& cons = MutableImpl();
    return std::find(cons.begin() + start, cons.begin() + end, item) - cons.begin();
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY uint64_t
  IndexImpl(const U& item, int64_t start, int64_t end, std::false_type) const {
    if ((!is_comparable_with_value<U>::value)) {
      return -1;
    }
    RTView item_view(item);
    SmartEqualTo func_equal;
    auto pred = [&item_view, &func_equal](reference val) { return func_equal(item_view, val); };
    auto& cons = MutableImpl();
    return std::find_if(cons.begin() + start, cons.begin() + end, pred) - cons.begin();
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY bool contains(const U& item) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    return this->__contains__(item, std::is_same<U_TYPE, value_type>());
  }

  MATXSCRIPT_INLINE_VISIBILITY reference __getitem__(int64_t i) const {
    return MutableImpl()[_check_index_error(i)];
  }

  MATXSCRIPT_INLINE_VISIBILITY reference get_item(int64_t i) const {
    return this->__getitem__(i);
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY void __setitem__(int64_t i, U&& item) const {
    GenericValueConverter<value_type> Converter;
    MutableImpl()[_check_index_error(i)] = Converter(std::forward<U>(item));
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY void set_item(int64_t i, U&& item) const {
    return this->__setitem__(i, std::forward<U>(item));
  }

  MATXSCRIPT_INLINE_VISIBILITY FTList<value_type> __getslice__(int64_t b,
                                                               int64_t e,
                                                               int64_t step = 1) const {
    MXCHECK_GT(step, 0) << "FTList.slice_load step must be gt 0";
    int64_t len = MutableImpl().size();
    b = slice_index_correction(b, len);
    e = slice_index_correction(e, len);
    if (e <= b) {
      return FTList<value_type>();
    } else {
      if (step == 1) {
        return FTList<value_type>(this->begin() + b, this->begin() + e);
      } else {
        FTList<value_type> new_list;
        new_list.reserve(e - b);
        auto itr_end = begin() + e;
        for (auto itr = begin() + b; itr < itr_end; itr += step) {
          new_list.push_back(*itr);
        }
        return new_list;
      }
    }
  }

  MATXSCRIPT_INLINE_VISIBILITY FTList<value_type> get_slice(int64_t b,
                                                            int64_t e,
                                                            int64_t step = 1) const {
    return this->__getslice__(b, e, step);
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY void __setslice__(int64_t start,
                                                 int64_t end,
                                                 const FTList<U>& rhs) const {
    using Converter = GenericValueConverter<value_type>;
    auto& cons = MutableImpl();
    MXCHECK(start >= 0 && end >= 0 && start <= end);
    int64_t len = cons.size();
    start = slice_index_correction(start, len);
    end = slice_index_correction(end, len);
    cons.erase(cons.begin() + start, cons.begin() + end);
    cons.insert(cons.begin() + start,
                IteratorAdapter<Converter, typename FTList<U>::iterator>(rhs.begin()),
                IteratorAdapter<Converter, typename FTList<U>::iterator>(rhs.end()));
  }

  MATXSCRIPT_INLINE_VISIBILITY void __setslice__(int64_t start, int64_t end, const Any& rhs) const {
    GenericValueConverter<value_type> Converter;
    auto& cons = MutableImpl();
    MXCHECK(start >= 0 && end >= 0 && start <= end);
    int64_t len = cons.size();
    start = slice_index_correction(start, len);
    end = slice_index_correction(end, len);
    std::vector<value_type> rhs_val;
    auto iterable = Iterator::MakeGenericIterator(rhs);
    bool has_next = iterable.HasNext();
    while (has_next) {
      rhs_val.emplace_back(Converter(iterable.Next(&has_next)));
    }
    cons.erase(cons.begin() + start, cons.begin() + end);
    cons.insert(cons.begin() + start, rhs_val.begin(), rhs_val.end());
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY void set_slice(int64_t b, int64_t e, const U& rl) const {
    return this->__setslice__(b, e, rl);
  }

  template <class U, class ItemEqualFunctor>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq_use_functor__(const FTList<U>& o,
                                                       ItemEqualFunctor func) const {
    auto& cons = MutableImpl();
    auto rhs_itr = o.begin();
    for (auto lhs_itr = cons.begin(); lhs_itr != cons.end(); ++lhs_itr, ++rhs_itr) {
      if (!func(*lhs_itr, *rhs_itr)) {
        return false;
      }
    }
    return true;
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTList<U>& o,
                                           type_relation::root_is_convertible) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    auto func_eq = [](const value_type& a, const U_TYPE& b) {
      return Iterator::all_items_equal(Iterator::MakeItemsIterator(a),
                                       Iterator::MakeItemsIterator(b));
    };
    return __eq_use_functor__(o, func_eq);
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTList<U>& o, type_relation::no_rel) const {
    return false;
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTList<U>& o, type_relation::one_is_any) const {
    return __eq_use_functor__(o, SmartEqualTo());
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTList<U>& o, type_relation::same_root) const {
    auto& cons = MutableImpl();
    auto rhs_itr = o.MutableImpl().begin();
    for (auto lhs_itr = cons.begin(); lhs_itr != cons.end(); ++lhs_itr, ++rhs_itr) {
      if (*lhs_itr != *rhs_itr) {
        return false;
      }
    }
    return true;
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const FTList<U>& o) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    if (size() != o.size()) {
      return false;
    }
    return __eq__(o, typename type_relation::traits<U_TYPE, value_type>::type{});
  }

  MATXSCRIPT_INLINE_VISIBILITY bool __eq__(const Any& o) const {
    if (o.type_code() == TypeIndex::kRuntimeFTList || o.type_code() == TypeIndex::kRuntimeList) {
      return Iterator::all_items_equal(__iter__(), Iterator::MakeItemsIterator(o));
    }
    return false;
  }

  MATXSCRIPT_INLINE_VISIBILITY FTList<value_type> __mul__(int64_t times) const {
    FTList<value_type> new_list{};
    if (MATXSCRIPT_UNLIKELY(times <= 0)) {
      return new_list;
    }
    new_list.reserve(times * this->size());
    for (int64_t i = 0; i < times; i++) {
      new_list.extend(*this);
    }
    return new_list;
  }

  MATXSCRIPT_INLINE_VISIBILITY FTList<value_type> repeat(int64_t times) const {
    return this->__mul__(times);
  }

  // for fuse list_make and repeat
  template <typename U>
  static FTList<value_type> repeat_one(U&& value, int64_t times) {
    FTList<value_type> new_list{};
    if (MATXSCRIPT_UNLIKELY(times <= 0)) {
      return new_list;
    }
    new_list.reserve(times);
    for (int64_t i = 0; i < times - 1; i++) {
      new_list.emplace_back(value);
    }
    new_list.emplace_back(std::forward<U>(value));
    return new_list;
  }
  template <typename U>
  static FTList<value_type> repeat_many(const std::initializer_list<U>& values, int64_t times) {
    FTList<value_type> new_list{};
    if (MATXSCRIPT_UNLIKELY(times <= 0)) {
      return new_list;
    }
    new_list.reserve(times * values.size());
    auto this_b = values.begin();
    auto this_e = values.end();
    for (int64_t i = 0; i < times; i++) {
      for (auto iter = this_b; iter != this_e; ++iter) {
        new_list.emplace_back(*iter);
      }
    }
    return new_list;
  }

  template <
      class U,
      typename = typename std::enable_if<std::is_same<
          value_type,
          typename std::remove_cv<typename std::remove_reference<U>::type>::type>::value>::type>
  MATXSCRIPT_INLINE_VISIBILITY FTList<value_type> __add__(const FTList<U>& o) const {
    return FTList<value_type>::Concat(*this, o);
  }

  MATXSCRIPT_INLINE_VISIBILITY RTValue __add__(const Any& o) const {
    if (o.IsObjectRef<FTList<value_type>>()) {
      return __add__(o.AsObjectRefNoCheck<FTList<value_type>>());
    } else {
      FTList<RTValue> ret;
      ret.extend((*this));
      ret.extend(o);
      return ret;
    }
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY void append(U&& item) const {
    GenericValueConverter<value_type> Conv;
    MutableImpl().emplace_back(Conv(std::forward<U>(item)));
  }

  MATXSCRIPT_INLINE_VISIBILITY void append(const char* const item) const {
    MutableImpl().emplace_back(String(item));
  }

  MATXSCRIPT_INLINE_VISIBILITY void append(const char32_t* const item) const {
    MutableImpl().emplace_back(Unicode(item));
  }

  MATXSCRIPT_INLINE_VISIBILITY void clear() const {
    MutableImpl().clear();
  }

  // TODO(maxiandi): implement list.copy

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY int64_t count(const U& item, std::true_type) const {
    auto& cons = MutableImpl();
    return std::count(cons.begin(), cons.end(), item);
  }
  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY int64_t count(const U& item, std::false_type) const {
    if ((!is_comparable_with_value<U>::value)) {
      return 0;
    }
    RTView item_view(item);
    SmartEqualTo func_equal;
    auto pred = [&item_view, &func_equal](reference val) { return func_equal(item_view, val); };
    auto& cons = MutableImpl();
    return std::count_if(cons.begin(), cons.end(), pred);
  }
  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY int64_t count(const U& item) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    return count(item, std::is_same<U_TYPE, value_type>());
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY void extend(const FTList<U>& items, std::true_type) const {
    auto& cons = MutableImpl();
    cons.reserve(cons.size() + items.size());
    for (const auto& item : items) {
      cons.emplace_back(item);
    }
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY void extend(const FTList<U>& items, std::false_type) const {
    auto& cons = MutableImpl();
    cons.reserve(cons.size() + items.size());
    GenericValueConverter<value_type> conv;
    for (const auto& item : items) {
      cons.emplace_back(conv(item));
    }
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY void extend(const FTList<U>& items) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    return extend(items, std::is_same<U_TYPE, value_type>{});
  }

  MATXSCRIPT_INLINE_VISIBILITY void extend(const Any& items) const {
    GenericValueConverter<value_type> Converter;
    auto iterable = Iterator::MakeGenericIterator(items);
    bool has_next = iterable.HasNext();
    while (has_next) {
      this->append(Converter(iterable.Next(&has_next)));
    }
  }

  // TODO(maxiandi): implement list.index
  // TODO(maxiandi): implement list.insert

  MATXSCRIPT_INLINE_VISIBILITY value_type pop(int64_t index = -1) const {
    value_type ret;
    auto& cons = MutableImpl();
    index = index_correction(index, cons.size());
    if (index >= 0 && index < cons.size()) {
      auto itr = cons.begin() + index;
      ret = std::move(*itr);
      cons.erase(itr);
    } else {
      if (cons.empty()) {
        MXTHROW << "[List.pop] IndexError: pop from empty list";
      } else {
        MXTHROW << "[List.pop] IndexError: pop index out of range";
      }
    }
    return ret;
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY void insert(int64_t index, U&& item) const {
    auto& cons = MutableImpl();
    index = index_correction(index, cons.size());
    if (index < 0) {
      index = 0;
    }
    if (index >= cons.size()) {
      append(item);
      return;
    }
    auto itr = cons.begin() + index;
    GenericValueConverter<value_type> Conv;
    cons.insert(itr, Conv(std::forward<U>(item)));
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY int64_t index(const U& x, int64_t start, int64_t end) const {
    int64_t len = size();
    start = slice_index_correction(start, len);
    end = slice_index_correction(end, len);

    if (start >= end) {
      THROW_PY_ValueError("ValueError: '", x, "' is not in list");
      return -1;
    }

    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    int64_t result = this->IndexImpl(x, start, end, std::is_same<U_TYPE, value_type>());

    if (result == end) {
      THROW_PY_ValueError("ValueError: '", x, "' is not in list");
      return -1;
    }
    return result;
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY int64_t index(const U& x, int64_t start = 0) const {
    return index(x, start, size());
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY void remove(const U& item, std::true_type) const {
    auto& cons = MutableImpl();
    auto it = std::find(cons.begin(), cons.end(), item);
    if (it != cons.end()) {
      cons.erase(it);
    } else {
      MXTHROW << "[list.remove] " << item << " not in list";
    }
  }

  template <typename U>
  MATXSCRIPT_INLINE_VISIBILITY void remove(const U& item, std::false_type) const {
    if ((!is_comparable_with_value<U>::value)) {
      MXTHROW << "[list.remove] " << item << " not in list";
    }
    RTView item_view(item);
    SmartEqualTo func_equal;
    auto pred = [&item_view, &func_equal](reference val) { return func_equal(item_view, val); };
    auto& cons = MutableImpl();
    auto it = std::find_if(cons.begin(), cons.end(), pred);
    if (it != cons.end()) {
      cons.erase(it);
    } else {
      MXTHROW << "[list.remove] " << item << " not in list";
    }
  }

  template <class U>
  MATXSCRIPT_INLINE_VISIBILITY void remove(const U& item) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    this->remove(item, std::is_same<U_TYPE, value_type>());
  }

  MATXSCRIPT_INLINE_VISIBILITY void reverse() const {
    auto& cons = MutableImpl();
    std::reverse(cons.begin(), cons.end());
  }

  MATXSCRIPT_INLINE_VISIBILITY void sort_by_std_less(bool reverse, std::true_type) const {
    if (reverse) {
      auto reverse_func = [](const T& lhs, const T& rhs) { return !std::less<T>{}(lhs, rhs); };
      sort::pdqsort(this->begin(), this->end(), reverse_func);
    } else {
      sort::pdqsort(this->begin(), this->end(), std::less<T>{});
    }
  }

  MATXSCRIPT_INLINE_VISIBILITY void sort_by_std_less(bool reverse, std::false_type) const {
    GenericValueConverter<RTView> conv;
    if (reverse) {
      auto reverse_func = [&conv](const T& lhs, const T& rhs) {
        return !Any::LessThan(conv(lhs), conv(rhs));
      };
      sort::pdqsort(this->begin(), this->end(), reverse_func);
    } else {
      auto func = [&conv](const T& lhs, const T& rhs) {
        return Any::LessThan(conv(lhs), conv(rhs));
      };
      sort::pdqsort(this->begin(), this->end(), func);
    }
  }

  MATXSCRIPT_INLINE_VISIBILITY void sort(bool reverse = false) const {
    constexpr bool std_less_support = std::is_arithmetic<value_type>::value ||
                                      std::is_same<value_type, String>::value ||
                                      std::is_same<value_type, Unicode>::value;
    return this->sort_by_std_less(
        reverse,
        typename std::conditional<std_less_support, std::true_type, std::false_type>::type{});
  }

  MATXSCRIPT_INLINE_VISIBILITY void sort(const Any& key, bool reverse = false) const {
    GenericValueConverter<RTView> conv;
    if (!key.IsObjectRef<UserDataRef>()) {
      THROW_PY_TypeError("'", key.type_name(), "' object is not callable");
    }
    auto key_func = key.AsObjectRefNoCheck<UserDataRef>();
    if (reverse) {
      auto reverse_func = [&key_func, &conv](const T& lhs, const T& rhs) {
        RTView lhs_view = conv(lhs);
        RTView rhs_view = conv(rhs);
        return Any::GreaterEqual(key_func.generic_call(PyArgs(&lhs_view, 1)),
                                 key_func.generic_call(PyArgs(&rhs_view, 1)));
      };
      sort::pdqsort(this->begin(), this->end(), reverse_func);
    } else {
      auto func = [&key_func, &conv](const T& lhs, const T& rhs) {
        RTView lhs_view = conv(lhs);
        RTView rhs_view = conv(rhs);
        return Any::LessThan(key_func.generic_call(PyArgs(&lhs_view, 1)),
                             key_func.generic_call(PyArgs(&rhs_view, 1)));
      };
      sort::pdqsort(this->begin(), this->end(), func);
    }
  }

 public:
  static MATXSCRIPT_INLINE_VISIBILITY FTList Concat(std::initializer_list<FTList> data) {
    FTList result;
    size_t cap = 0;
    for (auto& con : data) {
      cap += con.size();
    }
    result.reserve(cap);
    for (auto& con : data) {
      for (const auto& x : con) {
        result.push_back(x);
      }
    }
    return result;
  }

  static MATXSCRIPT_INLINE_VISIBILITY FTList Concat(const FTList& lhs, const FTList& rhs) {
    FTList result;
    result.reserve(lhs.size() + rhs.size());
    for (const auto& x : lhs) {
      result.push_back(x);
    }
    for (const auto& x : rhs) {
      result.push_back(x);
    }
    return result;
  }

 private:
  MATXSCRIPT_INLINE_VISIBILITY FTListNode<value_type>* MutableNode() const {
    return static_cast<FTListNode<value_type>*>(data_.get());
  }

  MATXSCRIPT_INLINE_VISIBILITY container_type& MutableImpl() const {
    return static_cast<FTListNode<value_type>*>(data_.get())->data_;
  }
};

namespace TypeIndex {
template <typename T>
struct type_index_traits<FTList<T>> {
  static constexpr int32_t value = kRuntimeFTList;
};
}  // namespace TypeIndex

// python methods
#define MATXSCRIPT_CHECK_FT_LIST_ARGS(FuncName, NumArgs)                               \
  MXCHECK(NumArgs == args.size()) << "[" << DemangleType(typeid(FTListNode<T>).name()) \
                                  << "::" << #FuncName << "] Expect " << NumArgs       \
                                  << " arguments but get " << args.size()

template <typename T>
const uint64_t FTListNode<T>::type_tag_ = std::hash<string_view>()(typeid(FTList<T>).name());
template <typename T>
const std::type_index FTListNode<T>::std_type_index_ = typeid(FTList<T>);
template <typename T>
const FTObjectBaseNode::FunctionTable FTListNode<T>::function_table_ = {
    {"__len__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__len__, 0);
       return self.AsObjectView<FTList<T>>().data().size();
     }},
    {"__iter__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__iter__, 0);
       return self.AsObjectView<FTList<T>>().data().iter();
     }},
    {"__contains__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__contains__, 1);
       return self.AsObjectView<FTList<T>>().data().contains(args[0].template As<RTValue>());
     }},
    {"__getitem__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__getitem__, 1);
       return T(self.AsObjectView<FTList<T>>().data().get_item(args[0].As<int64_t>()));
     }},
    {"__setitem__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__setitem__, 2);
       self.AsObjectView<FTList<T>>().data().set_item(args[0].As<int64_t>(), args[1]);
       return None;
     }},
    {"__getslice__",
     [](RTView self, PyArgs args) -> RTValue {
       static const String cls_name = DemangleType(typeid(FTListNode<T>).name());
       const auto& view = self.AsObjectView<FTList<T>>();
       switch (args.size()) {
         case 2: {
           return view.data().get_slice(args[0].As<int64_t>(), args[1].As<int64_t>());
         } break;
         case 3: {
           return view.data().get_slice(
               args[0].As<int64_t>(), args[1].As<int64_t>(), args[2].As<int64_t>());
         } break;
         default: {
           MXTHROW << "[" << DemangleType(typeid(FTListNode<T>).name())
                   << "::__getslice__] Expect 2 or 3 arguments but get " << args.size();
         } break;
       }
       return None;
     }},
    {"__setslice__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__setslice__, 3);
       self.AsObjectView<FTList<T>>().data().set_slice(
           args[0].As<int64_t>(), args[1].As<int64_t>(), args[2].template As<RTValue>());
       return None;
     }},
    {"__eq__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__eq__, 1);
       return self.AsObjectView<FTList<T>>().data().__eq__(args[0].template As<RTValue>());
     }},
    {"__mul__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__mul__, 1);
       return self.AsObjectView<FTList<T>>().data().__mul__(args[0].As<int64_t>());
     }},
    {"__add__",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(__add__, 1);
       return self.AsObjectView<FTList<T>>().data().__add__(args[0].template As<RTValue>());
     }},
    {"append",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(append, 1);
       self.AsObjectView<FTList<T>>().data().append(args[0].template As<RTValue>());
       return None;
     }},
    {"clear",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(clear, 0);
       self.AsObjectView<FTList<T>>().data().clear();
       return None;
     }},
    {"count",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(count, 1);
       return self.AsObjectView<FTList<T>>().data().count(args[0].template As<RTValue>());
     }},
    {"extend",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(extend, 1);
       self.AsObjectView<FTList<T>>().data().extend(args[0].template As<RTValue>());
       return None;
     }},
    {"pop",
     [](RTView self, PyArgs args) -> RTValue {
       static const String cls_name = DemangleType(typeid(FTListNode<T>).name());
       const auto& view = self.AsObjectView<FTList<T>>();
       switch (args.size()) {
         case 0: {
           return view.data().pop();
         } break;
         case 1: {
           return view.data().pop(args[0].As<int64_t>());
         } break;
         default: {
           MXTHROW << "[" << DemangleType(typeid(FTListNode<T>).name())
                   << "::pop] Expect 0 or 1 arguments but get " << args.size();
         } break;
       }
       return None;
     }},
    {"remove",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(remove, 1);
       self.AsObjectView<FTList<T>>().data().remove(args[0].template As<RTValue>());
       return None;
     }},
    {"insert",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(insert, 2);
       self.AsObjectView<FTList<T>>().data().insert(args[0].As<int64_t>(),
                                                    args[1].template As<RTValue>());
       return None;
     }},
    {"index",
     [](RTView self, PyArgs args) -> RTValue {
       switch (args.size()) {
         case 1: {
           int64_t start = 0;
           int64_t end = self.AsObjectView<FTList<T>>().data().size();
           return self.AsObjectView<FTList<T>>().data().index(
               args[0].template As<RTValue>(), start, end);
         } break;
         case 2: {
           int64_t end = self.AsObjectView<FTList<T>>().data().size();
           return self.AsObjectView<FTList<T>>().data().index(
               args[0].template As<RTValue>(), args[1].As<int64_t>(), end);
         } break;
         case 3: {
           return self.AsObjectView<FTList<T>>().data().index(
               args[0].template As<RTValue>(), args[1].As<int64_t>(), args[2].As<int64_t>());
         } break;
         default: {
           MXTHROW << "TypeError: index expected at most 3 arguments, got 4";
         } break;
       }
       return -1;
     }},
    {"reverse",
     [](RTView self, PyArgs args) -> RTValue {
       MATXSCRIPT_CHECK_FT_LIST_ARGS(reverse, 0);
       self.AsObjectView<FTList<T>>().data().reverse();
       return None;
     }},
    {"sort",
     [](RTView self, PyArgs args) -> RTValue {
       if (args.size() > 0) {
         RTValue* key_func = nullptr;
         bool reverse = false;
         list_details::trait_sort_kwargs(args, &key_func, &reverse);
         if (key_func) {
           self.AsObjectView<FTList<T>>().data().sort(*key_func, reverse);
         } else {
           self.AsObjectView<FTList<T>>().data().sort(reverse);
         }
       } else {
         self.AsObjectView<FTList<T>>().data().sort();
       }
       return None;
     }},
};

#undef MATXSCRIPT_CHECK_FT_LIST_ARGS

template <typename value_type>
static inline std::ostream& operator<<(std::ostream& os, FTList<value_type> const& n) {
  os << '[';
  for (auto it = n.begin(); it != n.end(); ++it) {
    if (it != n.begin()) {
      os << ", ";
    }
    if (std::is_same<value_type, String>::value) {
      os << "b'" << *it << "'";
    } else if (std::is_same<value_type, Unicode>::value) {
      os << "\'" << *it << "\'";
    } else {
      os << *it;
    }
  }
  os << ']';
  return os;
}

}  // namespace runtime
}  // namespace matxscript
