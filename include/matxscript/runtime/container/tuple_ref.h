// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm
 * https://github.com/apache/tvm/blob/v0.7/include/tvm/runtime/container.h
 * with changes applied:
 * - rename namespace
 * - implement some tuple methods
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

#include <initializer_list>

#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

#include "iterator_utils.h"

namespace matxscript {
namespace runtime {

class TupleNode;

/*! \brief reference to algebraic data type objects. */
class Tuple : public ObjectRef {
 public:
  using ContainerType = TupleNode;
  static constexpr bool _type_is_nullable = false;  // disable nullptr for performance

 public:
  using value_type = RTValue;
  using pointer = RTValue*;
  using const_pointer = const RTValue*;
  using reference = RTValue&;
  using const_reference = const RTValue&;
  using const_iterator = const RTValue*;
  using iterator = const_iterator;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = const_reverse_iterator;
  using size_type = size_t;
  using difference_type = std::ptrdiff_t;
  using FuncGetNextItem = std::function<value_type()>;
  using FuncEqualToValue = std::function<bool(const value_type&)>;

 public:
  Tuple() = default;
  explicit Tuple(ObjectPtr<Object> n) noexcept : ObjectRef(std::move(n)) {
  }
  Tuple(const Tuple& other) noexcept = default;
  Tuple(Tuple&& other) noexcept = default;
  Tuple& operator=(const Tuple& other) noexcept = default;
  Tuple& operator=(Tuple&& other) noexcept = default;

  /*!
   * \brief construct an Tuple object reference.
   * \param tag The tag of the Tuple object.
   * \param begin The begin iterator to the start of the fields array.
   * \param end The end iterator to the end of the fields array.
   * \return The constructed Tuple object reference.
   */
  template <
      typename Iterator,
      typename = typename std::enable_if<
          (!std::is_same<Iterator, const value_type*>::value) &&
          (!std::is_same<Iterator, value_type*>::value) && (!std::is_same<Iterator, Any*>::value) &&
          (!std::is_same<typename std::iterator_traits<Iterator>::value_type, void>::value)>::type>
  Tuple(Iterator begin, Iterator end) {
    using IteratorValueType = typename std::iterator_traits<Iterator>::value_type;
    GenericValueConverter<value_type> converter;
    size_t num = std::distance(begin, end);
    FuncGetNextItem func = [&begin, &converter]() -> value_type { return converter(*(begin++)); };
    this->Init(func, num);
  }

  Tuple(const Any* begin, const Any* end);

  /*!
   * \brief construct an Tuple object reference.
   * \param tag The tag of the Tuple object.
   * \param init The initializer list of fields.
   * \return The constructed Tuple object reference.
   */
  Tuple(std::initializer_list<value_type> init);

  template <typename... Args>
  Tuple(const std::tuple<Args...>& tup) {
    AllocN(sizeof...(Args));
    EmplaceNthFromTuple(details::iterator_int<sizeof...(Args) - 1>(), tup);
  }

  template <typename FIRST, typename SECOND>
  Tuple(std::pair<FIRST, SECOND> tup) {
    AllocN(2);
    EmplaceUnsafe(std::move(tup.first));
    EmplaceUnsafe(std::move(tup.second));
  }

  template <typename... Args>
  static inline Tuple dynamic(Args&&... args) {
    Tuple tup = Empty(sizeof...(Args));
    tup.EmplaceNthFromDynamic(details::iterator_int<sizeof...(Args) - 1>(),
                              std::forward<Args>(args)...);
    return tup;
  }

 public:
  // iterators
  Iterator iter() const;
  iterator begin() const;
  iterator end() const;
  reverse_iterator rbegin() const;
  reverse_iterator rend() const;

  bool operator==(const Tuple& other) const;

  bool operator>(const Tuple& other) const;
  bool operator>=(const Tuple& other) const;

  bool operator<(const Tuple& other) const {
    return other.operator>(*this);
  }
  bool operator<=(const Tuple& other) const {
    return other.operator>=(*this);
  }

  /*!
   * \brief Access element at index.
   *
   * \param idx The array index
   * \return const value_type
   */
  value_type& operator[](size_t idx) const;

  value_type& get_item(int64_t idx) const;

  template <typename U>
  void set_item(int64_t idx, U&& item) const {
    using U_TYPE = typename std::remove_cv<typename std::remove_reference<U>::type>::type;
    if (std::is_same<U_TYPE, value_type>::value) {
      this->operator[](idx) = std::forward<U>(item);
    } else {
      this->operator[](idx) = GenericValueConverter<value_type>{}(std::forward<U>(item));
    }
  }

  /*!
   * \brief Return the number of fields.
   */
  int64_t size() const;

  Tuple repeat(int64_t times) const;

  Tuple get_slice(int64_t b, int64_t e, int64_t step = 1) const;

  template <typename U>
  bool contains(U&& item) const {
    FuncEqualToValue fn_equal_to = [&item](const value_type& val) -> bool {
      return SmartEqualTo()(val, item);
    };
    return find_match_fn(fn_equal_to);
  }

  bool find_match_fn(const FuncEqualToValue& fn) const;

  template <typename U>
  int64_t count(U&& item) const {
    FuncEqualToValue fn_equal_to = [&item](const value_type& val) -> bool {
      return SmartEqualTo()(val, item);
    };
    return count_match_fn(fn_equal_to);
  }

  int64_t count_match_fn(const FuncEqualToValue& fn) const;

  static Tuple Concat(Tuple lhs, Tuple rhs);

 private:
  static Tuple Empty(size_t capacity = 0);
  void Init(const FuncGetNextItem& func, size_t len);
  void AllocN(size_t len);
  Tuple& EmplaceUnsafe(value_type&& ele);
  Tuple& EmplaceUnsafe(const value_type& ele) {
    return EmplaceUnsafe(value_type(ele));
  }

  template <typename TupleType, size_t N>
  void EmplaceNthFromTuple(details::iterator_int<N>, const TupleType& tup) {
    EmplaceNthFromTuple(details::iterator_int<N - 1>(), tup);
    EmplaceUnsafe(std::get<N>(tup));
  }

  template <typename TupleType>
  void EmplaceNthFromTuple(details::iterator_int<0>, const TupleType& tup) {
    EmplaceUnsafe(std::get<0>(tup));
  }

  template <size_t N, typename Element, typename... Args>
  void EmplaceNthFromDynamic(details::iterator_int<N>, Element&& ele, Args&&... args) {
    using ELEMENT_TYPE =
        typename std::remove_cv<typename std::remove_reference<Element>::type>::type;
    GenericValueConverter<value_type> converter;
    EmplaceUnsafe(converter(std::forward<Element>(ele)));
    EmplaceNthFromDynamic(details::iterator_int<N - 1>{}, std::forward<Args>(args)...);
  }

  template <typename Element>
  void EmplaceNthFromDynamic(details::iterator_int<0>, Element&& ele) {
    GenericValueConverter<value_type> converter;
    EmplaceUnsafe(converter(std::forward<Element>(ele)));
  }
};

namespace TypeIndex {
template <>
struct type_index_traits<Tuple> {
  static constexpr int32_t value = kRuntimeTuple;
};
}  // namespace TypeIndex

// TODO RTValue as Tuple constructor ???
template <>
MATXSCRIPT_ALWAYS_INLINE Tuple Any::As<Tuple>() const {
  MATXSCRIPT_RUNTIME_VALUE_CHECK_TYPE_CODE(value_.code, TypeIndex::kRuntimeTuple);
  return Tuple(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}
template <>
MATXSCRIPT_ALWAYS_INLINE Tuple Any::AsNoCheck<Tuple>() const {
  return Tuple(GetObjectPtr<Object>(static_cast<Object*>(value_.data.v_handle)));
}

template <>
bool IsConvertible<Tuple>(const Object* node);

extern std::ostream& operator<<(std::ostream& os, Tuple const& n);

}  // namespace runtime
}  // namespace matxscript
