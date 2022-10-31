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

#include <iterator>
#include <tuple>

#include <matxscript/runtime/container/unicode.h>
#include "_item_type_traits.h"
#include "_object_holder.h"
#include "_variadic_fwd.h"
#include "iterator_utils.h"

namespace matxscript {
namespace runtime {

namespace details {

template <class... Iterables>
struct zip_res {
  using type = std::tuple<typename item_type_traits<
      typename std::remove_cv<typename std::remove_reference<Iterables>::type>::type>::type...>;
};

template <typename... Iterables>
struct zip_iterator
    : public std::iterator<typename iterator_min<typename Iterables::iterator...>::type,
                           typename zip_res<Iterables...>::type> {
  std::tuple<typename Iterables::iterator...> it;
  using iterable_types = std::tuple<ITEM_TRANSFORM_TYPE<Iterables>...>;

  zip_iterator() = default;

  template <typename TupleType, size_t... I>
  zip_iterator(TupleType&& iters, std::index_sequence<I...>) : it(std::get<I>(iters).begin()...) {
  }

  template <typename TupleType, size_t... I>
  zip_iterator(iterator_npos, TupleType&& iters, std::index_sequence<I...>)
      : it(std::get<I>(iters).end()...) {
  }

  typename zip_res<Iterables...>::type operator*() const {
    return get_value(std::make_index_sequence<sizeof...(Iterables)>{});
  }

  zip_iterator& operator++() {
    next(std::make_index_sequence<sizeof...(Iterables)>{});
    return *this;
  }

  const zip_iterator operator++(int) {
    zip_iterator<Iterables...> tmp(*this);
    ++(*this);
    return tmp;
  }

  zip_iterator& operator+=(int64_t i) {
    advance(i, iterator_int<sizeof...(Iterables) - 1>());
    return *this;
  }

  zip_iterator operator+(int64_t i) const {
    zip_iterator<Iterables...> other(*this);
    other += i;
    return other;
  }

  bool operator==(zip_iterator const& other) const {
    return equal(other, iterator_int<sizeof...(Iterables) - 1>());
  }

  bool operator!=(zip_iterator const& other) const {
    return !(*this == other);
  }

  bool operator<(zip_iterator const& other) const {
    return !(*this == other);
  }

  int64_t operator-(zip_iterator const& other) const {
    return min_len(other, iterator_int<sizeof...(Iterables) - 1>());
  }

 private:
  template <size_t N>
  int64_t min_len(zip_iterator<Iterables...> const& other, iterator_int<N>) const {
    return std::min((int64_t)(std::get<N>(it) - std::get<N>(other.it)),
                    min_len(other, iterator_int<N - 1>()));
  }
  int64_t min_len(zip_iterator<Iterables...> const& other, iterator_int<0>) const {
    return std::get<0>(it) - std::get<0>(other.it);
  }

  template <size_t N>
  bool equal(zip_iterator<Iterables...> const& other, iterator_int<N>) const {
    return std::get<N>(other.it) == std::get<N>(it) || equal(other, iterator_int<N - 1>());
  }
  bool equal(zip_iterator<Iterables...> const& other, iterator_int<0>) const {
    return std::get<0>(other.it) == std::get<0>(it);
  }

  template <size_t I>
  void advance(int64_t i, iterator_int<I>) {
    std::get<I>(it) += i;
    advance(i, iterator_int<I - 1>());
  }
  void advance(int64_t i, iterator_int<0>) {
    std::get<0>(it) += i;
  }

  template <size_t... I>
  void next(std::index_sequence<I...>) {
    variadic_details::fwd(++std::get<I>(it)...);
  }

  template <size_t... I>
  typename zip_res<Iterables...>::type get_value(std::index_sequence<I...>) const {
    return std::make_tuple(transform_value(
        *std::get<I>(it), typename std::tuple_element<I, iterable_types>::type())...);
  }

  template <typename T_ELE>
  Unicode transform_value(T_ELE&& ele,
                          std::integral_constant<int, ITEM_TRANSFORM_TYPE_UNICODE>) const {
    return Unicode(1, ele);
  };
  template <typename T_ELE>
  T_ELE transform_value(T_ELE&& ele,
                        std::integral_constant<int, ITEM_TRANSFORM_TYPE_DEFAULT>) const {
    return ele;
  };
};

template <typename... Iterables>
struct zip : ObjectsHolder<true, Iterables...>, zip_iterator<Iterables...> {
  using iterator = zip_iterator<Iterables...>;
  using value_type = typename iterator::value_type;

  iterator end_iter;

  zip() = default;
  // Use an extra template to enable forwarding
  template <class... Types>
  zip(Types&&... iters)
      : ObjectsHolder<true, Iterables...>(std::forward<Types>(iters)...),
        zip_iterator<Iterables...>(this->values, std::make_index_sequence<sizeof...(Iterables)>{}),
        end_iter(iterator_npos(), this->values, std::make_index_sequence<sizeof...(Iterables)>{}) {
  }

  iterator& begin() {
    return *this;
  }
  iterator const& begin() const {
    return *this;
  }

  iterator const& end() const {
    return end_iter;
  }
};

}  // namespace details

template <class... Iterables>
typename details::zip<
    typename std::remove_cv<typename std::remove_reference<Iterables>::type>::type...>
builtins_zip(Iterables&&... seq) {
  using ReturnType = typename details::zip<
      typename std::remove_cv<typename std::remove_reference<Iterables>::type>::type...>;
  return ReturnType(std::forward<Iterables>(seq)...);
}

}  // namespace runtime
}  // namespace matxscript
