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
#include <utility>

#include <matxscript/runtime/variadic_traits.h>
#include "_object_holder.h"
#include "_variadic_fwd.h"
#include "iterator_utils.h"

namespace matxscript {
namespace runtime {

namespace details {

template <class FLambda, class... Iterables>
struct map_res {
  using type = typename variadic_details::function_signature<FLambda>::return_type;
};

template <typename FLambda, typename... Iterables>
struct map_iterator : std::iterator<typename iterator_min<typename Iterables::iterator...>::type,
                                    typename map_res<FLambda, Iterables...>::type> {
  using iterable_types = std::tuple<typename std::is_same<
      typename std::remove_cv<typename std::remove_reference<Iterables>::type>::type,
      Unicode>...>;
  std::tuple<typename Iterables::iterator...> it;
  FLambda func_;

  map_iterator() = default;
  template <size_t... I>
  map_iterator(FLambda const& func, std::tuple<Iterables...>& iters, std::index_sequence<I...>)
      : it(std::get<I>(iters).begin()...), func_(func) {
  }

  template <size_t... I>
  map_iterator(iterator_npos,
               FLambda const& func,
               std::tuple<Iterables...>& iters,
               std::index_sequence<I...>)
      : it(std::get<I>(iters).end()...), func_(func) {
  }

  typename map_res<FLambda, Iterables...>::type operator*() const {
    return get_value(std::make_index_sequence<sizeof...(Iterables)>{});
  }

  map_iterator& operator++() {
    next(std::make_index_sequence<sizeof...(Iterables)>{});
    return *this;
  }

  const map_iterator operator++(int) {
    map_iterator<FLambda, Iterables...> tmp(*this);
    ++(*this);
    return tmp;
  }

  map_iterator& operator+=(int64_t i) {
    advance(i, iterator_int<sizeof...(Iterables) - 1>());
    return *this;
  }
  map_iterator operator+(int64_t i) const {
    map_iterator<FLambda, Iterables...> other(*this);
    other += i;
    return other;
  }
  bool operator==(map_iterator const& other) const {
    return equal(other, iterator_int<sizeof...(Iterables) - 1>());
  }
  bool operator!=(map_iterator const& other) const {
    return !(*this == other);
  }
  bool operator<(map_iterator const& other) const {
    return !(*this == other);
  }
  int64_t operator-(map_iterator const& other) const {
    return min_len(other, iterator_int<sizeof...(Iterables) - 1>());
  }

 private:
  template <size_t N>
  int64_t min_len(map_iterator<FLambda, Iterables...> const& other, iterator_int<N>) const {
    return std::min((int64_t)(std::get<N>(it) - std::get<N>(other.it)),
                    min_len(other, iterator_int<N - 1>()));
  }
  int64_t min_len(map_iterator<FLambda, Iterables...> const& other, iterator_int<0>) const {
    return std::get<0>(it) - std::get<0>(other.it);
  }

  template <size_t N>
  bool equal(map_iterator const& other, iterator_int<N>) const {
    return std::get<N>(other.it) == std::get<N>(it) || equal(other, iterator_int<N - 1>());
  }
  bool equal(map_iterator const& other, iterator_int<0>) const {
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
  typename map_res<FLambda, Iterables...>::type get_value(std::index_sequence<I...>) const {
    return func_(transform_value(*std::get<I>(it),
                                 typename std::tuple_element<I, iterable_types>::type())...);
  }

  template <typename T_ELE>
  Unicode transform_value(T_ELE&& ele, std::true_type) const {
    return Unicode(1, ele);
  };
  template <typename T_ELE>
  T_ELE transform_value(T_ELE&& ele, std::false_type) const {
    return ele;
  };
};

template <typename FLambda, typename... Iterables>
struct map : ObjectsHolder<true, Iterables...>, map_iterator<FLambda, Iterables...> {
  using iterator = map_iterator<FLambda, Iterables...>;
  using value_type = typename iterator::value_type;

  iterator end_iter;

  map() = default;
  // Use an extra template to enable forwarding
  template <class... Types>
  map(FLambda const& func, Types&&... iters)
      : ObjectsHolder<true, Iterables...>(std::forward<Types>(iters)...),
        map_iterator<FLambda, Iterables...>(
            func, this->values, std::make_index_sequence<sizeof...(Iterables)>{}),
        end_iter(
            iterator_npos(), func, this->values, std::make_index_sequence<sizeof...(Iterables)>{}) {
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

template <typename FLambda, typename... Iterables>
auto builtins_map(FLambda&& func, Iterables&&... iters) -> details::map<
    typename std::remove_cv<typename std::remove_reference<FLambda>::type>::type,
    typename std::remove_cv<typename std::remove_reference<Iterables>::type>::type...> {
  return {std::forward<FLambda>(func), std::forward<Iterables>(iters)...};
}

}  // namespace runtime
}  // namespace matxscript
