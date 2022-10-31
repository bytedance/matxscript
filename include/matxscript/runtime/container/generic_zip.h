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

#include <type_traits>

#include "_item_type_traits.h"
#include "_object_holder.h"
#include "iterator_utils.h"

namespace matxscript {
namespace runtime {

namespace details {

template <class... Iterables>
struct generic_zip_res {
  using type =
      decltype(std::make_tuple(std::declval<typename item_type_traits<Iterables>::type>()...));
};

template <class... Iterables>
struct generic_zip : public ObjectsHolder<true, Iterables...> {
  generic_zip() = default;

  template <class... Types>
  generic_zip(Types&&... iters) : ObjectsHolder<true, Iterables...>(std::forward<Types>(iters)...) {
  }
  bool HasNext() const {
    return HasNextImpl(iterator_int<sizeof...(Iterables) - 1>());
  }
  typename generic_zip_res<Iterables...>::type Next() {
    return NextImpl(std::make_index_sequence<sizeof...(Iterables)>());
  }
  typename generic_zip_res<Iterables...>::type Next(bool* has_next) {
    return NextImpl(has_next, std::make_index_sequence<sizeof...(Iterables)>());
  }

 private:
  template <size_t I>
  typename std::tuple_element<I, typename generic_zip_res<Iterables...>::type>::type INextImpl(
      bool* has_next) {
    bool has_next_tmp = *has_next;
    auto ret = std::get<I>(this->values).Next(&has_next_tmp);
    *has_next &= has_next_tmp;
    return ret;
  }
  template <size_t... I>
  typename generic_zip_res<Iterables...>::type NextImpl(bool* has_next, std::index_sequence<I...>) {
    return std::make_tuple(INextImpl<I>(has_next)...);
  }

  template <size_t... I>
  typename generic_zip_res<Iterables...>::type NextImpl(std::index_sequence<I...>) {
    return std::make_tuple(std::get<I>(this->values).Next()...);
  }

  template <size_t N>
  bool HasNextImpl(iterator_int<N>) const {
    return std::get<N>(this->values).HasNext() && HasNextImpl(iterator_int<N - 1>());
  }
  bool HasNextImpl(iterator_int<0>) const {
    return std::get<0>(this->values).HasNext();
  }
};

}  // namespace details

template <class... Iterables>
details::generic_zip<
    typename std::remove_cv<typename std::remove_reference<Iterables>::type>::type...>
generic_builtins_zip(Iterables&&... seq) {
  return {std::forward<Iterables>(seq)...};
}

}  // namespace runtime
}  // namespace matxscript
