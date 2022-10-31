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

#include <tuple>
#include <type_traits>

#include "generic_funcs.h"

namespace matxscript {
namespace runtime {

namespace {
template <class...>
using to_void_t_impl = void;

template <typename T>
auto has_size_method_impl(int) -> decltype(std::declval<T>().size(), std::true_type{});
template <typename T>
auto has_size_method_impl(...) -> std::false_type;

template <typename, typename = void>
struct has_get_item_int_method_impl : std::false_type {};

template <typename T>
struct has_get_item_int_method_impl<T, to_void_t_impl<decltype(&T::get_item)>> : std::true_type {};

template <typename T>
using has_size_method = decltype(has_size_method_impl<T>(0));
}  // namespace

struct KernelBuiltinsLenDetails {
  template <typename... ARGS>
  MATXSCRIPT_ALWAYS_INLINE static int64_t run(const std::tuple<ARGS...>& container) {
    return std::tuple_size<std::tuple<ARGS...>>{};
  }
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static int64_t run(const std::pair<T1, T2>& container) {
    return 2;
  }
  template <typename T, typename = typename std::enable_if<has_size_method<T>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static int64_t run(const T& container) {
    return container.size();
  }
  MATXSCRIPT_ALWAYS_INLINE static int64_t run(const Any& container) {
    return kernel_object___len__(container);
  }
};

template <int64_t index, typename R>
struct KernelBuiltinsGetItemDetails {
  using return_type = typename std::remove_cv<typename std::remove_reference<R>::type>::type;
  template <typename... ARGS>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const std::tuple<ARGS...>& container) {
    return std::get<index>(container);
  }
  template <typename T,
            typename = typename std::enable_if<has_get_item_int_method_impl<T>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const T& container) {
    GenericValueConverter<return_type> converter;
    return converter(container.get_item(index));
  }
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const Any& container) {
    static_assert(std::is_same<return_type, RTValue>::value,
                  "Any.__getitem__ return_type must be Any");
    return kernel_object___getitem__(container, RTView(index));
  }
};

template <typename R>
struct KernelBuiltinsGetItemDetails<0, R> {
  using return_type = typename std::remove_cv<typename std::remove_reference<R>::type>::type;
  template <typename... ARGS>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const std::tuple<ARGS...>& container) {
    return std::get<0>(container);
  }
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const std::pair<T1, T2>& container) {
    return container.first;
  }
  template <typename T,
            typename = typename std::enable_if<has_get_item_int_method_impl<T>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const T& container) {
    GenericValueConverter<return_type> converter;
    return converter(container.get_item(0));
  }
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const Any& container) {
    static_assert(std::is_same<return_type, RTValue>::value,
                  "internal error: Any.__getitem__ return_type must be Any");
    return kernel_object___getitem__(container, RTView(0));
  }
};

template <typename R>
struct KernelBuiltinsGetItemDetails<1, R> {
  using return_type = typename std::remove_cv<typename std::remove_reference<R>::type>::type;
  template <typename... ARGS>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const std::tuple<ARGS...>& container) {
    return std::get<1>(container);
  }
  template <typename T1, typename T2>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const std::pair<T1, T2>& container) {
    return container.second;
  }
  template <typename T,
            typename = typename std::enable_if<has_get_item_int_method_impl<T>::value>::type>
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const T& container) {
    GenericValueConverter<return_type> converter;
    return converter(container.get_item(1));
  }
  MATXSCRIPT_ALWAYS_INLINE static return_type run(const Any& container) {
    static_assert(std::is_same<return_type, RTValue>::value,
                  "internal error: Any.__getitem__ return_type must be Any");
    return kernel_object___getitem__(container, RTView(1));
  }
};

template <int64_t index, typename R, typename T>
MATXSCRIPT_ALWAYS_INLINE R kernel_builtins_unpack(const T& container) {
  return KernelBuiltinsGetItemDetails<index, R>::run(container);
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_builtins_len(const T& container) {
  return KernelBuiltinsLenDetails::run(container);
}

}  // namespace runtime
}  // namespace matxscript
