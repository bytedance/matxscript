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

#include <matxscript/runtime/object.h>

namespace matxscript {
namespace runtime {

class Unicode;
class String;

namespace details {

template <typename... ARGS>
struct is_builtins_str_type;

template <typename T, typename... ARGS>
struct is_builtins_str_type<T, ARGS...> {
  using PURE_T = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  static constexpr bool value =
      (std::is_same<Unicode, PURE_T>::value || std::is_same<String, PURE_T>::value ||
       std::is_same<unicode_view, PURE_T>::value || std::is_same<string_view, PURE_T>::value) &&
      is_builtins_str_type<ARGS...>::value;
};

template <typename T>
struct is_builtins_str_type<T> {
  using PURE_T = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  static constexpr bool value =
      (std::is_same<Unicode, PURE_T>::value || std::is_same<String, PURE_T>::value ||
       std::is_same<unicode_view, PURE_T>::value || std::is_same<string_view, PURE_T>::value);
};

template <typename... ARGS>
struct is_object_type;

template <typename T, typename... ARGS>
struct is_object_type<T, ARGS...> {
  static constexpr bool value =
      std::is_base_of<
          ObjectRef,
          typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value &&
      is_object_type<ARGS...>::value;
};

template <typename T>
struct is_object_type<T> {
  static constexpr bool value = std::is_base_of<
      ObjectRef,
      typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value;
};

template <bool as_tuple, class T, class... Others>
struct ObjectsHolder;

template <class T, class... Others>
struct ObjectsHolder<true, T, Others...> {
  static_assert(is_object_type<T, Others...>::value || is_builtins_str_type<T, Others...>::value,
                "not a object reference or str");
  std::tuple<typename std::remove_cv<typename std::remove_reference<T>::type>::type,
             typename std::remove_cv<typename std::remove_reference<Others>::type>::type...>
      values;
  ObjectsHolder() = default;
  ObjectsHolder(T const& v, Others const&... o) : values(v, o...) {
  }
};

template <class T>
struct ObjectsHolder<false, T> {
  typename std::remove_cv<typename std::remove_reference<T>::type>::type values;
  ObjectsHolder() = default;
  ObjectsHolder(T const& v) : values(v) {
  }
};

template <class T>
struct ObjectsHolder<true, T> {
  std::tuple<typename std::remove_cv<typename std::remove_reference<T>::type>::type> values;
  ObjectsHolder() = default;
  ObjectsHolder(T const& v) : values(v) {
  }
};

}  // namespace details
}  // namespace runtime
}  // namespace matxscript
