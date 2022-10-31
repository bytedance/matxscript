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

#include <matxscript/runtime/object.h>

namespace matxscript {
namespace runtime {

namespace details {

template <typename T>
struct has_std_iterator {
  template <typename other>
  static char judge(typename other::iterator* x) {
    return 0;
  };
  template <typename other>
  static int judge(...) {
    return 1;
  };
  constexpr static bool value = sizeof(judge<T>(0)) == sizeof(char);
};

template <typename T>
struct has_value_type {
  template <typename other>
  static char judge(typename other::value_type* x) {
    return 0;
  };
  template <typename other>
  static int judge(...) {
    return 1;
  };
  constexpr static bool value = sizeof(judge<T>(0)) == sizeof(char);
};

template <bool HasIter, class T>
struct item_type_traits_helper;

template <class T>
struct item_type_traits_helper<false, T> {
  using type = RTValue;
};

template <class T>
struct item_type_traits_helper<true, T> {
  using type = typename std::iterator_traits<
      typename std::remove_cv<typename std::remove_reference<T>::type>::type::iterator>::value_type;
};

template <class T>
struct item_type_traits {
  using type = typename item_type_traits_helper<has_std_iterator<T>::value, T>::type;
};

template <>
struct item_type_traits<Unicode> {
  using type = Unicode;
};

template <>
struct item_type_traits<unicode_view> {
  using type = Unicode;
};

static constexpr int ITEM_TRANSFORM_TYPE_UNICODE = 0;
static constexpr int ITEM_TRANSFORM_TYPE_DEFAULT = 1;

template <class T>
struct _is_unicode_or_view_type : std::false_type {};

template <>
struct _is_unicode_or_view_type<Unicode> : std::true_type {};

template <>
struct _is_unicode_or_view_type<unicode_view> : std::true_type {};

template <class T>
using ITEM_TRANSFORM_TYPE =
    std::integral_constant<int,
                           _is_unicode_or_view_type<typename std::remove_cv<
                               typename std::remove_reference<T>::type>::type>::value
                               ? ITEM_TRANSFORM_TYPE_UNICODE
                               : ITEM_TRANSFORM_TYPE_DEFAULT>;

}  // namespace details
}  // namespace runtime
}  // namespace matxscript
