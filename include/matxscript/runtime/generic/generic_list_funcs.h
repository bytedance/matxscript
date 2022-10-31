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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/runtime_value.h>
#include <matxscript/runtime/type_helper_macros.h>

namespace matxscript {
namespace runtime {

namespace {

template <typename T>
struct remove_runtime_view {
  using type = T;
};

template <>
struct remove_runtime_view<string_view> {
  using type = String;
};

template <>
struct remove_runtime_view<unicode_view> {
  using type = Unicode;
};

template <>
struct remove_runtime_view<RTView> {
  using type = RTValue;
};

}  // namespace

/******************************************************************************
 * fused list ops
 *****************************************************************************/
template <typename... Args>
MATXSCRIPT_ALWAYS_INLINE auto kernel_list_fused_repeat_one(Args&&... args) {
  return List::repeat_one(std::forward<Args>(args)...);
}

MATXSCRIPT_ALWAYS_INLINE auto kernel_list_fused_repeat_many(
    const std::initializer_list<List::value_type>& values, int64_t times) {
  return List::repeat_many(values, times);
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE auto kernel_ft_list_fused_repeat_one(T&& value, int64_t times) {
  using T_TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  using ElementType = typename remove_runtime_view<T_TYPE>::type;
  return FTList<ElementType>::repeat_one(std::forward<T>(value), times);
}

template <typename T>
MATXSCRIPT_ALWAYS_INLINE auto kernel_ft_list_fused_repeat_many(
    const std::initializer_list<T>& values, int64_t times) {
  using T_TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  using ElementType = typename remove_runtime_view<T_TYPE>::type;
  return FTList<ElementType>::repeat_many(values, times);
}

}  // namespace runtime
}  // namespace matxscript
