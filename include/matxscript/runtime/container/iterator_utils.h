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

namespace details {

struct iterator_npos {};

template <size_t>
struct iterator_int {};  // compile-time counter

/* Get the "minimum" of all iterators :
   - only random => random
   - at least one forward => forward
   */
template <typename... Iters>
struct iterator_min;

template <typename T>
struct iterator_min<T> {
  using type = typename std::iterator_traits<T>::iterator_category;
};

template <typename T, typename... Iters>
struct iterator_min<T, Iters...> {
  using type =
      typename std::conditional<std::is_same<typename std::iterator_traits<T>::iterator_category,
                                             std::forward_iterator_tag>::value,
                                std::forward_iterator_tag,
                                typename iterator_min<Iters...>::type>::type;
};

}  // namespace details
}  // namespace runtime
}  // namespace matxscript
