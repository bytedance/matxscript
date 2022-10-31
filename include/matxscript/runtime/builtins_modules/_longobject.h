// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Objects/longobject.c
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

#include <cstddef>
#include <cstdint>

#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {
namespace py_builtins {

int64_t fast_mod_i64_i64(int64_t a, int64_t b);
int64_t fast_mod_u64_i64(uint64_t a, int64_t b);
int64_t fast_mod_i64_u64(int64_t a, uint64_t b);
int64_t fast_mod_u64_u64(uint64_t a, uint64_t b);

template <typename LEFT,
          typename RIGHT,
          typename = typename std::enable_if<std::is_integral<LEFT>::value>::type,
          typename = typename std::enable_if<std::is_integral<RIGHT>::value>::type>
MATXSCRIPT_ALWAYS_INLINE int64_t fast_mod(LEFT a, RIGHT b) {
  using LEFT_TYPE = typename std::remove_cv<typename std::remove_reference<LEFT>::type>::type;
  using RIGHT_TYPE = typename std::remove_cv<typename std::remove_reference<RIGHT>::type>::type;
  if (std::is_unsigned<LEFT_TYPE>::value && std::is_unsigned<RIGHT_TYPE>::value) {
    return fast_mod_u64_u64(uint64_t(a), uint64_t(b));
  }
  if (std::is_unsigned<LEFT_TYPE>::value && std::is_signed<RIGHT_TYPE>::value) {
    return fast_mod_u64_i64(uint64_t(a), int64_t(b));
  }
  if (std::is_signed<LEFT_TYPE>::value && std::is_unsigned<RIGHT_TYPE>::value) {
    return fast_mod_i64_u64(int64_t(a), uint64_t(b));
  }
  if (std::is_signed<LEFT_TYPE>::value && std::is_signed<RIGHT_TYPE>::value) {
    return fast_mod_i64_i64(int64_t(a), int64_t(b));
  }
  // unreachable code
  return fast_mod_i64_i64(int64_t(a), int64_t(b));
}

// floordiv
int64_t fast_floor_div_i64_i64(int64_t a, int64_t b);
template <typename LEFT,
          typename RIGHT,
          typename = typename std::enable_if<std::is_integral<LEFT>::value>::type,
          typename = typename std::enable_if<std::is_integral<RIGHT>::value>::type>
MATXSCRIPT_ALWAYS_INLINE int64_t fast_floor_div(LEFT a, RIGHT b) {
  using LEFT_TYPE = typename std::remove_cv<typename std::remove_reference<LEFT>::type>::type;
  using RIGHT_TYPE = typename std::remove_cv<typename std::remove_reference<RIGHT>::type>::type;
  static_assert(
      (sizeof(LEFT_TYPE) < 64 || (sizeof(LEFT_TYPE) == 64 && std::is_signed<LEFT_TYPE>::value)) &&
          (sizeof(RIGHT_TYPE) < 64 ||
           (sizeof(RIGHT_TYPE) == 64 && std::is_signed<RIGHT_TYPE>::value)),
      "type is not supported");
  return fast_floor_div_i64_i64(int64_t(a), int64_t(b));
}

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
