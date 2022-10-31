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

#include <cmath>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {
namespace floating_point {

template <class T>
MATXSCRIPT_ALWAYS_INLINE static
    typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    internal_almost_equals(T x, T y, int ulp) {
  // the machine epsilon has to be scaled to the magnitude of the values used
  // and multiplied by the desired precision in ULPs (units in the last place)
  return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
         // unless the result is subnormal
         || std::fabs(x - y) < std::numeric_limits<T>::min();
}

template <typename T1,
          typename T2,
          typename = typename std::enable_if<std::is_floating_point<T1>::value &&
                                             std::is_floating_point<T2>::value>::type>
MATXSCRIPT_ALWAYS_INLINE static bool AlmostEquals(const T1& lhs, const T2& rhs) {
  using Float1 = typename std::remove_cv<typename std::remove_reference<T1>::type>::type;
  using Float2 = typename std::remove_cv<typename std::remove_reference<T2>::type>::type;
  using Float = typename std::conditional<(sizeof(Float1) > sizeof(Float2)), Float1, Float2>::type;
  return internal_almost_equals<Float>(lhs, rhs, 4);
}

}  // namespace floating_point
}  // namespace runtime
}  // namespace matxscript
