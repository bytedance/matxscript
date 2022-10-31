// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Objects/floatobject.c
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
#include <matxscript/runtime/builtins_modules/_floatobject.h>

#include <matxscript/runtime/exceptions/exceptions.h>

#include <cassert>
#include <cmath>

namespace matxscript {
namespace runtime {
namespace py_builtins {

double float_div(double a, double b) {
  if (b == 0.0) {
    THROW_PY_ZeroDivisionError("float division by zero");
    return INFINITY;
  }
  return a / b;
}

double float_rem(double vx, double wx) {
  double mod;
  if (wx == 0.0) {
    THROW_PY_ZeroDivisionError("float modulo");
    return INFINITY;
  }
  mod = fmod(vx, wx);
  if (mod) {
    /* ensure the remainder has the same sign as the denominator */
    if ((wx < 0) != (mod < 0)) {
      mod += wx;
    }
  } else {
    /* the remainder is zero, and in the presence of signed zeroes
       fmod returns different results across platforms; ensure
       it has the same sign as the denominator. */
    mod = copysign(0.0, wx);
  }
  return mod;
}

std::pair<double, double> float_divmod(double vx, double wx) {
  double div, mod, floordiv;
  if (wx == 0.0) {
    THROW_PY_ZeroDivisionError("float divmod()");
    return {};
  }
  mod = fmod(vx, wx);
  /* fmod is typically exact, so vx-mod is *mathematically* an
     exact multiple of wx.  But this is fp arithmetic, and fp
     vx - mod is an approximation; the result is that div may
     not be an exact integral value after the division, although
     it will always be very close to one.
  */
  div = (vx - mod) / wx;
  if (mod) {
    /* ensure the remainder has the same sign as the denominator */
    if ((wx < 0) != (mod < 0)) {
      mod += wx;
      div -= 1.0;
    }
  } else {
    /* the remainder is zero, and in the presence of signed zeroes
       fmod returns different results across platforms; ensure
       it has the same sign as the denominator. */
    mod = copysign(0.0, wx);
  }
  /* snap quotient to nearest integral value */
  if (div) {
    floordiv = floor(div);
    if (div - floordiv > 0.5)
      floordiv += 1.0;
  } else {
    /* div is zero - get the same sign as the true quotient */
    floordiv = copysign(0.0, vx / wx); /* zero w/ sign of vx/wx */
  }
  return {floordiv, mod};
}

double float_floor_div(double v, double w) {
  auto t = float_divmod(v, w);
  return t.first;
}

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
