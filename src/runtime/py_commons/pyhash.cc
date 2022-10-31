// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Include/pyhash.h
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
#include <matxscript/runtime/py_commons/pyhash.h>

#include <cmath>

namespace matxscript {
namespace runtime {
namespace py_builtins {

/* Prime multiplier used in string and various other hashes. */
constexpr size_t _PyHASH_MULTIPLIER = 1000003UL; /* 0xf4243 */

/* Parameters used for the numeric hash implementation.  See notes for
   _Py_HashDouble in Python/pyhash.c.  Numeric hashes are based on
   reduction modulo the prime 2**_PyHASH_BITS - 1. */

constexpr size_t _PyHASH_BITS = sizeof(void*) >= 8 ? 61 : 31;
constexpr size_t _PyHASH_MODULUS = (((size_t)1 << _PyHASH_BITS) - 1);
constexpr size_t _PyHASH_INF = 314159;
constexpr size_t _PyHASH_NAN = 0;
constexpr size_t _PyHASH_IMAG = _PyHASH_MULTIPLIER;

size_t _Py_HashDouble(double v) noexcept {
  int e, sign;
  double m;
  size_t x, y;

  if (!std::isfinite(v)) {
    if (std::isinf(v))
      return v > 0 ? _PyHASH_INF : -_PyHASH_INF;
    else
      return _PyHASH_NAN;
  }

  m = frexp(v, &e);

  sign = 1;
  if (m < 0) {
    sign = -1;
    m = -m;
  }

  /* process 28 bits at a time;  this should work well both for binary
     and hexadecimal floating point. */
  x = 0;
  while (m) {
    x = ((x << 28) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - 28);
    m *= 268435456.0; /* 2**28 */
    e -= 28;
    y = (size_t)m; /* pull out integer part */
    m -= y;
    x += y;
    if (x >= _PyHASH_MODULUS)
      x -= _PyHASH_MODULUS;
  }

  /* adjust for the exponent;  first reduce it modulo _PyHASH_BITS */
  e = e >= 0 ? e % _PyHASH_BITS : _PyHASH_BITS - 1 - ((-1 - e) % _PyHASH_BITS);
  x = ((x << e) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - e);

  x = x * sign;
  if (x == (size_t)-1)
    x = (size_t)-2;
  return x;
}

size_t _Py_HashPointer(void* p) noexcept {
  size_t y = (size_t)p;
  /* bottom 3 or 4 bits are likely to be 0; rotate y by 4 to avoid
     excessive hash collisions for dicts and sets */
  y = (y >> 4) | (y << (8 * sizeof(void*) - 4));
  if (y == (size_t)-1)
    y = (size_t)-2;
  return y;
}

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
