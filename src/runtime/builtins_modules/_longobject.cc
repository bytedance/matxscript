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
#include <matxscript/runtime/builtins_modules/_longobject.h>

#include <matxscript/runtime/exceptions/exceptions.h>

#include <cassert>
#include <cmath>

namespace matxscript {
namespace runtime {
namespace py_builtins {

int64_t fast_mod_i64_i64(int64_t a, int64_t b) {
  int64_t mod;
  if (MATXSCRIPT_UNLIKELY(b == 0)) {
    THROW_PY_ZeroDivisionError("integer division or modulo by zero");
    return NAN;
  }
  if (MATXSCRIPT_UNLIKELY(a == 0)) {
    return 0;
  }

  /* negate: can't write this as abs_ival = -ival since that
   * invokes undefined behaviour when ival is LONG_MIN */
  uint64_t a_abs = a < 0 ? uint64_t(0) - uint64_t(a) : a;
  uint64_t b_abs = b < 0 ? uint64_t(0) - uint64_t(b) : b;

  if ((a ^ b) >= 0) {
    /* 'a' and 'b' have the same sign. */
    mod = a_abs % b_abs;
  } else {
    /* Either 'a' or 'b' is negative. */
    mod = b_abs - 1 - (a_abs - 1) % b_abs;
  }

  return b > 0 ? mod : -mod;
}

int64_t fast_mod_u64_i64(uint64_t a, int64_t b) {
  int64_t mod;

  if (MATXSCRIPT_UNLIKELY(b == 0)) {
    THROW_PY_ZeroDivisionError("integer division or modulo by zero");
    return NAN;
  }
  if (MATXSCRIPT_UNLIKELY(a == 0)) {
    return 0;
  }

  /* negate: can't write this as abs_ival = -ival since that
   * invokes undefined behaviour when ival is LONG_MIN */
  uint64_t b_abs = b < 0 ? uint64_t(0) - uint64_t(b) : b;

  if (b >= 0) {
    /* 'a' and 'b' have the same sign. */
    mod = a % b_abs;
  } else {
    /* Either 'a' or 'b' is negative. */
    mod = b_abs - 1 - (a - 1) % b_abs;
  }

  return b > 0 ? mod : -mod;
}

int64_t fast_mod_i64_u64(int64_t a, uint64_t b) {
  int64_t mod;
  if (MATXSCRIPT_UNLIKELY(b == 0)) {
    THROW_PY_ZeroDivisionError("integer division or modulo by zero");
    return NAN;
  }
  if (MATXSCRIPT_UNLIKELY(a == 0)) {
    return 0;
  }

  /* negate: can't write this as abs_ival = -ival since that
   * invokes undefined behaviour when ival is LONG_MIN */
  uint64_t a_abs = a < 0 ? uint64_t(0) - uint64_t(a) : a;

  if (a >= 0) {
    /* 'a' and 'b' have the same sign. */
    mod = a_abs % b;
  } else {
    /* Either 'a' or 'b' is negative. */
    mod = b - 1 - (a_abs - 1) % b;
  }

  return mod;
}

int64_t fast_mod_u64_u64(uint64_t a, uint64_t b) {
  if (MATXSCRIPT_UNLIKELY(b == 0)) {
    THROW_PY_ZeroDivisionError("integer division or modulo by zero");
    return NAN;
  }
  if (MATXSCRIPT_UNLIKELY(a == 0)) {
    return 0;
  }
  return a % b;
}

// floordiv
int64_t fast_floor_div_i64_i64(int64_t a, int64_t b) {
  int64_t div;

  if (MATXSCRIPT_UNLIKELY(b == 0)) {
    THROW_PY_ZeroDivisionError("integer division or modulo by zero");
    return NAN;
  }
  if (MATXSCRIPT_UNLIKELY(a == 0)) {
    return 0;
  }

  /* negate: can't write this as abs_ival = -ival since that
   * invokes undefined behaviour when ival is LONG_MIN */
  uint64_t a_abs = a < 0 ? uint64_t(0) - uint64_t(a) : a;
  uint64_t b_abs = b < 0 ? uint64_t(0) - uint64_t(b) : b;

  if ((a ^ b) >= 0) {
    /* 'a' and 'b' have the same sign. */
    div = a_abs / b_abs;
  } else {
    /* Either 'a' or 'b' is negative. */
    div = -1 - (a_abs - 1) / b_abs;
  }

  return div;
}

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
