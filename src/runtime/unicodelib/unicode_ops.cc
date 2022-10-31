// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
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
#include <matxscript/runtime/container/unicode.h>
#include <matxscript/runtime/unicodelib/unicode_ops.h>

namespace matxscript {
namespace runtime {

static inline Py_UCS4 handle_capital_sigma(const Py_UCS4* data, intptr_t length, intptr_t i) {
  intptr_t j;
  int final_sigma;
  Py_UCS4 c = 0; /* initialize to prevent gcc warning */
  /* U+03A3 is in the Final_Sigma context when, it is found like this:

   \p{cased}\p{case-ignorable}*U+03A3!(\p{case-ignorable}*\p{cased})

  where ! is a negation and \p{xxx} is a character with property xxx.
  */
  for (j = i - 1; j >= 0; j--) {
    c = data[j];
    if (!_PyUnicode_IsCaseIgnorable(c))
      break;
  }
  final_sigma = j >= 0 && _PyUnicode_IsCased(c);
  if (final_sigma) {
    for (j = i + 1; j < length; j++) {
      c = data[j];
      if (!_PyUnicode_IsCaseIgnorable(c))
        break;
    }
    final_sigma = j == length || !_PyUnicode_IsCased(c);
  }
  return (final_sigma) ? 0x3C2 : 0x3C3;
}

static inline int lower_ucs4(
    const Py_UCS4* data, intptr_t length, intptr_t i, Py_UCS4 c, Py_UCS4* mapped) {
  /* Obscure special case. */
  if (c == 0x3A3) {
    mapped[0] = handle_capital_sigma(data, length, i);
    return 1;
  }
  return _PyUnicode_ToLowerFull(c, mapped);
}

unicode_string py_unicode_do_upper(unicode_view input) {
  unicode_string result;
  intptr_t i;
  result.reserve(input.length() * 3);
  for (i = 0; i < input.length(); i++) {
    Py_UCS4 c = input[i], mapped[3];
    int n_res, j;
    n_res = _PyUnicode_ToUpperFull(c, mapped);
    for (j = 0; j < n_res; j++) {
      result.push_back(mapped[j]);
    }
  }
  return result;
}

unicode_string py_unicode_do_lower(unicode_view input) {
  unicode_string result;
  intptr_t i;
  result.reserve(input.length() * 3);
  for (i = 0; i < input.length(); i++) {
    Py_UCS4 c = input[i], mapped[3];
    int n_res, j;
    n_res = lower_ucs4(input.data(), input.length(), i, c, mapped);
    for (j = 0; j < n_res; j++) {
      result.push_back(mapped[j]);
    }
  }
  return result;
}

Unicode py_unicode_do_upper_optimize(unicode_view input) {
  Unicode result;
  intptr_t i;
  result.reserve(input.length() * 3);
  for (i = 0; i < input.length(); i++) {
    Py_UCS4 c = input[i], mapped[3];
    int n_res, j;
    n_res = _PyUnicode_ToUpperFull(c, mapped);
    for (j = 0; j < n_res; j++) {
      result.push_back(mapped[j]);
    }
  }
  return result;
}

Unicode py_unicode_do_lower_optimize(unicode_view input) {
  Unicode result;
  intptr_t i;
  result.reserve(input.length() * 3);
  for (i = 0; i < input.length(); i++) {
    Py_UCS4 c = input[i], mapped[3];
    int n_res, j;
    n_res = lower_ucs4(input.data(), input.length(), i, c, mapped);
    for (j = 0; j < n_res; j++) {
      result.push_back(mapped[j]);
    }
  }
  return result;
}

}  // namespace runtime
}  // namespace matxscript
