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
#pragma once

#include "py_unicodeobject.h"

#include <algorithm>
#include <string>

#include <matxscript/runtime/container/unicode_view.h>

namespace matxscript {
namespace runtime {

class Unicode;

extern unicode_string py_unicode_do_upper(unicode_view input);
extern unicode_string py_unicode_do_lower(unicode_view input);
extern Unicode py_unicode_do_upper_optimize(unicode_view input);
extern Unicode py_unicode_do_lower_optimize(unicode_view input);

inline bool py_unicode_isdigit(unicode_view input) noexcept {
  return std::all_of(input.begin(), input.end(), _PyUnicode_IsDigit);
}

inline bool py_unicode_isalpha(unicode_view input) noexcept {
  return std::all_of(input.begin(), input.end(), _PyUnicode_IsAlpha);
}

inline bool py_unicode_isspace(Py_UCS4 c) noexcept {
  return _PyUnicode_IsWhitespace(c);
}

}  // namespace runtime
}  // namespace matxscript
