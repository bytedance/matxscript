// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Include/pystrtod.h
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

#include <matxscript/runtime/container/string.h>

namespace matxscript {
namespace runtime {
namespace py_builtins {

extern double PyOS_string_to_double(const char* str, char** endptr);

extern String PyOS_double_to_string(
    double val, char format_code, int precision, int flags, int* type);

extern double _Py_parse_inf_or_nan(const char* p, char** endptr);

/* PyOS_double_to_string's "flags" parameter can be set to 0 or more of: */
constexpr int64_t Py_DTSF_SIGN = 0x01;      /* always add the sign */
constexpr int64_t Py_DTSF_ADD_DOT_0 = 0x02; /* if the result is an integer add ".0" */
constexpr int64_t Py_DTSF_ALT = 0x04;       /* "alternate" formatting. it's format_code \
                                             specific */

/* PyOS_double_to_string's "type", if non-NULL, will be set to one of: */
constexpr int64_t Py_DTST_FINITE = 0;
constexpr int64_t Py_DTST_INFINITE = 1;
constexpr int64_t Py_DTST_NAN = 2;

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
