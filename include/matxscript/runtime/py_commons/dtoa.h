// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Include/dtoa.h
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

namespace matxscript {
namespace runtime {
namespace py_builtins {

#ifndef PY_NO_SHORT_FLOAT_REPR

extern double _Py_dg_strtod(const char* str, char** ptr);
extern char* _Py_dg_dtoa(double d, int mode, int ndigits, int* decpt, int* sign, char** rve);
extern void _Py_dg_freedtoa(char* s);
extern double _Py_dg_stdnan(int sign);
extern double _Py_dg_infinity(int sign);

#endif

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
