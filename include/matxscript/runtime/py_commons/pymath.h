// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Include/pymath.h
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

#include <climits>
#include "pyconfig.h"

#ifdef HAVE_GCC_ASM_FOR_X87
namespace matxscript {
namespace runtime {
namespace py_builtins {

extern unsigned short _Py_get_387controlword(void);
extern void _Py_set_387controlword(unsigned short);

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
#endif

/* Return whether integral type *type* is signed or not. */
#define _Py_IntegralTypeSigned(type) ((type)(-1) < 0)
/* Return the maximum value of integral type *type*. */
#define _Py_IntegralTypeMax(type)                                                                 \
  ((_Py_IntegralTypeSigned(type)) ? (((((type)1 << (sizeof(type) * CHAR_BIT - 2)) - 1) << 1) + 1) \
                                  : ~(type)0)
/* Return the minimum value of integral type *type*. */
#define _Py_IntegralTypeMin(type) \
  ((_Py_IntegralTypeSigned(type)) ? -_Py_IntegralTypeMax(type) - 1 : 0)
/* Check whether *v* is in the range of integral type *type*. This is most
 * useful if *v* is floating-point, since demoting a floating-point *v* to an
 * integral type that cannot represent *v*'s integral part is undefined
 * behavior. */
#define _Py_InIntegralTypeRange(type, v) \
  (_Py_IntegralTypeMin(type) <= v && v <= _Py_IntegralTypeMax(type))
