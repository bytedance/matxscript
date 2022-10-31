// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Python/pymath.c
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
#include <matxscript/runtime/py_commons/pymath.h>

#ifdef HAVE_GCC_ASM_FOR_X87
namespace matxscript {
namespace runtime {
namespace py_builtins {

/* inline assembly for getting and setting the 387 FPU control word on
   gcc/x86 */
#ifdef _Py_MEMORY_SANITIZER
__attribute__((no_sanitize_memory))
#endif
unsigned short
_Py_get_387controlword(void) {
  unsigned short cw;
  __asm__ __volatile__("fnstcw %0" : "=m"(cw));
  return cw;
}

void _Py_set_387controlword(unsigned short cw) {
  __asm__ __volatile__("fldcw %0" : : "m"(cw));
}

}  // namespace py_builtins
}  // namespace runtime
}  // namespace matxscript
#endif
