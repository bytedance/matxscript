// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Include/pyport.h
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

#include "pyconfig.h" /* include for defines */
#include "pymacconfig.h"

/* A convenient way for code to know if clang's memory sanitizer is enabled. */
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#if !defined(_Py_MEMORY_SANITIZER)
#define _Py_MEMORY_SANITIZER
#endif
#endif
#endif

/* Py_ARITHMETIC_RIGHT_SHIFT
 * C doesn't define whether a right-shift of a signed integer sign-extends
 * or zero-fills.  Here a macro to force sign extension:
 * Py_ARITHMETIC_RIGHT_SHIFT(TYPE, I, J)
 *    Return I >> J, forcing sign extension.  Arithmetically, return the
 *    floor of I/2**J.
 * Requirements:
 *    I should have signed integer type.  In the terminology of C99, this can
 *    be either one of the five standard signed integer types (signed char,
 *    short, int, long, long long) or an extended signed integer type.
 *    J is an integer >= 0 and strictly less than the number of bits in the
 *    type of I (because C doesn't define what happens for J outside that
 *    range either).
 *    TYPE used to specify the type of I, but is now ignored.  It's been left
 *    in for backwards compatibility with versions <= 2.6 or 3.0.
 * Caution:
 *    I may be evaluated more than once.
 */
#ifdef SIGNED_RIGHT_SHIFT_ZERO_FILLS
#define Py_ARITHMETIC_RIGHT_SHIFT(TYPE, I, J) ((I) < 0 ? -1 - ((-1 - (I)) >> (J)) : (I) >> (J))
#else
#define Py_ARITHMETIC_RIGHT_SHIFT(TYPE, I, J) ((I) >> (J))
#endif

/*  The functions _Py_dg_strtod and _Py_dg_dtoa in Python/dtoa.c (which are
 *  required to support the short float repr introduced in Python 3.1) require
 *  that the floating-point unit that's being used for arithmetic operations
 *  on C doubles is set to use 53-bit precision.  It also requires that the
 *  FPU rounding mode is round-half-to-even, but that's less often an issue.
 *
 *  If your FPU isn't already set to 53-bit precision/round-half-to-even, and
 *  you want to make use of _Py_dg_strtod and _Py_dg_dtoa, then you should
 *
 *     #define HAVE_PY_SET_53BIT_PRECISION 1
 *
 *  and also give appropriate definitions for the following three macros:
 *
 *    _PY_SET_53BIT_PRECISION_START : store original FPU settings, and
 *        set FPU to 53-bit precision/round-half-to-even
 *    _PY_SET_53BIT_PRECISION_END : restore original FPU settings
 *    _PY_SET_53BIT_PRECISION_HEADER : any variable declarations needed to
 *        use the two macros above.
 *
 * The macros are designed to be used within a single C function: see
 * Python/pystrtod.c for an example of their use.
 */

/* get and set x87 control word for gcc/x86 */
#ifdef HAVE_GCC_ASM_FOR_X87
#define HAVE_PY_SET_53BIT_PRECISION 1
/* _Py_get/set_387controlword functions are defined in Python/pymath.c */
#define _Py_SET_53BIT_PRECISION_HEADER unsigned short old_387controlword, new_387controlword
#define _Py_SET_53BIT_PRECISION_START                             \
  do {                                                            \
    old_387controlword = _Py_get_387controlword();                \
    new_387controlword = (old_387controlword & ~0x0f00) | 0x0200; \
    if (new_387controlword != old_387controlword)                 \
      _Py_set_387controlword(new_387controlword);                 \
  } while (0)
#define _Py_SET_53BIT_PRECISION_END             \
  if (new_387controlword != old_387controlword) \
  _Py_set_387controlword(old_387controlword)
#endif

/* get and set x87 control word for VisualStudio/x86 */
#if defined(_MSC_VER) && !defined(_WIN64) && \
    !defined(_M_ARM) /* x87 not supported in 64-bit or ARM */
#define HAVE_PY_SET_53BIT_PRECISION 1
#define _Py_SET_53BIT_PRECISION_HEADER \
  unsigned int old_387controlword, new_387controlword, out_387controlword
/* We use the __control87_2 function to set only the x87 control word.
   The SSE control word is unaffected. */
#define _Py_SET_53BIT_PRECISION_START                                                       \
  do {                                                                                      \
    __control87_2(0, 0, &old_387controlword, NULL);                                         \
    new_387controlword = (old_387controlword & ~(_MCW_PC | _MCW_RC)) | (_PC_53 | _RC_NEAR); \
    if (new_387controlword != old_387controlword)                                           \
      __control87_2(new_387controlword, _MCW_PC | _MCW_RC, &out_387controlword, NULL);      \
  } while (0)
#define _Py_SET_53BIT_PRECISION_END                                                    \
  do {                                                                                 \
    if (new_387controlword != old_387controlword)                                      \
      __control87_2(old_387controlword, _MCW_PC | _MCW_RC, &out_387controlword, NULL); \
  } while (0)
#endif

#ifdef HAVE_GCC_ASM_FOR_MC68881
#define HAVE_PY_SET_53BIT_PRECISION 1
#define _Py_SET_53BIT_PRECISION_HEADER unsigned int old_fpcr, new_fpcr
#define _Py_SET_53BIT_PRECISION_START                          \
  do {                                                         \
    __asm__("fmove.l %%fpcr,%0" : "=g"(old_fpcr));             \
    /* Set double precision / round to nearest.  */            \
    new_fpcr = (old_fpcr & ~0xf0) | 0x80;                      \
    if (new_fpcr != old_fpcr)                                  \
      __asm__ volatile("fmove.l %0,%%fpcr" : : "g"(new_fpcr)); \
  } while (0)
#define _Py_SET_53BIT_PRECISION_END                            \
  do {                                                         \
    if (new_fpcr != old_fpcr)                                  \
      __asm__ volatile("fmove.l %0,%%fpcr" : : "g"(old_fpcr)); \
  } while (0)
#endif

/* default definitions are empty */
#ifndef HAVE_PY_SET_53BIT_PRECISION
#define _Py_SET_53BIT_PRECISION_HEADER
#define _Py_SET_53BIT_PRECISION_START
#define _Py_SET_53BIT_PRECISION_END
#endif

/* If we can't guarantee 53-bit precision, don't use the code
   in Python/dtoa.c, but fall back to standard code.  This
   means that repr of a float will be long (17 sig digits).

   Realistically, there are two things that could go wrong:

   (1) doubles aren't IEEE 754 doubles, or
   (2) we're on x86 with the rounding precision set to 64-bits
       (extended precision), and we don't know how to change
       the rounding precision.
 */

#if !defined(DOUBLE_IS_LITTLE_ENDIAN_IEEE754) && !defined(DOUBLE_IS_BIG_ENDIAN_IEEE754) && \
    !defined(DOUBLE_IS_ARM_MIXED_ENDIAN_IEEE754)
#define PY_NO_SHORT_FLOAT_REPR
#endif

/* double rounding is symptomatic of use of extended precision on x86.  If
   we're seeing double rounding, and we don't have any mechanism available for
   changing the FPU rounding precision, then don't use Python/dtoa.c. */
#if defined(X87_DOUBLE_ROUNDING) && !defined(HAVE_PY_SET_53BIT_PRECISION)
#define PY_NO_SHORT_FLOAT_REPR
#endif

/*
 * Convenient macros to deal with endianness of the platform. WORDS_BIGENDIAN is
 * detected by configure and defined in pyconfig.h. The code in pyconfig.h
 * also takes care of Apple's universal builds.
 */

#ifdef WORDS_BIGENDIAN
#define PY_BIG_ENDIAN 1
#define PY_LITTLE_ENDIAN 0
#else
#define PY_BIG_ENDIAN 0
#define PY_LITTLE_ENDIAN 1
#endif

/* Py_SAFE_DOWNCAST(VALUE, WIDE, NARROW)
 * Cast VALUE to type NARROW from type WIDE.  In Py_DEBUG mode, this
 * assert-fails if any information is lost.
 * Caution:
 *    VALUE may be evaluated more than once.
 */
#ifdef Py_DEBUG
#define Py_SAFE_DOWNCAST(VALUE, WIDE, NARROW) \
  (assert((WIDE)(NARROW)(VALUE) == (VALUE)), (NARROW)(VALUE))
#else
#define Py_SAFE_DOWNCAST(VALUE, WIDE, NARROW) (NARROW)(VALUE)
#endif
