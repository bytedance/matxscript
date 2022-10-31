// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from CPython.
 * https://github.com/python/cpython/blob/3.8/Include/pyctype.h
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

#include <matxscript/runtime/runtime_port.h>

namespace matxscript {
namespace runtime {

/* Argument must be a char or an int in [-128, 127] or [0, 255]. */
#define UCHAR_MASK(c) ((unsigned char)((c)&0xff))

#define UCHAR_CTF_LOWER 0x01
#define UCHAR_CTF_UPPER 0x02
#define UCHAR_CTF_ALPHA (UCHAR_CTF_LOWER | UCHAR_CTF_UPPER)
#define UCHAR_CTF_DIGIT 0x04
#define UCHAR_CTF_ALNUM (UCHAR_CTF_ALPHA | UCHAR_CTF_DIGIT)
#define UCHAR_CTF_SPACE 0x08
#define UCHAR_CTF_XDIGIT 0x10

extern MATX_DLL const unsigned int __uchar_type_table[256];

/* Unlike their C counterparts, the following macros are not meant to
 * handle an int with any of the values [EOF, 0-UCHAR_MAX]. The argument
 * must be a signed/unsigned char. */
#define UCHAR_ISLOWER(c) (__uchar_type_table[UCHAR_MASK(c)] & UCHAR_CTF_LOWER)
#define UCHAR_ISUPPER(c) (__uchar_type_table[UCHAR_MASK(c)] & UCHAR_CTF_UPPER)
#define UCHAR_ISALPHA(c) (__uchar_type_table[UCHAR_MASK(c)] & UCHAR_CTF_ALPHA)
#define UCHAR_ISDIGIT(c) (__uchar_type_table[UCHAR_MASK(c)] & UCHAR_CTF_DIGIT)
#define UCHAR_ISXDIGIT(c) (__uchar_type_table[UCHAR_MASK(c)] & UCHAR_CTF_XDIGIT)
#define UCHAR_ISALNUM(c) (__uchar_type_table[UCHAR_MASK(c)] & UCHAR_CTF_ALNUM)
#define UCHAR_ISSPACE(c) (__uchar_type_table[UCHAR_MASK(c)] & UCHAR_CTF_SPACE)

extern MATX_DLL const unsigned char __uchar_tolower_index[256];
extern MATX_DLL const unsigned char __uchar_toupper_index[256];

#define UCHAR_TOLOWER(c) (__uchar_tolower_index[UCHAR_MASK(c)])
#define UCHAR_TOUPPER(c) (__uchar_toupper_index[UCHAR_MASK(c)])

}  // namespace runtime
}  // namespace matxscript
