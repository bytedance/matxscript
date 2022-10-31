// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from python-farmhash
 *
 * Copyright (c) 2014, Veelion Chong

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#pragma once

#include <matxscript/runtime/builtins_modules/_longobject.h>
#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/container/tuple_ref.h>
#include <matxscript/runtime/container/unicode_view.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * farmhash origin function
 *****************************************************************************/

uint32_t kernel_farmhash_hash32(const string_view& s);
uint32_t kernel_farmhash_hash32(const unicode_view& s);
MATXSCRIPT_ALWAYS_INLINE uint32_t kernel_farmhash_hash32(const Any& s) {
  return s.Is<string_view>() ? kernel_farmhash_hash32(s.AsNoCheck<string_view>())
                             : kernel_farmhash_hash32(s.As<unicode_view>());
}

uint32_t kernel_farmhash_hash32withseed(const string_view& s, uint32_t seed);
uint32_t kernel_farmhash_hash32withseed(const unicode_view& s, uint32_t seed);
MATXSCRIPT_ALWAYS_INLINE uint32_t kernel_farmhash_hash32withseed(const Any& s, uint32_t seed) {
  return s.Is<string_view>() ? kernel_farmhash_hash32withseed(s.AsNoCheck<string_view>(), seed)
                             : kernel_farmhash_hash32withseed(s.As<unicode_view>(), seed);
}

uint64_t kernel_farmhash_hash64(const string_view& s);
uint64_t kernel_farmhash_hash64(const unicode_view& s);
MATXSCRIPT_ALWAYS_INLINE uint64_t kernel_farmhash_hash64(const Any& s) {
  return s.Is<string_view>() ? kernel_farmhash_hash64(s.AsNoCheck<string_view>())
                             : kernel_farmhash_hash64(s.As<unicode_view>());
}

uint64_t kernel_farmhash_hash64withseed(const string_view& s, uint64_t seed);
uint64_t kernel_farmhash_hash64withseed(const unicode_view& s, uint64_t seed);
MATXSCRIPT_ALWAYS_INLINE uint64_t kernel_farmhash_hash64withseed(const Any& s, uint64_t seed) {
  return s.Is<string_view>() ? kernel_farmhash_hash64withseed(s.AsNoCheck<string_view>(), seed)
                             : kernel_farmhash_hash64withseed(s.As<unicode_view>(), seed);
}

Tuple kernel_farmhash_hash128(const string_view& s);
Tuple kernel_farmhash_hash128(const unicode_view& s);
MATXSCRIPT_ALWAYS_INLINE Tuple kernel_farmhash_hash128(const Any& s) {
  return s.Is<string_view>() ? kernel_farmhash_hash128(s.AsNoCheck<string_view>())
                             : kernel_farmhash_hash128(s.As<unicode_view>());
}

Tuple kernel_farmhash_hash128withseed(const string_view& s,
                                      uint64_t seedlow64,
                                      uint64_t seedhigh64);
Tuple kernel_farmhash_hash128withseed(const unicode_view& s,
                                      uint64_t seedlow64,
                                      uint64_t seedhigh64);
MATXSCRIPT_ALWAYS_INLINE Tuple kernel_farmhash_hash128withseed(const Any& s,
                                                               uint64_t seedlow64,
                                                               uint64_t seedhigh64) {
  return s.Is<string_view>()
             ? kernel_farmhash_hash128withseed(s.AsNoCheck<string_view>(), seedlow64, seedhigh64)
             : kernel_farmhash_hash128withseed(s.As<unicode_view>(), seedlow64, seedhigh64);
}

uint32_t kernel_farmhash_fingerprint32(const string_view& s);
uint32_t kernel_farmhash_fingerprint32(const unicode_view& s);
MATXSCRIPT_ALWAYS_INLINE uint32_t kernel_farmhash_fingerprint32(const Any& s) {
  return s.Is<string_view>() ? kernel_farmhash_fingerprint32(s.AsNoCheck<string_view>())
                             : kernel_farmhash_fingerprint32(s.As<unicode_view>());
}

uint64_t kernel_farmhash_fingerprint64(const string_view& s);
uint64_t kernel_farmhash_fingerprint64(const unicode_view& s);
MATXSCRIPT_ALWAYS_INLINE uint64_t kernel_farmhash_fingerprint64(const Any& s) {
  return s.Is<string_view>() ? kernel_farmhash_fingerprint64(s.AsNoCheck<string_view>())
                             : kernel_farmhash_fingerprint64(s.As<unicode_view>());
}

Tuple kernel_farmhash_fingerprint128(const string_view& s);
Tuple kernel_farmhash_fingerprint128(const unicode_view& s);
MATXSCRIPT_ALWAYS_INLINE Tuple kernel_farmhash_fingerprint128(const Any& s) {
  return s.Is<string_view>() ? kernel_farmhash_fingerprint128(s.AsNoCheck<string_view>())
                             : kernel_farmhash_fingerprint128(s.As<unicode_view>());
}

/******************************************************************************
 * for fix overflow, some sugar
 *****************************************************************************/
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_hash64_mod(const string_view& s, int64_t y) {
  return py_builtins::fast_mod(kernel_farmhash_hash64(s), y);
}
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_hash64_mod(const unicode_view& s, int64_t y) {
  return py_builtins::fast_mod(kernel_farmhash_hash64(s), y);
}
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_hash64_mod(const Any& s, int64_t y) {
  return s.Is<string_view>() ? kernel_farmhash_hash64_mod(s.AsNoCheck<string_view>(), y)
                             : kernel_farmhash_hash64_mod(s.As<unicode_view>(), y);
}

MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_hash64withseed_mod(const string_view& s,
                                                                    uint64_t seed,
                                                                    int64_t y) {
  return py_builtins::fast_mod(kernel_farmhash_hash64withseed(s, seed), y);
}
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_hash64withseed_mod(const unicode_view& s,
                                                                    uint64_t seed,
                                                                    int64_t y) {
  return py_builtins::fast_mod(kernel_farmhash_hash64withseed(s, seed), y);
}
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_hash64withseed_mod(const Any& s,
                                                                    uint64_t seed,
                                                                    int64_t y) {
  return s.Is<string_view>()
             ? kernel_farmhash_hash64withseed_mod(s.AsNoCheck<string_view>(), seed, y)
             : kernel_farmhash_hash64withseed_mod(s.As<unicode_view>(), seed, y);
}

MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_fingerprint64_mod(const string_view& s,
                                                                   int64_t y) {
  return py_builtins::fast_mod(kernel_farmhash_fingerprint64(s), y);
}
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_fingerprint64_mod(const unicode_view& s,
                                                                   int64_t y) {
  return py_builtins::fast_mod(kernel_farmhash_fingerprint64(s), y);
}
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_fingerprint64_mod(const Any& s, int64_t y) {
  return s.Is<string_view>() ? kernel_farmhash_fingerprint64_mod(s.AsNoCheck<string_view>(), y)
                             : kernel_farmhash_fingerprint64_mod(s.As<unicode_view>(), y);
}

int64_t kernel_farmhash_fingerprint128_mod(const string_view& s, int64_t y);
int64_t kernel_farmhash_fingerprint128_mod(const unicode_view& s, int64_t y);
MATXSCRIPT_ALWAYS_INLINE int64_t kernel_farmhash_fingerprint128_mod(const Any& s, int64_t y) {
  return s.Is<string_view>() ? kernel_farmhash_fingerprint128_mod(s.AsNoCheck<string_view>(), y)
                             : kernel_farmhash_fingerprint128_mod(s.As<unicode_view>(), y);
}

}  // namespace runtime
}  // namespace matxscript
