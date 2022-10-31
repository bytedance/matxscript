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

#include <matxscript/runtime/pypi/kernel_farmhash.h>

#include <iostream>
#include "farmhash.h"

namespace matxscript {
namespace runtime {

uint32_t kernel_farmhash_hash32(const string_view& s) {
  return Hash32(s.data(), s.size());
}
uint32_t kernel_farmhash_hash32(const unicode_view& s) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return Hash32(utf8_bytes.data(), utf8_bytes.size());
}

uint32_t kernel_farmhash_hash32withseed(const string_view& s, uint32_t seed) {
  return Hash32WithSeed(s.data(), s.size(), seed);
}
uint32_t kernel_farmhash_hash32withseed(const unicode_view& s, uint32_t seed) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return Hash32WithSeed(utf8_bytes.data(), utf8_bytes.size(), seed);
}

uint64_t kernel_farmhash_hash64(const string_view& s) {
  return Hash64(s.data(), s.size());
}
uint64_t kernel_farmhash_hash64(const unicode_view& s) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return Hash64(utf8_bytes.data(), utf8_bytes.size());
}

uint64_t kernel_farmhash_hash64withseed(const string_view& s, uint64_t seed) {
  return Hash64WithSeed(s.data(), s.size(), seed);
}
uint64_t kernel_farmhash_hash64withseed(const unicode_view& s, uint64_t seed) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return Hash64WithSeed(utf8_bytes.data(), utf8_bytes.size(), seed);
}

Tuple kernel_farmhash_hash128(const string_view& s) {
  uint128_t h = Hash128(s.data(), s.size());
  uint64_t low64 = Uint128Low64(h);
  uint64_t high64 = Uint128High64(h);
  return Tuple::dynamic(RTValue(low64), RTValue(high64));
}
Tuple kernel_farmhash_hash128(const unicode_view& s) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return kernel_farmhash_hash128(utf8_bytes);
}

Tuple kernel_farmhash_hash128withseed(const string_view& s,
                                      uint64_t seedlow64,
                                      uint64_t seedhigh64) {
  uint128_t seed = Uint128(seedlow64, seedhigh64);
  uint128_t h = Hash128WithSeed(s.data(), s.size(), seed);
  uint64_t low64 = Uint128Low64(h);
  uint64_t high64 = Uint128High64(h);
  return Tuple::dynamic(low64, high64);
}
Tuple kernel_farmhash_hash128withseed(const unicode_view& s,
                                      uint64_t seedlow64,
                                      uint64_t seedhigh64) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return kernel_farmhash_hash128withseed(utf8_bytes, seedlow64, seedhigh64);
}

uint32_t kernel_farmhash_fingerprint32(const string_view& s) {
  return Fingerprint32(s.data(), s.size());
}
uint32_t kernel_farmhash_fingerprint32(const unicode_view& s) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return Fingerprint32(utf8_bytes.data(), utf8_bytes.size());
}

uint64_t kernel_farmhash_fingerprint64(const string_view& s) {
  return Fingerprint64(s.data(), s.size());
}

uint64_t kernel_farmhash_fingerprint64(const unicode_view& s) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return Fingerprint64(utf8_bytes.data(), utf8_bytes.size());
}

Tuple kernel_farmhash_fingerprint128(const string_view& s) {
  uint128_t h = Fingerprint128(s.data(), s.size());
  uint64_t low64 = Uint128Low64(h);
  uint64_t high64 = Uint128High64(h);
  return Tuple::dynamic(low64, high64);
}
Tuple kernel_farmhash_fingerprint128(const unicode_view& s) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return kernel_farmhash_fingerprint128(utf8_bytes);
}

/******************************************************************************
 * for fix overflow, some sugar
 *****************************************************************************/
int64_t kernel_farmhash_fingerprint128_mod(const string_view& s, int64_t y) {
  uint128_t h = Fingerprint128(s.data(), s.size());
  uint64_t low64 = Uint128Low64(h);
  uint64_t high64 = Uint128High64(h);
  return py_builtins::fast_mod(low64, y);
}
int64_t kernel_farmhash_fingerprint128_mod(const unicode_view& s, int64_t y) {
  auto utf8_bytes = UnicodeHelper::Encode(s);
  return kernel_farmhash_fingerprint128_mod(utf8_bytes, y);
}

}  // namespace runtime
}  // namespace matxscript
