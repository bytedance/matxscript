// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
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

#include <type_traits>

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {
/******************************************************************************
 * Generic Constructor for python
 *****************************************************************************/
namespace Kernel_Iterable {
// iterable object
MATXSCRIPT_ALWAYS_INLINE File make(File obj) {
  return obj;
}

// iterator
MATXSCRIPT_ALWAYS_INLINE Iterator make(Iterator obj) {
  return obj;
}
// generic iterator
Iterator make(const Any& obj);
template <class T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
MATXSCRIPT_ALWAYS_INLINE Iterator make(const T& obj) {
  GenericValueConverter<RTView> Converter;
  return make(static_cast<const Any&>(Converter(obj)));
}
}  // namespace Kernel_Iterable

namespace Kernel_bool {
MATXSCRIPT_ALWAYS_INLINE constexpr bool make() noexcept {
  return 0;
}
MATXSCRIPT_ALWAYS_INLINE constexpr bool make(int32_t i32) noexcept {
  return static_cast<bool>(i32);
}
MATXSCRIPT_ALWAYS_INLINE constexpr bool make(int64_t i64) noexcept {
  return static_cast<bool>(i64);
}
MATXSCRIPT_ALWAYS_INLINE constexpr bool make(float d32) noexcept {
  return static_cast<bool>(d32);
}
MATXSCRIPT_ALWAYS_INLINE constexpr bool make(double d64) noexcept {
  return static_cast<bool>(d64);
}
MATXSCRIPT_ALWAYS_INLINE constexpr bool make(bool b) noexcept {
  return b;
}

template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
MATXSCRIPT_ALWAYS_INLINE bool make(const T& c) {
  return c.size() != 0;
}

bool make(const Any& c);
}  // namespace Kernel_bool

namespace Kernel_int64_t {
MATXSCRIPT_ALWAYS_INLINE int64_t make() {
  return 0;
}
MATXSCRIPT_ALWAYS_INLINE int64_t make(int32_t i32) {
  return static_cast<int64_t>(i32);
}
MATXSCRIPT_ALWAYS_INLINE int64_t make(int64_t i64) {
  return i64;
}
MATXSCRIPT_ALWAYS_INLINE int64_t make(float d32) {
  return static_cast<int64_t>(d32);
}
MATXSCRIPT_ALWAYS_INLINE int64_t make(double d64) {
  return static_cast<int64_t>(d64);
}
MATXSCRIPT_ALWAYS_INLINE int64_t make(bool b) {
  return static_cast<int64_t>(b);
}

int64_t make(const String& us, int64_t base = 10);
int64_t make(const Unicode& us, int64_t base = 10);
int64_t make(const Any& c, int64_t base = 10);
}  // namespace Kernel_int64_t

namespace Kernel_double {
MATXSCRIPT_ALWAYS_INLINE double make(int32_t i32) {
  return static_cast<double>(i32);
}
MATXSCRIPT_ALWAYS_INLINE double make(int64_t i64) {
  return static_cast<double>(i64);
}
MATXSCRIPT_ALWAYS_INLINE double make(float d32) {
  return static_cast<double>(d32);
}
MATXSCRIPT_ALWAYS_INLINE double make(double d64) {
  return d64;
}
MATXSCRIPT_ALWAYS_INLINE double make(bool b) {
  return static_cast<double>(b);
}

double make(const String& us);
double make(const Unicode& us);
double make(const Any& c);
}  // namespace Kernel_double

namespace Kernel_String {
MATXSCRIPT_ALWAYS_INLINE String make() {
  return String();
}
MATXSCRIPT_ALWAYS_INLINE String make(String s) {
  return s;
}
MATXSCRIPT_ALWAYS_INLINE String make(string_view s) {
  return String(s.data(), s.size());
}
MATXSCRIPT_ALWAYS_INLINE String make(const char* const s) {
  return String(string_view(s));
}
String make(const Unicode& us, const Unicode& encoding = U"UTF-8");
String make(const Any& c);
}  // namespace Kernel_String

namespace Kernel_Unicode {
MATXSCRIPT_ALWAYS_INLINE Unicode make() {
  return Unicode();
}
MATXSCRIPT_ALWAYS_INLINE Unicode make(unicode_view s) {
  return Unicode(s.data(), s.size());
}
MATXSCRIPT_ALWAYS_INLINE Unicode make(Unicode s) {
  return s;
}
MATXSCRIPT_ALWAYS_INLINE Unicode make(const char32_t* const s) {
  return Unicode(s);
}
Unicode make(int32_t i32);
MATXSCRIPT_ALWAYS_INLINE Unicode make(bool b) {
  static char32_t const repr[2][6] = {U"False", U"True\0"};
  return repr[b];
}
Unicode make(int64_t value);
Unicode make(double d64);
Unicode make(float d32);
Unicode make(const Any& c);
Unicode make(const IUserDataSharedViewRoot& c);
}  // namespace Kernel_Unicode

namespace Kernel_Dict {
MATXSCRIPT_ALWAYS_INLINE Dict make() {
  return Dict{};
}
MATXSCRIPT_ALWAYS_INLINE Dict make(std::initializer_list<Dict::value_type> init_args) {
  return Dict(init_args);
}
Dict make(const Dict& c);
Dict make(const Any& c);
}  // namespace Kernel_Dict

namespace Kernel_Tuple {
MATXSCRIPT_ALWAYS_INLINE Tuple make() {
  return Tuple();
}
MATXSCRIPT_ALWAYS_INLINE Tuple make(std::initializer_list<RTValue> init_args) {
  return Tuple(init_args);
}
}  // namespace Kernel_Tuple

namespace Kernel_List {
MATXSCRIPT_ALWAYS_INLINE List make() {
  return List();
}
MATXSCRIPT_ALWAYS_INLINE List make(std::initializer_list<RTValue> init_args) {
  return List(init_args);
}
MATXSCRIPT_ALWAYS_INLINE List make(const List& c) {
  return List(c.begin(), c.end());
}
MATXSCRIPT_ALWAYS_INLINE List make(const Set& c) {
  return List(c.begin(), c.end());
}
List make(const Iterator& c);
List make(const Any& c);
}  // namespace Kernel_List

namespace Kernel_Set {
MATXSCRIPT_ALWAYS_INLINE Set make() {
  return Set();
}
MATXSCRIPT_ALWAYS_INLINE Set make(std::initializer_list<RTValue> init_args) {
  return Set(init_args);
}
MATXSCRIPT_ALWAYS_INLINE Set make(const Set& c) {
  return Set(c.begin(), c.end());
}
MATXSCRIPT_ALWAYS_INLINE Set make(const List& c) {
  return Set(c.begin(), c.end());
}
Set make(const Iterator& c);
Set make(const Any& c);
}  // namespace Kernel_Set

namespace Kernel_NDArray {
NDArray make(const Any& list,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str = U"cpu");
NDArray make(const List& list,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str = U"cpu");
NDArray make(const FTList<int64_t>& list,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str = U"cpu");
NDArray make(const FTList<double>& list,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str = U"cpu");
NDArray make(int64_t scalar,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str = U"cpu");
NDArray make(double scalar,
             const List& shape,
             const Unicode& dtype_str,
             const Unicode& ctx_str = U"cpu");
}  // namespace Kernel_NDArray

namespace Kernel_Trie {
MATXSCRIPT_ALWAYS_INLINE Trie make() {
  return Trie();
}
Trie make(const Dict& d);
}  // namespace Kernel_Trie

namespace Kernel_Regex {
template <typename... ARGS>
MATXSCRIPT_ALWAYS_INLINE Regex make(ARGS&&... args) {
  return Regex(std::forward<ARGS>(args)...);
}
}  // namespace Kernel_Regex

namespace Kernel_OpaqueObject {
template <typename... ARGS>
MATXSCRIPT_ALWAYS_INLINE OpaqueObject make(ARGS&&... args) {
  return OpaqueObject(std::forward<ARGS>(args)...);
}
}  // namespace Kernel_OpaqueObject

}  // namespace runtime
}  // namespace matxscript
