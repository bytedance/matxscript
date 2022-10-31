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

#include <matxscript/runtime/_almost_equal.h>
#include <matxscript/runtime/_is_comparable.h>
#include <matxscript/runtime/builtins_modules/_floatobject.h>
#include <matxscript/runtime/builtins_modules/_longobject.h>
#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container/string_helper.h>
#include <matxscript/runtime/container/unicode_helper.h>
#include <matxscript/runtime/exceptions/exceptions.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/generic/generic_constructor_funcs.h>
#include <matxscript/runtime/object.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

// Compared to the virtual function implemented in an object,
// the distribution here will not increase the object bytes

/******************************************************************************
 * Arithmetic logical operators
 *****************************************************************************/
struct ArithOps {
  /******************************************************************************
   * Generic add
   *****************************************************************************/
  // double
  static inline double add(double lhs, const Any& rhs) {
    return lhs + rhs.template As<double>();
  }
  static inline double add(float lhs, const Any& rhs) {
    return lhs + rhs.template As<double>();
  }
  static inline double add(const Any& lhs, double rhs) {
    return lhs.template As<double>() + rhs;
  }
  static inline double add(const Any& lhs, float rhs) {
    return lhs.template As<double>() + rhs;
  }
  // int64_t
  static inline RTValue add(int64_t lhs, const Any& rhs) {
    switch (rhs.type_code()) {
      case TypeIndex::kRuntimeInteger: {
        return static_cast<int64_t>(lhs + rhs.value_.data.v_int64);
      } break;
      case TypeIndex::kRuntimeFloat: {
        return static_cast<double>(lhs + rhs.value_.data.v_float64);
      } break;
      default: {
        THROW_PY_TypeError("unsupported operand type(s) for +: 'int' and ", rhs.type_name(), "'");
      } break;
    }
    return 0;
  }
  static inline RTValue add(int32_t lhs, const Any& rhs) {
    return add(int64_t(lhs), rhs);
  }
  static inline RTValue add(const Any& lhs, int64_t rhs) {
    return add(rhs, lhs);
  }
  static inline RTValue add(const Any& lhs, int32_t rhs) {
    return add(int64_t(rhs), lhs);
  }
  // bytes
  static inline String add(const string_view& lhs, const string_view& rhs) {
    return StringHelper::Concat(lhs, rhs);
  }
  static inline String add(const string_view& lhs, const Any& rhs) {
    if (!rhs.Is<string_view>()) {
      THROW_PY_TypeError("unsupported operand type(s) for +: 'bytes' and '", rhs.type_name(), "'");
    }
    return StringHelper::Concat(lhs, rhs.AsNoCheck<string_view>());
  }
  static inline String add(const Any& lhs, const string_view& rhs) {
    if (!lhs.Is<string_view>()) {
      THROW_PY_TypeError("unsupported operand type(s) for +: '", lhs.type_name(), "' and 'bytes'");
    }
    return StringHelper::Concat(lhs.AsNoCheck<string_view>(), rhs);
  }
  // unicode
  static inline Unicode add(const unicode_view& lhs, const unicode_view& rhs) {
    return UnicodeHelper::Concat(lhs, rhs);
  }
  static inline Unicode add(const unicode_view& lhs, const Any& rhs) {
    if (!rhs.Is<unicode_view>()) {
      THROW_PY_TypeError("unsupported operand type(s) for +: 'str' and '", rhs.type_name(), "'");
    }
    return UnicodeHelper::Concat(lhs, rhs.AsNoCheck<unicode_view>());
  }
  static inline Unicode add(const Any& lhs, const unicode_view& rhs) {
    if (!lhs.Is<unicode_view>()) {
      THROW_PY_TypeError("unsupported operand type(s) for +: '", lhs.type_name(), "' and 'str'");
    }
    return UnicodeHelper::Concat(lhs.AsNoCheck<unicode_view>(), rhs);
  }

  // List or FTList
  template <typename LIterator, typename RIterator>
  static inline List general_list_concat(LIterator lhs_b,
                                         LIterator lhs_e,
                                         RIterator rhs_b,
                                         RIterator rhs_e) {
    List result;
    result.reserve((lhs_e - lhs_b) + (rhs_e - rhs_b));
    for (auto itr = lhs_b; itr != lhs_e; ++itr) {
      result.template append(*itr);
    }
    for (auto itr = rhs_b; itr != rhs_e; ++itr) {
      result.template append(*itr);
    }
    return result;
  }
  template <typename L, typename R>
  static inline List general_list_concat(L lhs, R rhs) {
    if (lhs.use_count() == 1 && rhs.use_count() == 1) {
      return general_list_concat(std::make_move_iterator(lhs.begin()),
                                 std::make_move_iterator(lhs.end()),
                                 std::make_move_iterator(rhs.begin()),
                                 std::make_move_iterator(rhs.end()));
    } else if (rhs.use_count() == 1) {
      return general_list_concat(lhs.begin(),
                                 lhs.end(),
                                 std::make_move_iterator(rhs.begin()),
                                 std::make_move_iterator(rhs.end()));
    } else {
      return general_list_concat(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }
  }

  template <typename U>
  static inline FTList<U> add(FTList<U> lhs, FTList<U> rhs) {
    return FTList<U>::Concat(std::move(lhs), std::move(rhs));
  }

  template <typename L,
            typename R,
            typename = typename std::enable_if<!std::is_same<L, R>::value>::type>
  static inline List add(FTList<L> lhs, FTList<R> rhs) {
    return general_list_concat(std::move(lhs), std::move(rhs));
  }

  template <typename U>
  static inline List add(List lhs, FTList<U> rhs) {
    return general_list_concat(std::move(lhs), std::move(rhs));
  }

  template <typename U>
  static inline List add(FTList<U> lhs, List rhs) {
    return general_list_concat(std::move(lhs), std::move(rhs));
  }

  template <typename U>
  static inline List add(FTList<U> lhs, const Any& rhs) {
    if (rhs.type_code() == TypeIndex::kRuntimeList) {
      return general_list_concat(std::move(lhs), rhs.template AsObjectViewNoCheck<List>().data());
    } else if (rhs.type_code() == TypeIndex::kRuntimeFTList) {
      if (rhs.template Is<FTList<U>>()) {
        return general_list_concat(std::move(lhs),
                                   rhs.template AsObjectViewNoCheck<FTList<U>>().data());
      } else {
        List result;
        result.extend(Kernel_Iterable::make(lhs));
        result.extend(Kernel_Iterable::make(rhs));
        return result;
      }
    } else {
      THROW_PY_TypeError("can only concatenate list (not \"", rhs.type_name(), "\") to list");
      return List{};
    }
  }

  template <typename U>
  static inline List add(const Any& lhs, FTList<U> rhs) {
    if (lhs.type_code() == TypeIndex::kRuntimeList) {
      return general_list_concat(lhs.template AsObjectViewNoCheck<List>().data(), std::move(rhs));
    } else if (lhs.type_code() == TypeIndex::kRuntimeFTList) {
      if (lhs.template Is<FTList<U>>()) {
        return general_list_concat(lhs.template AsObjectViewNoCheck<FTList<U>>().data(),
                                   std::move(rhs));
      } else {
        List result;
        result.extend(Kernel_Iterable::make(lhs));
        result.extend(Kernel_Iterable::make(rhs));
        return result;
      }
    } else {
      THROW_PY_TypeError("can only concatenate list (not \"", lhs.type_name(), "\") to list");
      return List{};
    }
  }

  static inline List add(List lhs, List rhs) {
    return List::Concat(std::move(lhs), std::move(rhs));
  }
  static inline List add(const Any& lhs, List rhs) {
    if (lhs.type_code() == TypeIndex::kRuntimeList) {
      return general_list_concat(lhs.template AsObjectViewNoCheck<List>().data(), std::move(rhs));
    } else if (lhs.type_code() == TypeIndex::kRuntimeFTList) {
      List result;
      result.extend(Kernel_Iterable::make(lhs));
      result.extend(std::move(rhs));
      return result;
    } else {
      THROW_PY_TypeError("can only concatenate list (not \"", lhs.type_name(), "\") to list");
      return List{};
    }
  }
  static inline List add(List lhs, const Any& rhs) {
    if (rhs.type_code() == TypeIndex::kRuntimeList) {
      return general_list_concat(std::move(lhs), rhs.template AsObjectViewNoCheck<List>().data());
    } else if (rhs.type_code() == TypeIndex::kRuntimeFTList) {
      List result;
      result.extend(std::move(lhs));
      result.extend(Kernel_Iterable::make(rhs));
      return result;
    } else {
      THROW_PY_TypeError("can only concatenate list (not \"", rhs.type_name(), "\") to list");
      return List{};
    }
  }

  // Tuple
  static inline Tuple add(Tuple lhs, Tuple rhs) {
    return Tuple::Concat(std::move(lhs), std::move(rhs));
  }
  static inline Tuple add(const Any& lhs, Tuple rhs) {
    if (lhs.type_code() != TypeIndex::kRuntimeTuple) {
      THROW_PY_TypeError(
          "TypeError: can only concatenate tuple (not \"", lhs.type_name(), "\") to tuple");
    }
    return Tuple::Concat(lhs.template AsObjectViewNoCheck<Tuple>().data(), std::move(rhs));
  }
  static inline Tuple add(Tuple lhs, const Any& rhs) {
    if (rhs.type_code() != TypeIndex::kRuntimeTuple) {
      THROW_PY_TypeError(
          "TypeError: can only concatenate tuple (not \"", rhs.type_name(), "\") to tuple");
    }
    return Tuple::Concat(std::move(lhs), rhs.template AsObjectViewNoCheck<Tuple>().data());
  }

  // Any + Any
  static RTValue add(const Any& lhs, const Any& rhs);

  /******************************************************************************
   * Generic mul
   *****************************************************************************/
  // double/float mul only accept double or int
  template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
  static inline double mul(T lhs, const Any& rhs) {
    switch (rhs.type_code()) {
      case TypeIndex::kRuntimeInteger: {
        return lhs * rhs.value_.data.v_int64;
      } break;
      case TypeIndex::kRuntimeFloat: {
        return lhs * rhs.value_.data.v_float64;
      } break;
      default: {
        THROW_PY_TypeError(
            "unsupported operand type(s) for *: 'float' and '", rhs.type_name(), "'");
        return 0.0;
      } break;
    }
  }
  template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
  static inline double mul(const Any& lhs, T rhs) {
    switch (lhs.type_code()) {
      case TypeIndex::kRuntimeInteger: {
        return lhs.value_.data.v_int64 * rhs;
      } break;
      case TypeIndex::kRuntimeFloat: {
        return lhs.value_.data.v_float64 * rhs;
      } break;
      default: {
        THROW_PY_TypeError(
            "unsupported operand type(s) for *: '", lhs.type_name(), "' and 'float'");
        return 0.0;
      } break;
    }
  }
  // List/String/Unicode * integer
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline String mul(string_view lhs, T rhs) {
    return StringHelper::Repeat(lhs, static_cast<int64_t>(rhs));
  }
  static inline String mul(string_view lhs, const Any& rhs) {
    if (!rhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", rhs.type_name(), "'");
    }
    return StringHelper::Repeat(lhs, rhs.template AsNoCheck<int64_t>());
  }
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline Unicode mul(unicode_view lhs, T rhs) {
    return UnicodeHelper::Repeat(lhs, static_cast<int64_t>(rhs));
  }
  static inline Unicode mul(unicode_view lhs, const Any& rhs) {
    if (!rhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", rhs.type_name(), "'");
    }
    return UnicodeHelper::Repeat(lhs, rhs.template AsNoCheck<int64_t>());
  }
  template <typename U,
            typename T,
            typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline FTList<U> mul(const FTList<U>& lhs, T rhs) {
    return lhs.repeat(static_cast<int64_t>(rhs));
  }
  template <typename U>
  static inline FTList<U> mul(const FTList<U>& lhs, const Any& rhs) {
    if (!rhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", rhs.type_name(), "'");
    }
    return lhs.repeat(rhs.template AsNoCheck<int64_t>());
  }
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline List mul(const List& lhs, T rhs) {
    return lhs.repeat(static_cast<int64_t>(rhs));
  }
  static inline List mul(const List& lhs, const Any& rhs) {
    if (!rhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", rhs.type_name(), "'");
    }
    return lhs.repeat(rhs.template AsNoCheck<int64_t>());
  }
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline Tuple mul(const Tuple& lhs, T rhs) {
    return lhs.repeat(static_cast<int64_t>(rhs));
  }
  static inline Tuple mul(const Tuple& lhs, const Any& rhs) {
    if (!rhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", rhs.type_name(), "'");
    }
    return lhs.repeat(rhs.template AsNoCheck<int64_t>());
  }
  // integer * List/String/Unicode
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline String mul(T lhs, string_view rhs) {
    return StringHelper::Repeat(rhs, static_cast<int64_t>(lhs));
  }
  static inline String mul(const Any& lhs, string_view rhs) {
    if (!lhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", lhs.type_name(), "'");
    }
    return StringHelper::Repeat(rhs, lhs.template AsNoCheck<int64_t>());
  }
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline Unicode mul(T lhs, unicode_view rhs) {
    return UnicodeHelper::Repeat(rhs, static_cast<int64_t>(lhs));
  }
  static inline Unicode mul(const Any& lhs, unicode_view rhs) {
    if (!lhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", lhs.type_name(), "'");
    }
    return UnicodeHelper::Repeat(rhs, lhs.template AsNoCheck<int64_t>());
  }
  template <typename U,
            typename T,
            typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline FTList<U> mul(T lhs, const FTList<U>& rhs) {
    return rhs.repeat(static_cast<int64_t>(lhs));
  }
  template <typename U>
  static inline FTList<U> mul(const Any& lhs, const FTList<U>& rhs) {
    if (!lhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", lhs.type_name(), "'");
    }
    return rhs.repeat(lhs.template AsNoCheck<int64_t>());
  }
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline List mul(T lhs, const List& rhs) {
    return rhs.repeat(static_cast<int64_t>(lhs));
  }
  static inline List mul(const Any& lhs, const List& rhs) {
    if (!lhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", lhs.type_name(), "'");
    }
    return rhs.repeat(lhs.template AsNoCheck<int64_t>());
  }
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline Tuple mul(T lhs, const Tuple& rhs) {
    return rhs.repeat(static_cast<int64_t>(lhs));
  }
  static inline Tuple mul(const Any& lhs, const Tuple& rhs) {
    if (!lhs.Is<int64_t>()) {
      THROW_PY_TypeError("can't multiply sequence by non-int of type '", lhs.type_name(), "'");
    }
    return rhs.repeat(lhs.template AsNoCheck<int64_t>());
  }
  // integer * Any
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline RTValue mul(T lhs, const Any& rhs) {
    switch (rhs.type_code()) {
      case TypeIndex::kRuntimeInteger: {
        return static_cast<int64_t>(lhs) * rhs.value_.data.v_int64;
      } break;
      case TypeIndex::kRuntimeFloat: {
        return static_cast<int64_t>(lhs) * rhs.value_.data.v_float64;
      } break;
      case TypeIndex::kRuntimeString: {
        return mul(rhs.template As<string_view>(), static_cast<int64_t>(lhs));
      } break;
      case TypeIndex::kRuntimeUnicode: {
        return mul(rhs.template As<unicode_view>(), static_cast<int64_t>(lhs));
      } break;
      case TypeIndex::kRuntimeList: {
        return mul(static_cast<int64_t>(lhs), rhs.template AsObjectViewNoCheck<List>().data());
      } break;
      case TypeIndex::kRuntimeFTList: {
        return rhs.template AsObjectViewNoCheck<FTObjectBase>().data().generic_call_attr(
            "__mul__", {RTView(lhs)});
      } break;
      case TypeIndex::kRuntimeTuple: {
        return mul(static_cast<int64_t>(lhs), rhs.template AsObjectViewNoCheck<Tuple>().data());
      } break;
      default: {
        THROW_PY_TypeError("unsupported operand type(s) for *: 'int' and '", rhs.type_name(), "'");
      } break;
    }
    return 0;
  }
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  static inline RTValue mul(const Any& lhs, T rhs) {
    return mul<T>(rhs, lhs);
  }
  // Any * Any
  static RTValue mul(const Any& lhs, const Any& rhs);

  /******************************************************************************
   * Generic sub
   *****************************************************************************/
  static inline double sub(double lhs, const Any& rhs) {
    switch (rhs.type_code()) {
      case TypeIndex::kRuntimeInteger: {
        return lhs - rhs.value_.data.v_int64;
      } break;
      case TypeIndex::kRuntimeFloat: {
        return lhs - rhs.value_.data.v_float64;
      } break;
      default: {
        THROW_PY_TypeError(
            "unsupported operand type(s) for -: 'float' and '", rhs.type_name(), "'");
        return 0.0;
      } break;
    }
  }
  static inline double sub(float lhs, const Any& rhs) {
    return sub(static_cast<double>(lhs), rhs);
  }
  static inline double sub(const Any& lhs, double rhs) {
    switch (lhs.type_code()) {
      case TypeIndex::kRuntimeInteger: {
        return lhs.value_.data.v_int64 - rhs;
      } break;
      case TypeIndex::kRuntimeFloat: {
        return lhs.value_.data.v_float64 - rhs;
      } break;
      default: {
        THROW_PY_TypeError(
            "unsupported operand type(s) for -: '", lhs.type_name(), "' and 'float'");
        return 0.0;
      } break;
    }
  }
  static inline double sub(const Any& lhs, float rhs) {
    return sub(lhs, static_cast<double>(rhs));
  }
  static RTValue sub(int64_t lhs, const Any& rhs);
  static inline RTValue sub(int32_t lhs, const Any& rhs) {
    return sub(static_cast<int64_t>(lhs), rhs);
  }
  static RTValue sub(const Any& lhs, int64_t rhs);
  static inline RTValue sub(const Any& lhs, int32_t rhs) {
    return sub(lhs, static_cast<int64_t>(rhs));
  }
  static RTValue sub(const Any& lhs, const Any& rhs);

  /******************************************************************************
   * Generic abs
   *****************************************************************************/
  static RTValue abs(const Any& x);

  /******************************************************************************
   * Generic div
   *****************************************************************************/
  static inline double div(double lhs, double rhs) {
    double result = py_builtins::float_div(lhs, rhs);
    if (std::isnan(result) || std::isinf(result)) {
      THROW_PY_ValueError("math domain error");
    }
    return result;
  }

  /******************************************************************************
   * Generic floordiv
   *****************************************************************************/
  static inline double floordiv(double lhs, double rhs) {
    double result = py_builtins::float_floor_div(lhs, rhs);
    if (std::isnan(result) || std::isinf(result)) {
      THROW_PY_ValueError("math domain error");
    }
    return result;
  }
  static inline double floordiv(double lhs, const Any& rhs) {
    return floordiv(lhs, rhs.As<double>());
  }
  static inline double floordiv(const Any& lhs, double rhs) {
    return floordiv(lhs.As<double>(), rhs);
  }
  static inline int64_t floordiv(int64_t lhs, int64_t rhs) {
    int64_t result = py_builtins::fast_floor_div(lhs, rhs);
    if (std::isnan(result) || std::isinf(result)) {
      THROW_PY_ValueError("math domain error");
    }
    return result;
  }
  static RTValue floordiv(int64_t lhs, const Any& rhs);
  static RTValue floordiv(const Any& lhs, int64_t rhs);
  static RTValue floordiv(const Any& lhs, const Any& rhs);

  /******************************************************************************
   * Generic floormod
   *****************************************************************************/
  static inline double floormod(double lhs, double rhs) {
    double result = py_builtins::float_rem(lhs, rhs);
    if (std::isnan(result) || std::isinf(result)) {
      THROW_PY_ValueError("math domain error");
    }
    return result;
  }
  static inline double floormod(double lhs, const Any& rhs) {
    return floormod(lhs, rhs.As<double>());
  }
  static inline double floormod(const Any& lhs, double rhs) {
    return floormod(lhs.As<double>(), rhs);
  }
  static inline int64_t floormod(int64_t lhs, int64_t rhs) {
    auto r = py_builtins::fast_mod(lhs, rhs);
    if (std::isnan(r) || std::isinf(r)) {
      THROW_PY_ValueError("math domain error");
    }
    return r;
  }
  static RTValue floormod(int64_t lhs, const Any& rhs);
  static RTValue floormod(const Any& lhs, int64_t rhs);
  static RTValue floormod(const Any& lhs, const Any& rhs);

  /******************************************************************************
   * Generic and/or
   *****************************************************************************/
  template <class T>
  struct is_view_type {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    static constexpr bool value = std::is_same<TYPE, unicode_view>::value ||
                                  std::is_same<TYPE, string_view>::value ||
                                  std::is_same<TYPE, RTView>::value;
    using type = typename std::conditional<value, std::true_type, std::false_type>::type;
  };
  template <class T>
  struct is_str_or_str_view_type {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    static constexpr bool value =
        std::is_same<TYPE, Unicode>::value || std::is_same<TYPE, String>::value ||
        std::is_same<TYPE, unicode_view>::value || std::is_same<TYPE, string_view>::value;
    using type = typename std::conditional<value, std::true_type, std::false_type>::type;
  };
  template <typename T1,
            typename T2,
            typename = typename std::enable_if<!(is_str_or_str_view_type<T1>::value &&
                                                 is_str_or_str_view_type<T2>::value)>::type>
  static inline RTValue And(const T1& lhs, const T2& rhs) {
    return Kernel_bool::make(lhs) ? RTValue(rhs) : RTValue(lhs);
  }

  template <typename T, typename = typename std::enable_if<!is_view_type<T>::value>::type>
  static inline T And(const T& lhs, const T& rhs) {
    return Kernel_bool::make(lhs) ? rhs : lhs;
  }
  static inline Unicode And(const unicode_view& lhs, const unicode_view& rhs) {
    return (!lhs.empty()) ? Unicode(rhs) : Unicode(lhs);
  }
  static inline String And(const string_view& lhs, const string_view& rhs) {
    return (!lhs.empty()) ? String(rhs) : String(lhs);
  }

  template <typename T1,
            typename T2,
            typename = typename std::enable_if<!(is_str_or_str_view_type<T1>::value &&
                                                 is_str_or_str_view_type<T2>::value)>::type>
  static inline RTValue Or(const T1& lhs, const T2& rhs) {
    return Kernel_bool::make(lhs) ? RTValue(lhs) : RTValue(rhs);
  }

  template <typename T, typename = typename std::enable_if<!is_view_type<T>::value>::type>
  static inline T Or(const T& lhs, const T& rhs) {
    return Kernel_bool::make(lhs) ? lhs : rhs;
  }
  static inline String Or(const string_view& lhs, const string_view& rhs) {
    return (!lhs.empty()) ? String(lhs) : String(rhs);
  }
  static inline Unicode Or(const unicode_view& lhs, const unicode_view& rhs) {
    return (!lhs.empty()) ? Unicode(lhs) : Unicode(rhs);
  }

  /******************************************************************************
   * Generic logic op
   *****************************************************************************/
  // equal
  template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
  static inline bool eq(const T& lhs, const Any& rhs) {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return rhs.Is<TYPE>() && lhs == rhs.AsNoCheck<TYPE>();
  }
  // template <typename T, typename = typename std::enable_if<std::is_base_of<T,
  // FTObjectBase>::value>::type> static inline bool eq(const T& lhs, const Any& rhs) {
  //   return lhs.__eq__(rhs);
  // }
  template <typename U>
  static inline bool eq(const FTList<U>& lhs, const Any& rhs) {
    return lhs.__eq__(rhs);
  }
  template <typename U>
  static inline bool eq(const FTSet<U>& lhs, const Any& rhs) {
    return lhs.__eq__(rhs);
  }
  template <typename K, typename V>
  static inline bool eq(const FTDict<K, V>& lhs, const Any& rhs) {
    return lhs.__eq__(rhs);
  }
  static inline bool eq(double lhs, const Any& rhs) {
    return (rhs.Is<double>() || rhs.Is<int64_t>()) &&
           floating_point::AlmostEquals(lhs, rhs.AsNoCheck<double>());
  }
  static inline bool eq(int64_t lhs, const Any& rhs) {
    return (rhs.Is<int64_t>() && lhs == rhs.AsNoCheck<int64_t>()) ||
           (rhs.Is<double>() &&
            floating_point::AlmostEquals(static_cast<double>(lhs), rhs.AsNoCheck<double>()));
  }
  static inline bool eq(const String& lhs, const Any& rhs) {
    return rhs.Is<String>() && lhs.view() == rhs.AsNoCheck<string_view>();
  }
  static inline bool eq(const Unicode& lhs, const Any& rhs) {
    return rhs.Is<Unicode>() && lhs.view() == rhs.AsNoCheck<unicode_view>();
  }

  template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
  static inline bool eq(const Any& lhs, const T& rhs) {
    return eq(rhs, lhs);
  }
  static bool eq(const Any& lhs, const Any& rhs);

  template <typename U1, typename U2>
  static inline bool eq(const FTList<U1>& lhs, const FTList<U2>& rhs) {
    return lhs == rhs;
  }
  template <typename U>
  static inline bool eq(const FTList<U>& lhs, const List& rhs) {
    return lhs.__eq__(RTView(rhs));
  }
  template <typename U>
  static inline bool eq(const List& lhs, const FTList<U>& rhs) {
    return eq(rhs, lhs);
  }
  template <typename U1, typename U2>
  static inline bool eq(const FTSet<U1>& lhs, const FTSet<U2>& rhs) {
    return lhs == rhs;
  }
  template <typename U>
  static inline bool eq(const FTSet<U>& lhs, const Set& rhs) {
    return lhs.__eq__(RTView(rhs));
  }
  template <typename U>
  static inline bool eq(const Set& lhs, const FTSet<U>& rhs) {
    return eq(rhs, lhs);
  }
  template <typename K1, typename V1, typename K2, typename V2>
  static inline bool eq(const FTDict<K1, V1>& lhs, const FTDict<K2, V2>& rhs) {
    return lhs == rhs;
  }
  template <typename K, typename V>
  static inline bool eq(const FTDict<K, V>& lhs, const Dict& rhs) {
    return lhs.__eq__(RTView(rhs));
  }
  template <typename K, typename V>
  static inline bool eq(const Dict& lhs, const FTDict<K, V>& rhs) {
    return eq(rhs, lhs);
  }

  // ne
  template <typename LEFT_TYPE, typename RIGHT_TYPE>
  static inline bool ne(LEFT_TYPE&& lhs, RIGHT_TYPE&& rhs) {
    return !eq(std::forward<LEFT_TYPE>(lhs), std::forward<RIGHT_TYPE>(rhs));
  }

  // gt
  template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
  static inline bool gt(const T& lhs, const Any& rhs) {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return rhs.Is<TYPE>() && lhs > rhs.AsNoCheck<TYPE>();
  }
  static inline bool gt(double lhs, const Any& rhs) {
    if (rhs.Is<double>() || rhs.Is<int64_t>()) {
      return lhs > rhs.AsNoCheck<double>();
    } else {
      THROW_PY_TypeError(
          "'>' not supported between instances of 'float' and '", rhs.type_name(), "'");
    }
  }
  static inline bool gt(int64_t lhs, const Any& rhs) {
    if (rhs.Is<double>()) {
      return lhs > rhs.AsNoCheck<double>();
    } else if (rhs.Is<int64_t>()) {
      return lhs > rhs.AsNoCheck<int64_t>();
    } else {
      THROW_PY_TypeError(
          "'>' not supported between instances of 'int' and '", rhs.type_name(), "'");
    }
  }
  static inline bool gt(const String& lhs, const Any& rhs) {
    return lhs.view() > rhs.As<string_view>();
  }
  static inline bool gt(const Unicode& lhs, const Any& rhs) {
    return lhs.view() > rhs.As<unicode_view>();
  }

  template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
  static inline bool gt(const Any& lhs, const T& rhs) {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return lhs.Is<TYPE>() && lhs.AsNoCheck<TYPE>() > rhs;
  }
  static inline bool gt(const Any& lhs, double rhs) {
    if (lhs.Is<double>() || lhs.Is<int64_t>()) {
      return lhs.AsNoCheck<double>() > rhs;
    } else {
      THROW_PY_TypeError(
          "'>' not supported between instances of '", lhs.type_name(), "' and 'float'");
    }
  }
  static inline bool gt(const Any& lhs, int64_t rhs) {
    if (lhs.Is<double>()) {
      return lhs.AsNoCheck<double>() > rhs;
    } else if (lhs.Is<int64_t>()) {
      return lhs.AsNoCheck<int64_t>() > rhs;
    } else {
      THROW_PY_TypeError(
          "'>' not supported between instances of '", lhs.type_name(), "' and 'int'");
    }
  }
  static inline bool gt(const Any& lhs, const String& rhs) {
    return lhs.As<string_view>() > rhs.view();
  }
  static inline bool gt(const Any& lhs, const Unicode& rhs) {
    return lhs.As<unicode_view>() > rhs.view();
  }

  static bool gt(const Any& lhs, const Any& rhs);

  // ge
  template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
  static inline bool ge(const T& lhs, const Any& rhs) {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return rhs.Is<TYPE>() && lhs >= rhs.AsNoCheck<TYPE>();
  }
  static inline bool ge(double lhs, const Any& rhs) {
    if (rhs.Is<double>() || rhs.Is<int64_t>()) {
      double r = rhs.AsNoCheck<double>();
      return floating_point::AlmostEquals(lhs, r) || lhs > r;
    } else {
      THROW_PY_TypeError(
          "'>=' not supported between instances of 'float' and '", rhs.type_name(), "'");
    }
  }
  static inline bool ge(int64_t lhs, const Any& rhs) {
    if (rhs.Is<double>()) {
      double r = rhs.AsNoCheck<double>();
      return floating_point::AlmostEquals(static_cast<double>(lhs), r) || lhs > r;
    } else if (rhs.Is<int64_t>()) {
      return lhs >= rhs.AsNoCheck<int64_t>();
    } else {
      THROW_PY_TypeError(
          "'>=' not supported between instances of 'int' and '", rhs.type_name(), "'");
    }
  }
  static inline bool ge(const String& lhs, const Any& rhs) {
    return lhs.view() >= rhs.As<string_view>();
  }
  static inline bool ge(const Unicode& lhs, const Any& rhs) {
    return lhs.view() >= rhs.As<unicode_view>();
  }

  template <typename T, typename = typename std::enable_if<!is_runtime_value<T>::value>::type>
  static inline bool ge(const Any& lhs, const T& rhs) {
    using TYPE = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    return lhs.Is<TYPE>() && lhs.AsNoCheck<TYPE>() >= rhs;
  }
  static inline bool ge(const Any& lhs, double rhs) {
    if (lhs.Is<double>() || lhs.Is<int64_t>()) {
      double l = lhs.AsNoCheck<double>();
      return floating_point::AlmostEquals(l, rhs) || l > rhs;
    } else {
      THROW_PY_TypeError(
          "'>=' not supported between instances of '", lhs.type_name(), "' and 'float'");
    }
  }
  static inline bool ge(const Any& lhs, int64_t rhs) {
    if (lhs.Is<double>()) {
      double l = lhs.AsNoCheck<double>();
      return floating_point::AlmostEquals(l, static_cast<double>(rhs)) || l > rhs;
    } else if (lhs.Is<int64_t>()) {
      return lhs.AsNoCheck<int64_t>() >= rhs;
    } else {
      THROW_PY_TypeError(
          "'>=' not supported between instances of '", lhs.type_name(), "' and 'int'");
    }
  }
  static inline bool ge(const Any& lhs, const String& rhs) {
    return lhs.As<string_view>() >= rhs.view();
  }
  static inline bool ge(const Any& lhs, const Unicode& rhs) {
    return lhs.As<unicode_view>() >= rhs.view();
  }

  static bool ge(const Any& lhs, const Any& rhs);

  // lt
  template <typename LEFT_TYPE, typename RIGHT_TYPE>
  static inline bool lt(LEFT_TYPE&& lhs, RIGHT_TYPE&& rhs) {
    return !ge(std::forward<LEFT_TYPE>(lhs), std::forward<RIGHT_TYPE>(rhs));
  }

  // le
  template <typename LEFT_TYPE, typename RIGHT_TYPE>
  static inline bool le(LEFT_TYPE&& lhs, RIGHT_TYPE&& rhs) {
    return !gt(std::forward<LEFT_TYPE>(lhs), std::forward<RIGHT_TYPE>(rhs));
  }
};

/******************************************************************************
 * wrapped builtin math func
 *****************************************************************************/

/******************************************************************************
 * template for math funcs with fixed signature
 *****************************************************************************/
template <typename FuncType>
class Math;
template <typename R, typename... Args>
struct Math<R(Args...)> {
  using FuncType = R(Args...);
  template <FuncType func>
  static inline R check_call(Args... args) {
    auto r = func(std::forward<Args>(args)...);
    MXCHECK(!std::isnan(r) && !std::isinf(r)) << "ValueError: math domain error";
    return r;
  }
};

}  // namespace runtime
}  // namespace matxscript
