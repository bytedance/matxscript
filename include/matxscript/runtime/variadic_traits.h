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

#include <utility>

namespace matxscript {
namespace runtime {

namespace variadic_details {

template <typename T>
struct func_signature_helper {
  using type = void;
  using return_type = void;
  static constexpr size_t num_args = 0;
  template <size_t i>
  struct arg {
    typedef void type;
  };
};

template <typename T, typename R, typename... Args>
struct func_signature_helper<R (T::*)(Args...)> {
  using type = R(Args...);
  using return_type = R;
  static constexpr size_t num_args = sizeof...(Args);
  template <size_t i>
  struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

template <typename T, typename R, typename... Args>
struct func_signature_helper<R (T::*)(Args...) const> {
  using type = R(Args...);
  using return_type = R;
  static constexpr size_t num_args = sizeof...(Args);
  template <size_t i>
  struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

template <typename T>
struct function_signature {
  using type = typename func_signature_helper<decltype(&T::operator())>::type;
  using return_type = typename func_signature_helper<decltype(&T::operator())>::return_type;
  static constexpr size_t num_args = func_signature_helper<decltype(&T::operator())>::num_args;
  template <size_t i>
  struct arg {
    typedef typename func_signature_helper<decltype(&T::operator())>::template arg<i>::type type;
  };
};

// handle case of function.
template <typename R, typename... Args>
struct function_signature<R(Args...)> {
  using type = R(Args...);
  using return_type = R;
  static constexpr size_t num_args = sizeof...(Args);
  template <size_t i>
  struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

// handle case of function ptr.
template <typename R, typename... Args>
struct function_signature<R (*)(Args...)> {
  using type = R(Args...);
  using return_type = R;
  static constexpr size_t num_args = sizeof...(Args);
  template <size_t i>
  struct arg {
    typedef typename std::tuple_element<i, std::tuple<Args...>>::type type;
  };
};

template <typename... Pack>
struct pack {};

template <typename, typename>
struct add_to_pack;

template <typename A, typename... R>
struct add_to_pack<A, pack<R...>> {
  typedef pack<A, R...> type;
};

template <typename A>
struct add_to_pack<A, void> {
  typedef pack<A> type;
};

template <typename, typename>
struct convert_to_function;

template <typename R, typename... Args>
struct convert_to_function<R, pack<Args...>> {
  using type = std::function<R(Args...)>;
};

template <size_t N, typename Head, typename Tail>
struct variadic_head_n;

template <size_t N, typename Head, typename TMP, typename... Tail>
struct variadic_head_n<N, Head, pack<TMP, Tail...>> {
  using type =
      typename variadic_head_n<N - 1, typename add_to_pack<TMP, Head>::type, pack<Tail...>>::type;
};

template <typename Head, typename TMP, typename... Tail>
struct variadic_head_n<0, Head, pack<TMP, Tail...>> {
  using type = Head;
};

template <typename, typename>
struct bind_default_args;

template <typename R, typename... Args>
struct bind_default_args<R, pack<Args...>> {
  using FLambdaParitial = typename convert_to_function<R, pack<Args...>>::type;
  template <size_t num_defaults, typename FLambda, typename TDefaults, size_t... N>
  static inline constexpr FLambdaParitial bind(FLambda f,
                                               TDefaults defaults,
                                               std::index_sequence<N...> unused) {
    return [f, defaults](Args&&... args) {
      return f(std::forward<Args>(args)...,
               std::get<N + num_defaults - sizeof...(N)>(TDefaults(defaults))...);
    };
  }
};

template <typename R>
struct bind_default_args<R, pack<>> {
  using FLambdaParitial = std::function<R()>;
  template <size_t num_defaults, typename FLambda, typename TDefaults, size_t... N>
  static inline constexpr FLambdaParitial bind(FLambda f,
                                               TDefaults defaults,
                                               std::index_sequence<N...> unused) {
    static_assert(num_defaults == sizeof...(N), "mismatch");
    return [f, defaults]() { return f(std::get<N>(TDefaults(defaults))...); };
  }
};

template <size_t NumInput, typename FLambda>
struct partial_function_helper;

template <size_t NumInput, typename R, typename... Args>
struct partial_function_helper<NumInput, R(Args...)> {
  static_assert(NumInput <= sizeof...(Args), "input arg num overflow");
  using FLambda = std::function<R(Args...)>;
  using pack_type = typename variadic_head_n<NumInput, void, pack<Args...>>::type;
  using FLambdaParitial = typename convert_to_function<R, pack_type>::type;

  template <size_t NumDefaults, typename TDefaults>
  static inline FLambdaParitial bind(FLambda f, TDefaults defaults) {
    static_assert(NumDefaults <= sizeof...(Args), "default args num overflow");
    return bind_default_args<R, pack_type>::template bind<NumDefaults>(
        f, defaults, std::make_index_sequence<sizeof...(Args) - NumInput>());
  }
};

template <typename R, typename... Args>
struct partial_function_helper<0, R(Args...)> {
  using FLambda = std::function<R(Args...)>;
  using FLambdaParitial = std::function<R()>;

  template <size_t NumDefaults, typename TDefaults>
  static inline FLambdaParitial bind(FLambda f, TDefaults defaults) {
    static_assert(NumDefaults == sizeof...(Args), "default args num not match");
    return bind_default_args<R, pack<>>::template bind<NumDefaults>(
        f, defaults, std::make_index_sequence<NumDefaults>());
  }
};

}  // namespace variadic_details
}  // namespace runtime
}  // namespace matxscript
