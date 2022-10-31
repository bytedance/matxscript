// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: The structure of the TypedNativeFunction is inspired by TVM.
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

#include <array>
#include <initializer_list>
#include <memory>

#include <matxscript/runtime/demangle.h>
#include <matxscript/runtime/function.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/variadic_traits.h>

namespace matxscript {
namespace runtime {

template <typename FType>
class TypedNativeFunction;

// for function
template <typename R, typename... Args>
class TypedNativeFunction<R(Args...)> {
 public:
  using TSelf = TypedNativeFunction<R(Args...)>;
  TypedNativeFunction() = default;
  TypedNativeFunction(std::nullptr_t) {
  }

  inline TypedNativeFunction(NativeFunction func) : native_func_(std::move(func)) {
  }

  template <typename FLambda,
            typename = typename std::enable_if<
                std::is_convertible<FLambda, std::function<R(Args...)>>::value>::type>
  TypedNativeFunction(const FLambda& typed_lambda,
                      String name = DemangleType(typeid(FLambda).name())) {  // NOLINT(*)
    function_name_ = std::move(name);
    this->AssignTypedLambda<FLambda>(typed_lambda);
  }

  template <typename FLambda,
            typename = typename std::enable_if<
                std::is_convertible<FLambda, std::function<R(Args...)>>::value>::type>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    function_name_ = DemangleType(typeid(FLambda).name());
    this->AssignTypedLambda<FLambda>(typed_lambda);
    return *this;
  }

  /*!
   * \brief copy assignment operator from PackedFunc.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(NativeFunction native_func) {
    function_name_ = DemangleType(typeid(native_func).name());
    native_func_ = std::move(native_func);
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  MATXSCRIPT_ALWAYS_INLINE R operator()(Args&&... args) const;

  /*!
   * \brief convert to PackedFunc
   * \return the internal PackedFunc
   */
  operator NativeFunction() const {
    return packed();
  }
  const NativeFunction& packed() const {
    return native_func_;
  }
  bool operator==(std::nullptr_t null) const {
    return native_func_ == nullptr;
  }
  bool operator!=(std::nullptr_t null) const {
    return native_func_ != nullptr;
  }

  template <typename... TDefaultArgs>
  inline void SetDefaultArgs(TDefaultArgs... def_args);

 private:
  friend class TRetValue;
  std::function<R(Args...)> raw_func_;
  NativeFunction native_func_no_default_;
  NativeFunction native_func_;
  String function_name_;
  /*!
   * \brief Assign the packed field using a typed lambda function.
   *
   * \param flambda The lambda function.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template <typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

namespace native_function_details {

template <typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(const F& f,
                                           PyArgs args_pack,
                                           RTValue* rv,
                                           Args&&... unpacked_args) {
    using FLambdaSig = variadic_details::function_signature<F>;
    using ARG_I_RAW_TYPE = typename FLambdaSig::template arg<index>::type;
    using ARG_I_TYPE =
        typename std::remove_cv<typename std::remove_reference<ARG_I_RAW_TYPE>::type>::type;
    using Converter = GenericValueConverter<ARG_I_TYPE>;
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>::run(
        f,
        args_pack,
        rv,
        std::forward<Args>(unpacked_args)...,
        args_pack[index].template As<ARG_I_TYPE>());
  }
};

template <typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(const F& f,
                                           PyArgs args_pack,
                                           RTValue* rv,
                                           Args&&... unpacked_args) {
    using RetType = decltype(f(std::forward<Args>(unpacked_args)...));
    if (std::is_same<RetType, R>::value) {
      *rv = f(std::forward<Args>(unpacked_args)...);
    } else {
      *rv = R(f(std::forward<Args>(unpacked_args)...));
    }
  }
};

template <int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(const F& f,
                                           PyArgs args_pack,
                                           RTValue* rv,
                                           Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template <typename R, int nargs, typename F>
MATXSCRIPT_ALWAYS_INLINE void unpack_call(const F& f, PyArgs args, RTValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, args, rv);
}

template <typename R>
struct typed_packed_call_dispatcher {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static R run(const NativeFunction& pf, Args&&... args) {
    std::initializer_list<RTView> args_pack{std::forward<Args>(args)...};
    return pf(PyArgs(args_pack)).template As<R>();
  }
};

template <>
struct typed_packed_call_dispatcher<void> {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(const NativeFunction& pf, Args&&... args) {
    std::initializer_list<RTView> args_pack{std::forward<Args>(args)...};
    pf(PyArgs(args_pack));
  }
};

template <typename FLambda, typename TDefaultArgs, size_t NumAllDefaults, size_t N>
inline NativeFunction gen_one_partial_functions_impl(FLambda func_lambda, TDefaultArgs&& defaults) {
  using FLambdaSig = variadic_details::function_signature<FLambda>;
  using FuncType = typename FLambdaSig::type;
  using ReturnType = typename FLambdaSig::return_type;
  return [func_lambda, defaults](PyArgs args) -> RTValue {
    constexpr size_t NumInput = FLambdaSig::num_args - N - 1;
    auto new_lambda = variadic_details::partial_function_helper<NumInput, FuncType>::template bind<
        NumAllDefaults>(func_lambda, defaults);
    RTValue ret;
    native_function_details::unpack_call<ReturnType, NumInput>(new_lambda, args, &ret);
    return ret;
  };
}

template <typename FLambda, typename TDefaultArgs, size_t... N>
inline std::array<NativeFunction, sizeof...(N)> gen_partial_functions_impl(
    FLambda func_lambda, TDefaultArgs&& defaults, std::index_sequence<N...> unused) {
  return {gen_one_partial_functions_impl<FLambda, TDefaultArgs, sizeof...(N), N>(
      func_lambda, std::forward<TDefaultArgs>(defaults))...};
}

template <typename FLambda>
struct partial_functions_maker;

template <typename R, typename... Args>
struct partial_functions_maker<R(Args...)> {
  using FLambda = std::function<R(Args...)>;
  template <size_t NUM_DEFAULTS, typename TDefaultArgs>
  static inline std::array<NativeFunction, NUM_DEFAULTS> run(FLambda func_lambda,
                                                             TDefaultArgs&& defaults) {
    return gen_partial_functions_impl(func_lambda,
                                      std::forward<TDefaultArgs>(defaults),
                                      std::make_index_sequence<NUM_DEFAULTS>());
  }
};

template <typename FLambda, typename... DefaultArgs>
inline std::array<NativeFunction, sizeof...(DefaultArgs)> gen_lambdas_with_defaults(
    FLambda func_lambda, DefaultArgs&&... defaults) {
  auto default_tuple_args = std::make_tuple(std::forward<DefaultArgs>(defaults)...);
  return partial_functions_maker<typename variadic_details::function_signature<FLambda>::type>::
      template run<sizeof...(defaults)>(func_lambda, std::move(default_tuple_args));
}

}  // namespace native_function_details

template <typename R, typename... Args>
template <typename FType>
inline void TypedNativeFunction<R(Args...)>::AssignTypedLambda(FType flambda) {
  raw_func_ = flambda;
  String func_name = function_name_;
  native_func_ = [func_name, flambda](PyArgs args) -> RTValue {
    MXCHECK_EQ(sizeof...(Args), args.size()) << "[" << func_name << "] Expect " << sizeof...(Args)
                                             << " arguments but get " << args.size();
    RTValue ret;
    native_function_details::unpack_call<R, sizeof...(Args)>(flambda, args, &ret);
    return ret;
  };
  native_func_no_default_ = native_func_;
}

template <typename R, typename... Args>
MATXSCRIPT_ALWAYS_INLINE R TypedNativeFunction<R(Args...)>::operator()(Args&&... args) const {
  return native_function_details::typed_packed_call_dispatcher<R>::run(native_func_,
                                                                       std::forward<Args>(args)...);
}

template <typename R, typename... Args>
template <typename... TDefaultArgs>
inline void TypedNativeFunction<R(Args...)>::SetDefaultArgs(TDefaultArgs... def_args) {
  auto funcs = native_function_details::gen_lambdas_with_defaults(
      raw_func_, std::forward<TDefaultArgs>(def_args)...);
  String func_name = function_name_;
  std::array<NativeFunction, sizeof...(def_args) + 1> all_funcs;
  all_funcs[0] = native_func_no_default_;
  for (size_t i = 0; i < sizeof...(def_args); ++i) {
    all_funcs[i + 1] = std::move(funcs[i]);
  }
  native_func_ = [func_name, all_funcs](PyArgs args) -> RTValue {
    constexpr size_t max_args = sizeof...(Args);
    constexpr size_t min_args = max_args - sizeof...(TDefaultArgs);
    MXCHECK(args.size() <= max_args && args.size() >= min_args)
        << "[" << func_name << "] Expect (" << min_args << ", " << max_args
        << ") arguments but get " << args.size();
    return all_funcs[sizeof...(Args) - args.size()](args);
  };
}

// for object method
template <typename R, typename... Args>
class TypedNativeFunction<R(void*, Args...)> {
 public:
  using TSelf = TypedNativeFunction<R(void*, Args...)>;
  TypedNativeFunction() = default;
  TypedNativeFunction(std::nullptr_t) {
  }

  inline TypedNativeFunction(NativeMethod func) : native_func_(std::move(func)) {
  }

  template <typename FLambda,
            typename = typename std::enable_if<
                std::is_convertible<FLambda, std::function<R(void*, Args...)>>::value>::type>
  TypedNativeFunction(const FLambda& typed_lambda,
                      String name = DemangleType(typeid(FLambda).name())) {  // NOLINT(*)
    function_name_ = std::move(name);
    this->AssignTypedLambda(typed_lambda);
  }

  template <typename FLambda,
            typename = typename std::enable_if<
                std::is_convertible<FLambda, std::function<R(void*, Args...)>>::value>::type>
  TSelf& operator=(FLambda typed_lambda) {  // NOLINT(*)
    function_name_ = DemangleType(typeid(FLambda).name());
    this->AssignTypedLambda(typed_lambda);
    return *this;
  }
  /*!
   * \brief copy assignment operator from PackedFunc.
   * \param packed The packed function.
   * \returns reference to self.
   */
  TSelf& operator=(NativeMethod native_func) {
    function_name_ = DemangleType(typeid(native_func).name());
    native_func_ = std::move(native_func);
    return *this;
  }
  /*!
   * \brief Invoke the operator.
   * \param args The arguments
   * \returns The return value.
   */
  MATXSCRIPT_ALWAYS_INLINE R operator()(void* self, Args&&... args) const;

  /*!
   * \brief convert to PackedFunc
   * \return the internal PackedFunc
   */
  operator NativeMethod() const {
    return packed();
  }
  const NativeMethod& packed() const {
    return native_func_;
  }
  bool operator==(std::nullptr_t null) const {
    return native_func_ == nullptr;
  }
  bool operator!=(std::nullptr_t null) const {
    return native_func_ != nullptr;
  }

  template <typename... TDefaultArgs>
  inline void SetDefaultArgs(TDefaultArgs... def_args);

 private:
  friend class TRetValue;
  /*! \brief The internal packed function */
  std::function<R(void*, Args...)> raw_func_;
  NativeMethod native_func_;
  NativeMethod native_func_no_default_;
  String function_name_;
  /*!
   * \brief Assign the packed field using a typed lambda function.
   *
   * \param flambda The lambda function.
   * \tparam FLambda The lambda function type.
   * \note We capture the lambda when possible for maximum efficiency.
   */
  template <typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

namespace native_method_details {

template <typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(
      const F& f, void* self, PyArgs args_pack, RTValue* rv, Args&&... unpacked_args) {
    // construct a movable argument value
    // which allows potential move of argument to the input of F.
    using FLambdaSig = variadic_details::function_signature<F>;
    using ARG_I_RAW_TYPE = typename FLambdaSig::template arg<index + 1>::type;
    using ARG_I_TYPE =
        typename std::remove_cv<typename std::remove_reference<ARG_I_RAW_TYPE>::type>::type;
    using Converter = GenericValueConverter<ARG_I_TYPE>;
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>::run(
        f,
        self,
        args_pack,
        rv,
        std::forward<Args>(unpacked_args)...,
        args_pack[index].template As<ARG_I_TYPE>());
  }
};

template <typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(
      const F& f, void* self, PyArgs args_pack, RTValue* rv, Args&&... unpacked_args) {
    using RetType = decltype(f(self, std::forward<Args>(unpacked_args)...));
    if (std::is_same<RetType, R>::value) {
      *rv = f(self, std::forward<Args>(unpacked_args)...);
    } else {
      *rv = R(f(self, std::forward<Args>(unpacked_args)...));
    }
  }
};

template <int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(
      const F& f, void* self, PyArgs args_pack, RTValue* rv, Args&&... unpacked_args) {
    f(self, std::forward<Args>(unpacked_args)...);
  }
};

template <typename R, int nargs, typename F>
MATXSCRIPT_ALWAYS_INLINE void unpack_call(const F& f, void* self, PyArgs args, RTValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, self, args, rv);
}

template <typename R>
struct typed_packed_call_dispatcher {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static R run(const NativeMethod& pf, void* self, Args&&... args) {
    std::initializer_list<RTView> args_pack{std::forward<Args>(args)...};
    return pf(self, PyArgs(args_pack)).template As<R>();
  }
};

template <>
struct typed_packed_call_dispatcher<void> {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static void run(const NativeMethod& pf, void* self, Args&&... args) {
    std::initializer_list<RTView> args_pack{std::forward<Args>(args)...};
    pf(self, PyArgs(args_pack));
  }
};

template <typename T>
struct func_signature_helper {
  using type = void;
};

template <typename T, typename R, typename... Args>
struct func_signature_helper<R (T::*)(void*, Args...)> {
  using type = R(void*, Args...);
};

template <typename T, typename R, typename... Args>
struct func_signature_helper<R (T::*)(void*, Args...) const> {
  using type = R(void*, Args...);
};

template <typename T>
struct function_signature {
  using type = typename func_signature_helper<decltype(&T::operator())>::type;
};

// handle case of function.
template <typename R, typename... Args>
struct function_signature<R(void*, Args...)> {
  using type = R(void*, Args...);
};

template <size_t NumAllDefaults, size_t N, typename R, typename... Args, typename TDefaultArgs>
inline NativeMethod gen_one_partial_functions_impl(std::function<R(void*, Args...)> func_lambda,
                                                   TDefaultArgs&& defaults) {
  return [func_lambda, defaults](void* self, PyArgs args) -> RTValue {
    constexpr size_t NumInput = sizeof...(Args) - N - 1;
    auto new_lambda =
        variadic_details::partial_function_helper<NumInput + 1, R(void*, Args...)>::template bind<
            NumAllDefaults>(func_lambda, defaults);
    RTValue ret;
    native_method_details::unpack_call<R, NumInput>(new_lambda, self, args, &ret);
    return ret;
  };
}

template <typename R, typename... Args, typename TDefaultArgs, size_t... N>
inline std::array<NativeMethod, sizeof...(N)> gen_partial_functions_impl(
    std::function<R(void*, Args...)> func_lambda,
    TDefaultArgs&& defaults,
    std::index_sequence<N...> unused) {
  return {gen_one_partial_functions_impl<sizeof...(N), N>(func_lambda,
                                                          std::forward<TDefaultArgs>(defaults))...};
}

template <typename FLambda>
struct partial_functions_maker;

template <typename R, typename... Args>
struct partial_functions_maker<R(void*, Args...)> {
  using FLambda = std::function<R(void*, Args...)>;
  template <size_t NUM_DEFAULTS, typename TDefaultArgs>
  static inline std::array<NativeMethod, NUM_DEFAULTS> run(FLambda func_lambda,
                                                           TDefaultArgs&& defaults) {
    return gen_partial_functions_impl(func_lambda,
                                      std::forward<TDefaultArgs>(defaults),
                                      std::make_index_sequence<NUM_DEFAULTS>());
  }
};

template <typename FLambda, typename... DefaultArgs>
inline std::array<NativeMethod, sizeof...(DefaultArgs)> gen_lambdas_with_defaults(
    FLambda func_lambda, DefaultArgs&&... defaults) {
  auto default_tuple_args = std::make_tuple(std::forward<DefaultArgs>(defaults)...);
  return partial_functions_maker<typename variadic_details::function_signature<FLambda>::type>::
      template run<sizeof...(defaults)>(func_lambda, std::move(default_tuple_args));
}

}  // namespace native_method_details

template <typename R, typename... Args>
template <typename FLambda>
inline void TypedNativeFunction<R(void*, Args...)>::AssignTypedLambda(FLambda flambda) {
  raw_func_ = flambda;
  String func_name = function_name_;
  native_func_ = [func_name, flambda](void* self, PyArgs args) -> RTValue {
    MXCHECK_EQ(sizeof...(Args), args.size()) << "[" << func_name << "] Expect " << sizeof...(Args)
                                             << " arguments but get " << args.size();
    RTValue ret;
    native_method_details::unpack_call<R, sizeof...(Args)>(flambda, self, args, &ret);
    return ret;
  };
  native_func_no_default_ = native_func_;
}

template <typename R, typename... Args>
MATXSCRIPT_ALWAYS_INLINE R
TypedNativeFunction<R(void*, Args...)>::operator()(void* self, Args&&... args) const {
  return native_method_details::typed_packed_call_dispatcher<R>::run(
      native_func_, self, std::forward<Args>(args)...);
}

template <typename R, typename... Args>
template <typename... TDefaultArgs>
inline void TypedNativeFunction<R(void*, Args...)>::SetDefaultArgs(TDefaultArgs... def_args) {
  auto funcs = native_method_details::gen_lambdas_with_defaults(
      raw_func_, std::forward<TDefaultArgs>(def_args)...);
  String func_name = function_name_;
  std::array<NativeMethod, sizeof...(def_args) + 1> all_funcs;
  all_funcs[0] = native_func_;
  for (size_t i = 0; i < sizeof...(def_args); ++i) {
    all_funcs[i + 1] = std::move(funcs[i]);
  }
  native_func_ = [func_name, all_funcs](void* self, PyArgs args) -> RTValue {
    constexpr size_t max_args = sizeof...(Args);
    constexpr size_t min_args = max_args - sizeof...(TDefaultArgs);
    MXCHECK(args.size() <= max_args && args.size() >= min_args)
        << "[" << func_name << "] Expect (" << min_args << ", " << max_args
        << ") arguments but get " << args.size();
    return all_funcs[sizeof...(Args) - args.size()](self, args);
  };
}

}  // namespace runtime
}  // namespace matxscript
