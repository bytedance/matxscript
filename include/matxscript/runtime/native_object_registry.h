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

#include <memory>

#include <matxscript/runtime/container/string_view.h>
#include <matxscript/runtime/demangle.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/type_name_traits.h>
#include <matxscript/runtime/typed_native_function.h>

namespace matxscript {
namespace runtime {

namespace constructor_details {
template <typename R, int nleft, int index, typename Constructor>
struct unpack_call_dispatcher {
  using FLambdaSig = variadic_details::function_signature<Constructor>;
  using ARG_I_RAW_TYPE = typename FLambdaSig::template arg<index>::type;
  using ARG_I_TYPE =
      typename std::remove_cv<typename std::remove_reference<ARG_I_RAW_TYPE>::type>::type;
  using Converter = GenericValueConverter<ARG_I_TYPE>;
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static R run(const Constructor& body,
                                        PyArgs args_pack,
                                        Args&&... unpacked_args) {
    return unpack_call_dispatcher<R, nleft - 1, index + 1, Constructor>::run(
        body,
        args_pack,
        std::forward<Args>(unpacked_args)...,
        args_pack[index].template As<ARG_I_TYPE>());
  }
};

template <typename R, int index, typename Constructor>
struct unpack_call_dispatcher<R, 0, index, Constructor> {
  template <typename... Args>
  MATXSCRIPT_ALWAYS_INLINE static R run(const Constructor& body,
                                        PyArgs args_pack,
                                        Args&&... unpacked_args) {
    return body(std::forward<Args>(unpacked_args)...);
  }
};

template <typename R, int nargs, typename Constructor>
MATXSCRIPT_ALWAYS_INLINE R unpack_call(const Constructor& body, PyArgs args) {
  return unpack_call_dispatcher<R, nargs, 0, Constructor>::run(body, args);
}

template <typename FLambda>
struct ClassConstructor;

template <typename ClassType, typename... Args>
struct ClassConstructor<std::shared_ptr<ClassType>(Args...)> {
  using FLambda = std::function<std::shared_ptr<ClassType>(Args...)>;

  MATXSCRIPT_ALWAYS_INLINE static std::shared_ptr<ClassType> make(const FLambda& body,
                                                                  string_view class_name,
                                                                  PyArgs args) {
    if (args.size() != sizeof...(Args)) {
      std::initializer_list<String> arg_names{DemangleType(typeid(Args).name())...};
      auto arg_name_repr = StringHelper::JoinStringList(", ", arg_names);
      MXTHROW << "[" << class_name << "::" << class_name << "(" << arg_name_repr << ")] Expect "
              << sizeof...(Args) << " arguments but get " << args.size();
    }
    return unpack_call<std::shared_ptr<ClassType>, sizeof...(Args), FLambda>(body, args);
  }
};

template <typename ClassType, typename... Args>
struct ClassConstructor<ClassType(Args...)> {
  using FLambda = std::function<std::shared_ptr<ClassType>(Args...)>;

  MATXSCRIPT_ALWAYS_INLINE static std::shared_ptr<ClassType> make(string_view class_name,
                                                                  PyArgs args) {
    const FLambda& body = [](Args&&... args_init) -> std::shared_ptr<ClassType> {
      return std::make_shared<ClassType>(std::forward<Args>(args_init)...);
    };
    if (args.size() != sizeof...(Args)) {
      std::initializer_list<String> arg_names{DemangleType(typeid(Args).name())...};
      auto arg_name_repr = StringHelper::JoinStringList(", ", arg_names);
      MXTHROW << "[" << class_name << "::" << class_name << "(" << arg_name_repr << ")] Expect "
              << sizeof...(Args) << " arguments but get " << args.size();
    }
    return unpack_call<std::shared_ptr<ClassType>, sizeof...(Args), FLambda>(body, args);
  }
};

}  // namespace constructor_details

class OpKernel;
class JitObject;

class NativeObjectRegistry {
 public:
  using NativeObjectConstructor = std::function<std::shared_ptr<void>(PyArgs args)>;
  using NativeMethod = std::function<RTValue(void* self, PyArgs args)>;

 public:
  // constructor
  NativeObjectConstructor construct;
  // function table is unbound
  ska::flat_hash_map<string_view, NativeMethod> function_table_;
  // is native op
  bool is_native_op_ = false;
  // is jit object
  bool is_jit_object_ = false;
  // threadsafety
  bool threadsafety_ = true;
  // class name
  string_view class_name;

  // object document
  string_view __doc__;

  std::type_index type_id_ = typeid(void);

  MATX_DLL static NativeObjectRegistry& Register(string_view name, bool override = false);
  MATX_DLL static bool Remove(string_view name);
  MATX_DLL static NativeObjectRegistry* Get(string_view name);
  MATX_DLL static std::vector<string_view> ListNames();
  MATX_DLL static std::vector<string_view> ListPureObjNames();

  // register function
  NativeObjectRegistry& RegisterFunction(string_view name, NativeMethod func) {
    MXCHECK(!function_table_.count(name))
        << "Class: " << class_name << " Function: \"" << name << "\" is already registered";
    function_table_.emplace(name, std::move(func));
    return *this;
  }

  template <
      typename FLambda,
      typename... TDefaultArgs,
      typename = typename std::enable_if<!std::is_convertible<FLambda, NativeMethod>::value>::type>
  NativeObjectRegistry& def(string_view name, FLambda func, TDefaultArgs&&... defaults) {
    using FLambdaSignature = typename variadic_details::function_signature<FLambda>;
    static_assert(!std::is_reference<typename FLambdaSignature::return_type>::value,
                  "NativeObject method return reference");
    String func_name = String(class_name) + "::" + name;
    TypedNativeFunction<typename FLambdaSignature::type> tnf(std::move(func), std::move(func_name));
    if (sizeof...(TDefaultArgs) > 0) {
      tnf.SetDefaultArgs(std::forward<TDefaultArgs>(defaults)...);
    }
    MXCHECK(!function_table_.count(name))
        << "Class: " << class_name << " Function: \"" << name << "\" is already registered";
    function_table_.emplace(name, tnf.packed());
    return *this;
  }

  NativeObjectRegistry& def(string_view name, NativeMethod func) {
    return RegisterFunction(name, std::move(func));
  }

  NativeObjectRegistry& doc(string_view doc) {
    __doc__ = doc;
    return *this;
  }

  template <typename FLambda,
            typename = typename std::enable_if<
                !std::is_convertible<FLambda, NativeObjectConstructor>::value>::type>
  NativeObjectRegistry& SetConstructor(const FLambda& body) {
    using FLambdaSig = typename variadic_details::function_signature<FLambda>;
    if (!(std::is_same<typename FLambdaSig::return_type, std::shared_ptr<void>>::value ||
          type_id_ == typeid(std::shared_ptr<typename FLambdaSig::return_type>))) {
      MXTHROW << "MATX_REGISTER_NATIVE_OBJECT(" << class_name << ") mismatch, expect '"
              << DemangleType(type_id_.name()) << "', but get '"
              << DemangleType(typeid(typename FLambdaSig::return_type).name()) << "'";
    }
    construct = [body, this](PyArgs args) -> std::shared_ptr<void> {
      return constructor_details::ClassConstructor<typename FLambdaSig::type>::make(
          body, class_name, args);
    };
    return *this;
  }

  template <typename FLambda,
            typename = typename std::enable_if<std::is_class<
                typename variadic_details::function_signature<FLambda>::return_type>::value>::type>
  NativeObjectRegistry& SetConstructor() {
    using FLambdaSig = typename variadic_details::function_signature<FLambda>;
    if (type_id_ != typeid(typename FLambdaSig::return_type)) {
      MXTHROW << "MATX_REGISTER_NATIVE_OBJECT(" << class_name << ") mismatch, expect '"
              << DemangleType(type_id_.name()) << "', but get '"
              << DemangleType(typeid(typename FLambdaSig::return_type).name()) << "'";
    }
    construct = [this](PyArgs args) -> std::shared_ptr<void> {
      return constructor_details::ClassConstructor<typename FLambdaSig::type>::make(class_name,
                                                                                    args);
    };
    return *this;
  }

  inline NativeObjectRegistry& SetConstructor(NativeObjectConstructor body) {
    construct = std::move(body);
    return *this;
  }

  template <class U>
  inline NativeObjectRegistry& SetIsNativeOp() {
#ifdef MATXSCRIPT_RUNTIME_PIPELINE_OP_KERNEL_H
    is_native_op_ = std::is_base_of<OpKernel, U>::value;
#else
    is_native_op_ = false;
#endif
    return *this;
  }

  template <class U>
  inline NativeObjectRegistry& SetIsJitObject() {
#ifdef MATXSCRIPT_RUNTIME_JIT_OBJECT_H
    is_jit_object_ = std::is_same<JitObject, U>::value;
#else
    is_jit_object_ = false;
#endif
    return *this;
  }

  inline NativeObjectRegistry& SetClassName(string_view cls_name) {
    class_name = cls_name;
    return *this;
  }
  inline NativeObjectRegistry& SetThreadSafety(bool state) {
    threadsafety_ = state;
    return *this;
  }

  inline NativeObjectRegistry& SetTypeId(std::type_index idx) {
    type_id_ = idx;
    return *this;
  }

  // Internal class.
  struct Manager;

 protected:
  friend struct Manager;
};

#define MATX_NATIVE_OBJECT_VAR_DEF(ClassName) \
  static MATXSCRIPT_ATTRIBUTE_UNUSED auto& __make_##MATX_NATIVE_OBJECT##ClassName

#define MATX_REGISTER_NATIVE_OBJECT(ClassName)                                \
  MATXSCRIPT_REGISTER_TYPE_NAME_TRAITS(ClassName);                            \
  MATXSCRIPT_STR_CONCAT(MATX_NATIVE_OBJECT_VAR_DEF(ClassName), __COUNTER__) = \
      ::matxscript::runtime::NativeObjectRegistry::Register(#ClassName)       \
          .SetIsNativeOp<ClassName>()                                         \
          .SetIsJitObject<ClassName>()                                        \
          .SetClassName(#ClassName)                                           \
          .SetTypeId(typeid(ClassName))

}  // namespace runtime
}  // namespace matxscript
