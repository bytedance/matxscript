// Copyright 2022 ByteDance Ltd. and/or its affiliates.
/*
 * Acknowledgement: This file originates from incubator-tvm.
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

/*!
 * \file matx/runtime/registry.h
 * \brief This file defines the global function registry.
 *
 *  The registered functions will be made available to front-end
 *  as well as backend users.
 *
 *  The registry stores type-erased functions.
 *  Each registered function is automatically exposed
 *  to front-end language(e.g. python).
 *
 *  Front-end can also pass callbacks as PackedFunc, or register
 *  then into the same global registry in C++.
 *  The goal is to mix the front-end language and the MATXScript back-end.
 *
 * \code
 *   // register the function as MyAPIFuncName
 *   MATXSCRIPT_REGISTER_GLOBAL(MyAPIFuncName)
 *   .set_body([](PyArgs args) -> RTValue {
 *     // my code.
 *   });
 * \endcode
 */

#include <string>
#include <utility>
#include <vector>

#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/typed_native_function.h>

namespace matxscript {
namespace runtime {

class FunctionRegistry {
 public:
  using NativeFunction = std::function<RTValue(PyArgs args)>;

 public:
  NativeFunction function_;
  string_view type_name_;

  // object document
  string_view __doc__;
  // TODO: rename ?
  bool __is_native__ = false;

  MATX_DLL static FunctionRegistry& Register(string_view name, bool override = false);
  MATX_DLL static bool Remove(string_view name);
  MATX_DLL static NativeFunction* Get(string_view name);
  MATX_DLL static FunctionRegistry* GetRegistry(string_view name);
  MATX_DLL static std::vector<string_view> ListNames();

  template <typename FLambda,
            typename... TDefaultArgs,
            typename =
                typename std::enable_if<!std::is_convertible<FLambda, NativeFunction>::value>::type>
  FunctionRegistry& def(FLambda func, TDefaultArgs&&... defaults) {
    using FLambdaSignature = typename variadic_details::function_signature<FLambda>;
    static_assert(!std::is_reference<typename FLambdaSignature::return_type>::value,
                  "NativeObject method return reference");
    TypedNativeFunction<typename FLambdaSignature::type> tnf(std::move(func), type_name_);
    if (sizeof...(TDefaultArgs) > 0) {
      tnf.SetDefaultArgs(std::forward<TDefaultArgs>(defaults)...);
    }
    function_ = tnf.packed();
    return *this;
  }

  template <typename FLambda>
  FunctionRegistry& set_body_typed(FLambda func) {
    using FLambdaSignature = typename variadic_details::function_signature<FLambda>;
    return def(
        TypedNativeFunction<typename FLambdaSignature::type>(std::move(func), type_name_).packed());
  }

  FunctionRegistry& def(NativeFunction func) {
    function_ = std::move(func);
    return *this;
  }

  // TODO: remove this ?
  FunctionRegistry& set_body(NativeFunction func) {
    return def(std::move(func));
  }

  FunctionRegistry& doc(string_view doc) {
    __doc__ = doc;
    return *this;
  }

  FunctionRegistry& SetIsNative(bool is_native) {
    __is_native__ = is_native;
    return *this;
  }

  inline FunctionRegistry& SetFuncName(string_view cls_name) {
    type_name_ = cls_name;
    return *this;
  }

  // Internal class.
  struct Manager;

 protected:
  friend struct Manager;
};

#define MATXSCRIPT_FUNCTION_VAR_DEF(Func) \
  static MATXSCRIPT_ATTRIBUTE_UNUSED auto& __make_##MATXSCRIPT_FUNCTION##Func

#define MATX_REGISTER_NATIVE_FUNC(Func)                                   \
  MATXSCRIPT_STR_CONCAT(MATXSCRIPT_FUNCTION_VAR_DEF(Func), __COUNTER__) = \
      ::matxscript::runtime::FunctionRegistry::Register(#Func)            \
          .SetIsNative(true)                                              \
          .SetFuncName(#Func)                                             \
          .def(Func)

#define MATX_REGISTER_NATIVE_NAMED_FUNC(Name, Func)                       \
  MATXSCRIPT_STR_CONCAT(MATXSCRIPT_FUNCTION_VAR_DEF(Func), __COUNTER__) = \
      ::matxscript::runtime::FunctionRegistry::Register(Name)             \
          .SetIsNative(true)                                              \
          .SetFuncName(Name)                                              \
          .def(Func)

#define MATXSCRIPT_REGISTER_GLOBAL(OpName)                                  \
  MATXSCRIPT_STR_CONCAT(MATXSCRIPT_FUNCTION_VAR_DEF(GLOBAL), __COUNTER__) = \
      ::matxscript::runtime::FunctionRegistry::Register(OpName).SetFuncName(OpName)

}  // namespace runtime
}  // namespace matxscript
