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
#include <matxscript/runtime/native_func_maker.h>

#include <matxscript/runtime/container/native_func_private.h>
#include <matxscript/runtime/container/user_data_interface.h>
#include <matxscript/runtime/container/user_data_private.h>
#include <matxscript/runtime/container/user_data_ref.h>
#include <matxscript/runtime/py_args.h>
#include <matxscript/runtime/registry.h>
#include <matxscript/runtime/runtime_value.h>

namespace matxscript {
namespace runtime {

namespace {
void* creater(void* data) {
  return new (data) NativeFuncUserData();
}

void deleter(ILightUserData* data) {
  ((NativeFuncUserData*)(data))->~NativeFuncUserData();
}
}  // namespace

MATX_DLL UserDataRef make_native_function(string_view func_name) {
  auto native_function_register = FunctionRegistry::Get(func_name);
  MXCHECK(native_function_register != nullptr) << "Native Function not found: " << func_name;
  auto ret = UserDataRef(0, 0, sizeof(NativeFuncUserData), creater, deleter, nullptr);
  ((NativeFuncUserData*)(ret->ud_ptr))->__call__ = native_function_register;
  return ret;
}

MATX_DLL RTValue call_native_function(string_view func_name, PyArgs args) {
  auto native_function_register = FunctionRegistry::Get(func_name);
  MXCHECK(native_function_register != nullptr) << "Native Function not found: " << func_name;
  return (*native_function_register)(args);
}

MATXSCRIPT_REGISTER_GLOBAL("native.call_native_function").set_body([](PyArgs args) -> RTValue {
  MXCHECK(args.size() >= 1);
  auto function_name = args[0].As<string_view>();
  return call_native_function(function_name, PyArgs(args.begin() + 1, args.size() - 1));
});

}  // namespace runtime
}  // namespace matxscript