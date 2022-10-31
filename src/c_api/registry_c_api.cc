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

#include <matxscript/runtime/container.h>
#include <matxscript/runtime/container_private.h>
#include <matxscript/runtime/ft_container.h>
#include <matxscript/runtime/native_func_maker.h>
#include <matxscript/runtime/native_object_maker.h>
#include <matxscript/runtime/native_object_registry.h>
#include <matxscript/runtime/registry.h>

namespace matxscript {
namespace runtime {

/******************************************************************************
 * Native Functions
 *****************************************************************************/

MATXSCRIPT_REGISTER_GLOBAL("native.Func_Exist").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[native.CheckExist] Expect 1 arguments but get " << args.size();
  String cls_name = args[0].As<String>();
  return FunctionRegistry::Get(cls_name) != nullptr;
});

MATXSCRIPT_REGISTER_GLOBAL("native.Func_ListNames").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 0) << "[native.ListNames] Expect 0 arguments but get " << args.size();
  auto names = FunctionRegistry::ListNames();
  List result;
  for (auto& name : names) {
    auto* reg = FunctionRegistry::GetRegistry(name);
    if (reg->__is_native__) {
      result.append(String(name.data(), name.size()).decode());
    }
  }
  return result;
});

MATXSCRIPT_REGISTER_GLOBAL("native.Func_Get").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[native.Func_Get] Expect 1 arguments but get " << args.size();
  String func_name = args[0].As<String>();
  return make_native_function(func_name.view());
});

MATXSCRIPT_REGISTER_GLOBAL("native.Func_Call").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 1) << "[native.Func_Call] Expect 1 or more arguments but get "
                             << args.size();
  UserDataRef ud_ref = args[0].As<UserDataRef>();
  std::vector<RTValue> ctor_args;
  for (size_t i = 1; i < args.size(); ++i) {
    ctor_args.emplace_back(args[i].As<RTValue>());
  }
  return ud_ref.generic_call(PyArgs(ctor_args.data(), ctor_args.size()));
});

/******************************************************************************
 * Native Objects
 *****************************************************************************/

MATXSCRIPT_REGISTER_GLOBAL("native.Exist").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[native.CheckExist] Expect 1 arguments but get " << args.size();
  String cls_name = args[0].As<String>();
  return NativeObjectRegistry::Get(cls_name) != nullptr;
});

MATXSCRIPT_REGISTER_GLOBAL("native.ListNames").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 0) << "[native.ListNames] Expect 0 arguments but get " << args.size();
  auto names = NativeObjectRegistry::ListNames();
  List result;
  for (auto& name : names) {
    result.append(String(name.data(), name.size()).decode());
  }
  return result;
});

MATXSCRIPT_REGISTER_GLOBAL("native.ListPureObjNames").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 0) << "[native.ListPureObjNames] Expect 0 arguments but get "
                             << args.size();
  auto names = NativeObjectRegistry::ListPureObjNames();
  List result;
  for (auto& name : names) {
    result.append(String(name.data(), name.size()).decode());
  }
  return result;
});

MATXSCRIPT_REGISTER_GLOBAL("native.GetFunctionTable").set_body([](PyArgs args) -> RTValue {
  MXCHECK_EQ(args.size(), 1) << "[native.GetFunctionTable] Expect 1 or more arguments but get "
                             << args.size();
  String cls_name = args[0].As<String>();
  auto native_user_data_register = NativeObjectRegistry::Get(cls_name);
  MXCHECK(native_user_data_register != nullptr) << "Native class not found: " << cls_name;
  List result;
  for (auto& fn_pair : native_user_data_register->function_table_) {
    result.append(String(fn_pair.first.data(), fn_pair.first.size()).decode());
  }
  return result;
});

MATXSCRIPT_REGISTER_GLOBAL("native.CreateNativeObject").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 1) << "[native.CreateNativeObject] Expect 1 or more arguments but get "
                             << args.size();
  String cls_name = args[0].As<String>();
  std::vector<RTValue> ctor_args;
  for (size_t i = 1; i < args.size(); ++i) {
    ctor_args.emplace_back(args[i].As<RTValue>());
  }
  return make_native_userdata(cls_name, PyArgs(ctor_args.data(), ctor_args.size()));
});

MATXSCRIPT_REGISTER_GLOBAL("native.ClassNameIsNativeOp").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 1) << "[native.ClassNameIsNativeOp] Expect 1 or more arguments but get "
                             << args.size();
  String cls_name = args[0].As<String>();
  auto native_user_data_register = NativeObjectRegistry::Get(cls_name);
  if (native_user_data_register) {
    return native_user_data_register->is_native_op_;
  } else {
    return false;
  }
});

MATXSCRIPT_REGISTER_GLOBAL("native.ClassNameIsJitObject").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 1) << "[native.ClassNameIsJitObject] Expect 1 or more arguments but get "
                             << args.size();
  String cls_name = args[0].As<String>();
  auto native_user_data_register = NativeObjectRegistry::Get(cls_name);
  if (native_user_data_register) {
    return native_user_data_register->is_jit_object_;
  } else {
    return false;
  }
});

MATXSCRIPT_REGISTER_GLOBAL("native.NativeObject_IsNativeOp").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 1)
      << "[native.NativeObject_IsNativeOp] Expect 1 or more arguments but get " << args.size();
  UserDataRef ud_ref = args[0].As<UserDataRef>();
  if (ud_ref->ud_ptr->type_2_71828182846() != UserDataStructType::kNativeData) {
    return false;
  } else {
    NativeObject* nud_ptr = dynamic_cast<NativeObject*>(ud_ref->ud_ptr);
    return nud_ptr->is_native_op_;
  }
});

MATXSCRIPT_REGISTER_GLOBAL("native.NativeObject_IsJitObject").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 1)
      << "[native.NativeObject_IsJitObject] Expect 1 or more arguments but get " << args.size();
  UserDataRef ud_ref = args[0].As<UserDataRef>();
  if (ud_ref->ud_ptr->type_2_71828182846() != UserDataStructType::kNativeData) {
    return false;
  } else {
    NativeObject* nud_ptr = dynamic_cast<NativeObject*>(ud_ref->ud_ptr);
    return nud_ptr->is_jit_object_;
  }
});

MATXSCRIPT_REGISTER_GLOBAL("native.NativeObject_Call").set_body([](PyArgs args) -> RTValue {
  MXCHECK_GE(args.size(), 2) << "[native.NativeObject_Call] Expect 2 or more arguments but get "
                             << args.size();
  UserDataRef ud_ref = args[0].As<UserDataRef>();
  String func_name = args[1].As<String>();
  std::vector<RTValue> ctor_args;
  for (size_t i = 2; i < args.size(); ++i) {
    ctor_args.emplace_back(args[i].As<RTValue>());
  }
  return ud_ref.generic_call_attr(func_name, PyArgs(ctor_args.data(), ctor_args.size()));
});

}  // namespace runtime
}  // namespace matxscript
